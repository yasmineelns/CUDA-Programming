/**
 * gpu_utils.cuh — Shared utilities for deterministic and stochastic GPU kernels
 *
 * Contents
 * --------
 *  - CUDA_CHECK error-handling macro
 *  - MY_PI constant
 *  - GPUSample struct (quadrature point: position, normal, area weight)
 *  - triangulate()        host: fan-triangulate a convex polygon
 *  - sampleTriangle()     host: stratified quadrature samples on a triangle
 *  - rayHitsTriangle()    device: Möller–Trumbore ray–triangle intersection
 */

#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include "geometry.h"

// ─────────────────────────────────────────────────────────────────────────────
// Constants & error handling
// ─────────────────────────────────────────────────────────────────────────────

#ifndef MY_PI
#define MY_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(err) \
    do { \
        if ((err) != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

// ─────────────────────────────────────────────────────────────────────────────
// Quadrature point
// ─────────────────────────────────────────────────────────────────────────────

/**
 * One quadrature point on a surface triangle.
 * Stores 3D position, outward unit normal, and the area weight dA.
 */
struct GPUSample {
    double x, y, z;    // world-space position of the sample point
    double nx, ny, nz; // outward unit normal of the parent triangle
    double w;          // area weight: triangle_area / num_samples_in_triangle
};

// ─────────────────────────────────────────────────────────────────────────────
// Fan triangulation (host)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Decompose any convex polygon into triangles using a fan from vertex 0.
 *
 * A polygon with n vertices produces n-2 triangles:
 *   (v0,v1,v2), (v0,v2,v3), ..., (v0,v_{n-2},v_{n-1})
 *
 * This is the simplest valid triangulation for convex polygons and produces
 * no degenerate triangles as long as the vertices are ordered consistently.
 */
inline std::vector<Triangle> triangulate(const std::vector<Vec3>& poly)
{
    std::vector<Triangle> tris;
    // fan: connect vertex 0 to every consecutive edge
    for (int i = 1; i + 1 < (int)poly.size(); ++i)
        tris.push_back({ poly[0], poly[i], poly[i+1] });
    return tris;
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangle sampling (host)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fill `out` with stratified quadrature points over one triangle.
 *
 * The N×N grid in (u,v) barycentric space is split into two sub-grids of
 * points with offsets (1/3, 1/3) and (2/3, 2/3) inside each cell, giving
 * roughly 2·ceil(N²/2) sample points per triangle.  Points with u+v > 1
 * fall outside the triangle and are discarded.
 *
 * Each sample carries the same area weight  dA = triangle_area / num_samples.
 */
static void sampleTriangle(const Triangle& tri, int N,
                            std::vector<GPUSample>& out)
{
    // Compute the triangle's outward normal and area
    Vec3 AB   = tri[1] - tri[0];
    Vec3 AC   = tri[2] - tri[0];
    Vec3 nc   = cross3(AB, AC);
    double area = 0.5 * norm3(nc);
    Vec3 n    = normalize3(nc);

    // First pass: count valid samples to compute the per-sample area weight
    int cnt = 0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            if ((i + 1.0/3.0)/N + (j + 1.0/3.0)/N <= 1.0) ++cnt;
            if ((i + 2.0/3.0)/N + (j + 2.0/3.0)/N <= 1.0) ++cnt;
        }
    if (cnt == 0) cnt = 1;          // degenerate triangle safety guard
    double dA = area / cnt;         // each sample represents this area

    // Second pass: generate the sample points
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            // two candidate points per (i,j) cell
            double us[2] = { (i + 1.0/3.0)/N, (i + 2.0/3.0)/N };
            double vs[2] = { (j + 1.0/3.0)/N, (j + 2.0/3.0)/N };
            for (int s = 0; s < 2; ++s) {
                double u = us[s], v = vs[s];
                if (u + v > 1.0) continue;   // outside triangle, discard
                double w = 1.0 - u - v;       // barycentric third coordinate

                // interpolate world-space position: P = w·v0 + u·v1 + v·v2
                GPUSample sp;
                sp.x  = w*tri[0][0] + u*tri[1][0] + v*tri[2][0];
                sp.y  = w*tri[0][1] + u*tri[1][1] + v*tri[2][1];
                sp.z  = w*tri[0][2] + u*tri[1][2] + v*tri[2][2];
                sp.nx = n[0]; sp.ny = n[1]; sp.nz = n[2];
                sp.w  = dA;
                out.push_back(sp);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ray–triangle intersection (device)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Möller–Trumbore ray–triangle intersection test.
 *
 * Returns true if the ray  origin + t·dir  hits the triangle (a,b,c)
 * at a parameter t in (tmin, tmax).
 *
 * Used for occlusion testing: shoot a ray from a sample on A toward a sample
 * on B and check whether any obstacle triangle blocks it.
 */
__device__ static bool rayHitsTriangle(
    // ray
    double ox, double oy, double oz,   // origin
    double dx, double dy, double dz,   // direction (need not be unit length)
    // triangle vertices
    double ax, double ay, double az,
    double bx, double by, double bz,
    double cx, double cy, double cz,
    // valid interval for t
    double tmin, double tmax)
{
    const double EPS = 1e-9;

    // Edge vectors
    double E1x = bx-ax, E1y = by-ay, E1z = bz-az;
    double E2x = cx-ax, E2y = cy-ay, E2z = cz-az;

    // h = dir × E2  (used to compute the determinant)
    double hx = dy*E2z - dz*E2y;
    double hy = dz*E2x - dx*E2z;
    double hz = dx*E2y - dy*E2x;

    double det = E1x*hx + E1y*hy + E1z*hz;
    if (fabs(det) < EPS) return false;   // ray parallel to triangle

    double inv_det = 1.0 / det;

    // Vector from vertex a to ray origin
    double sx = ox-ax, sy = oy-ay, sz = oz-az;

    // First barycentric coordinate
    double u = inv_det * (sx*hx + sy*hy + sz*hz);
    if (u < 0.0 || u > 1.0) return false;

    // q = s × E1
    double qx = sy*E1z - sz*E1y;
    double qy = sz*E1x - sx*E1z;
    double qz = sx*E1y - sy*E1x;

    // Second barycentric coordinate
    double v = inv_det * (dx*qx + dy*qy + dz*qz);
    if (v < 0.0 || u + v > 1.0) return false;

    // Ray parameter t at the intersection
    double t = inv_det * (E2x*qx + E2y*qy + E2z*qz);
    return (t > tmin && t < tmax);
}
