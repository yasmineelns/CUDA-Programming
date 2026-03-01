/**
 * geometry.h
 *
 * Generates a regular icosahedron (20 equilateral triangular faces, 12 vertices)
 * centred at (cx, cy, cz) with circumradius `radius`.
 *
 * Also provides a helper to build a simple cubic obstacle (6 quad faces split
 * into 12 triangles) for the Phase-2 visibility experiments.
 */

#pragma once
#include <vector>
#include <array>
#include <cmath>

// Guard so ff_wrapper.h (which may also be included alone) doesn't re-declare
#define GEOMETRY_H_TYPES_DEFINED
using Vec3     = std::array<double, 3>;
using Triangle = std::array<Vec3, 3>;

// ── Utility ──────────────────────────────────────────────────────────────────

static inline Vec3 operator+(Vec3 a, Vec3 b) { return {a[0]+b[0], a[1]+b[1], a[2]+b[2]}; }
static inline Vec3 operator-(Vec3 a, Vec3 b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
static inline Vec3 operator*(double s, Vec3 a) { return {s*a[0], s*a[1], s*a[2]}; }

static inline double dot3(Vec3 a, Vec3 b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
static inline Vec3 cross3(Vec3 a, Vec3 b) {
    return { a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0] };
}
static inline double norm3(Vec3 a) { return std::sqrt(dot3(a,a)); }
static inline Vec3 normalize3(Vec3 a) { double n = norm3(a); return {a[0]/n, a[1]/n, a[2]/n}; }

// Face centroid
static inline Vec3 centroid3(const Triangle& t) {
    return { (t[0][0]+t[1][0]+t[2][0])/3.0,
             (t[0][1]+t[1][1]+t[2][1])/3.0,
             (t[0][2]+t[1][2]+t[2][2])/3.0 };
}

// Face area (0.5 * |AB x AC|)
static inline double triArea(const Triangle& t) {
    Vec3 ab = t[1] - t[0];
    Vec3 ac = t[2] - t[0];
    return 0.5 * norm3(cross3(ab, ac));
}

// ── Icosahedron ───────────────────────────────────────────────────────────────
//
// A regular icosahedron has 12 vertices and 20 triangular faces.
// We use the classic "two rings + poles" construction.

inline std::vector<Triangle> generateIcosahedron(
    double cx, double cy, double cz, double radius)
{
    // The 12 vertices of a regular icosahedron (unit circumradius, centred at origin)
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;  // golden ratio ≈ 1.618

    // Raw vertices (not normalised to unit sphere yet)
    Vec3 raw[12] = {
        {-1,  t,  0}, { 1,  t,  0}, {-1, -t,  0}, { 1, -t,  0},
        { 0, -1,  t}, { 0,  1,  t}, { 0, -1, -t}, { 0,  1, -t},
        { t,  0, -1}, { t,  0,  1}, {-t,  0, -1}, {-t,  0,  1}
    };

    // Normalise to unit sphere then scale
    Vec3 verts[12];
    for (int i = 0; i < 12; ++i) {
        Vec3 n = normalize3(raw[i]);
        verts[i] = { cx + radius * n[0],
                     cy + radius * n[1],
                     cz + radius * n[2] };
    }

    // 20 faces (vertex indices), wound consistently outward
    int faces[20][3] = {
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
        {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
        {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}
    };

    std::vector<Triangle> result;
    result.reserve(20);
    for (int f = 0; f < 20; ++f) {
        Triangle tri;
        tri[0] = verts[faces[f][0]];
        tri[1] = verts[faces[f][1]];
        tri[2] = verts[faces[f][2]];
        result.push_back(tri);
    }
    return result;
}

// ── Subdivision (optional higher-density meshes) ──────────────────────────────
//
// Subdivide each triangle into 4 by splitting edges at mid-points and
// projecting to the sphere. Repeat `depth` times to get 20*4^depth faces.

static inline Vec3 midpointOnSphere(Vec3 a, Vec3 b, double r, double cx, double cy, double cz) {
    Vec3 mid = { (a[0]+b[0])/2.0, (a[1]+b[1])/2.0, (a[2]+b[2])/2.0 };
    // Translate to origin, normalise, scale, translate back
    Vec3 centred = { mid[0]-cx, mid[1]-cy, mid[2]-cz };
    Vec3 n = normalize3(centred);
    return { cx + r*n[0], cy + r*n[1], cz + r*n[2] };
}

inline std::vector<Triangle> subdivide(
    const std::vector<Triangle>& mesh, int depth,
    double cx, double cy, double cz, double radius)
{
    std::vector<Triangle> current = mesh;
    for (int d = 0; d < depth; ++d) {
        std::vector<Triangle> next;
        next.reserve(current.size() * 4);
        for (const auto& tri : current) {
            Vec3 a = tri[0], b = tri[1], c = tri[2];
            Vec3 ab = midpointOnSphere(a, b, radius, cx, cy, cz);
            Vec3 bc = midpointOnSphere(b, c, radius, cx, cy, cz);
            Vec3 ca = midpointOnSphere(c, a, radius, cx, cy, cz);
            next.push_back({a, ab, ca});
            next.push_back({b, bc, ab});
            next.push_back({c, ca, bc});
            next.push_back({ab, bc, ca});
        }
        current = next;
    }
    return current;
}

// ── Obstacle: axis-aligned box ────────────────────────────────────────────────
//
// Returns 12 triangles forming the surface of the box [cx-hx, cx+hx] x ... .
// Used in Phase-2 visibility experiments.

inline std::vector<Triangle> generateBox(
    double cx, double cy, double cz,
    double hx, double hy, double hz)
{
    double x0 = cx-hx, x1 = cx+hx;
    double y0 = cy-hy, y1 = cy+hy;
    double z0 = cz-hz, z1 = cz+hz;

    Vec3 corners[8] = {
        {x0,y0,z0},{x1,y0,z0},{x1,y1,z0},{x0,y1,z0},
        {x0,y0,z1},{x1,y0,z1},{x1,y1,z1},{x0,y1,z1}
    };
    // 6 faces × 2 triangles each
    int quads[6][4] = {
        {0,1,2,3},{4,7,6,5},{0,4,5,1},{2,6,7,3},{0,3,7,4},{1,5,6,2}
    };
    std::vector<Triangle> tris;
    tris.reserve(12);
    for (auto& q : quads) {
        tris.push_back({corners[q[0]], corners[q[1]], corners[q[2]]});
        tris.push_back({corners[q[0]], corners[q[2]], corners[q[3]]});
    }
    return tris;
}

// ── Möller–Trumbore ray–triangle intersection ─────────────────────────────────
//
// Returns true if the ray (origin O, direction D) hits triangle tri at
// parameter t in (tmin, tmax).

static inline bool rayTriangle(
    Vec3 O, Vec3 D,
    const Triangle& tri,
    double tmin, double tmax,
    double& tHit)
{
    const double EPS = 1e-9;
    Vec3 E1 = tri[1] - tri[0];
    Vec3 E2 = tri[2] - tri[0];
    Vec3 h  = cross3(D, E2);
    double a = dot3(E1, h);
    if (std::fabs(a) < EPS) return false;
    double f = 1.0 / a;
    Vec3 s = O - tri[0];
    double u = f * dot3(s, h);
    if (u < 0.0 || u > 1.0) return false;
    Vec3 q = cross3(s, E1);
    double v = f * dot3(D, q);
    if (v < 0.0 || u + v > 1.0) return false;
    tHit = f * dot3(E2, q);
    return (tHit > tmin && tHit < tmax);
}

// ── Visibility test ───────────────────────────────────────────────────────────
//
// Returns false if the segment from P to Q is blocked by any face in `obstacles`.

static inline bool segmentVisible(
    Vec3 P, Vec3 Q,
    const std::vector<Triangle>& obstacles)
{
    Vec3 D = Q - P;
    double len = norm3(D);
    if (len < 1e-12) return true;
    Vec3 Dn = { D[0]/len, D[1]/len, D[2]/len };
    double tHit;
    for (const auto& obs : obstacles) {
        if (rayTriangle(P, Dn, obs, 1e-6, len - 1e-6, tHit))
            return false;
    }
    return true;
}
