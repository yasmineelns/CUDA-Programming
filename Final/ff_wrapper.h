/**
 * ff_wrapper.h
 *
 * C++ bridge to the reference exact view factor library (ffp.c / claussenp.c).
 *
 * The reference FormFactor() function computes the analytical view factor
 * between two planar polygons p (np vertices) and q (nq vertices) in 3D.
 *
 * IMPORTANT: FormFactor() uses a contour-integral formula that can return
 * positive values even when the source face's outward normal points away from
 * the target face (i.e. the face is physically back-facing).  We must therefore
 * pre-filter each face pair with an explicit cosine visibility test, before 
 * accumulating the contribution.
 *
 * For convex polyhedra, the total view factor from object A to object B is:
 *
 *   F_{A->B} = (1 / A_total_A) * sum_{i in faces(A)} sum_{j in faces(B)}
 *                  [cos_p > 0 AND cos_q > 0]  *  A_i * F_{i->j}
 *
 * where cos_p = dot(n_i, centroid_ij) / |centroid_ij|
 *       cos_q = -dot(n_j, centroid_ij) / |centroid_ij|   (opposite sign for q)
 *
 * When `subdivisions > 0`, each input face is split into 4^subdivisions
 * sub-triangles (flat mid-point subdivision) before the double loop.  The
 * cosine filter and obstacle check are then applied at the sub-face centroid
 * level.  Finer subdivision → smaller sub-faces → the centroid approximation
 * for partial occlusion becomes increasingly accurate.  This is essential for
 * Phase 2 (with obstacle), where a single large face can be partially occluded
 * and a single centroid ray would make a wrong all-or-nothing decision.
 */

#pragma once
#include <cmath>
#include <vector>
#include <array>

// ── Link against the reference library ──────────────────────────────────────
extern "C" {
    // FormFactor(p, np, q, nq) - defined in ffp.c compiled with -DDOUBLE
    double FormFactor(double (*p)[3], int np, double (*q)[3], int nq);
    // Area(p, np) - area of a polygon
    double Area(double (*p)[3], int np);
}

// ── Helper types (only declared here if geometry.h was not included first) ────
#ifndef GEOMETRY_H_TYPES_DEFINED
#define GEOMETRY_H_TYPES_DEFINED
using Vec3     = std::array<double, 3>;
using Triangle = std::array<Vec3, 3>;  // 3 vertices, each a 3D point
#endif

// ── Internal: face centroid and outward normal from 3 vertices ───────────────
static inline void faceGeometry(const double p[3][3],
                                 double centroid[3], double normal[3])
{
    for (int c = 0; c < 3; ++c)
        centroid[c] = (p[0][c] + p[1][c] + p[2][c]) / 3.0;

    double ab[3] = { p[1][0]-p[0][0], p[1][1]-p[0][1], p[1][2]-p[0][2] };
    double ac[3] = { p[2][0]-p[0][0], p[2][1]-p[0][1], p[2][2]-p[0][2] };
    double nc[3] = { ab[1]*ac[2]-ab[2]*ac[1],
                     ab[2]*ac[0]-ab[0]*ac[2],
                     ab[0]*ac[1]-ab[1]*ac[0] };
    double len = std::sqrt(nc[0]*nc[0] + nc[1]*nc[1] + nc[2]*nc[2]);
    if (len < 1e-15) { normal[0]=normal[1]=normal[2]=0.0; return; }
    for (int c = 0; c < 3; ++c) normal[c] = nc[c] / len;
}

// ── Compute exact view factor between two convex polyhedra ───────────────────
//
// faces_A / faces_B : triangulated surfaces
// subdivisions      : each input face is split into 4^subdivisions sub-triangles
//                     before the integration loop.  The cosine visibility filter
//                     and obstacle check are applied at the sub-face centroid
//                     level, so finer subdivision → more accurate filtering,
//                     converging to the same per-sample logic as the GPU kernels.
// obstacles         : (optional) list of obstacle triangles; a sub-face pair is
//                     skipped if the centroid-to-centroid segment is occluded.
//
// NOTE: geometry.h must be included before ff_wrapper.h so that subdivide()
//       and segmentVisible() are available.
inline double exactViewFactorPolyhedra(
    const std::vector<Triangle>& faces_A,
    const std::vector<Triangle>& faces_B,
    int subdivisions = 0,
    const std::vector<Triangle>& obstacles = {})
{
    // Flat mid-point subdivision: each triangle is split into 4 by inserting
    // edge mid-points, without any sphere projection.  This is correct for
    // arbitrary planar faces (unlike the sphere-aware subdivide() in geometry.h
    // which re-projects mid-points onto the circumsphere).
    auto flatSubdivide = [](const std::vector<Triangle>& mesh, int depth)
        -> std::vector<Triangle>
    {
        std::vector<Triangle> cur = mesh;
        for (int d = 0; d < depth; ++d) {
            std::vector<Triangle> next;
            next.reserve(cur.size() * 4);
            for (const auto& tri : cur) {
                Vec3 a = tri[0], b = tri[1], c = tri[2];
                Vec3 ab = { (a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2 };
                Vec3 bc = { (b[0]+c[0])/2, (b[1]+c[1])/2, (b[2]+c[2])/2 };
                Vec3 ca = { (c[0]+a[0])/2, (c[1]+a[1])/2, (c[2]+a[2])/2 };
                next.push_back({a, ab, ca});
                next.push_back({b, bc, ab});
                next.push_back({c, ca, bc});
                next.push_back({ab, bc, ca});
            }
            cur = next;
        }
        return cur;
    };

    const auto subA = flatSubdivide(faces_A, subdivisions);
    const auto subB = flatSubdivide(faces_B, subdivisions);

    double total_area_A = 0.0;
    double weighted_sum  = 0.0;

    for (const auto& fa : subA) {
        double p[3][3];
        for (int v = 0; v < 3; ++v)
            for (int c = 0; c < 3; ++c)
                p[v][c] = fa[v][c];

        double area_a = Area(p, 3);
        total_area_A += area_a;

        double cen_p[3], n_p[3];
        faceGeometry(p, cen_p, n_p);

        for (const auto& fb : subB) {
            double q[3][3];
            for (int v = 0; v < 3; ++v)
                for (int c = 0; c < 3; ++c)
                    q[v][c] = fb[v][c];

            double cen_q[3], n_q[3];
            faceGeometry(q, cen_q, n_q);

            // Direction vector from sub-face centroid of p to centroid of q
            double dx = cen_q[0]-cen_p[0], dy = cen_q[1]-cen_p[1], dz = cen_q[2]-cen_p[2];
            double r = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (r < 1e-12) continue;

            // Cosine visibility test at sub-face centroid level
            double cos_p =  (n_p[0]*dx + n_p[1]*dy + n_p[2]*dz) / r;
            double cos_q = -(n_q[0]*dx + n_q[1]*dy + n_q[2]*dz) / r;
            if (cos_p <= 0.0 || cos_q <= 0.0) continue;

            // Obstacle occlusion test (Phase 2): skip if centroid segment blocked
            if (!obstacles.empty()) {
                Vec3 P = {cen_p[0], cen_p[1], cen_p[2]};
                Vec3 Q = {cen_q[0], cen_q[1], cen_q[2]};
                if (!segmentVisible(P, Q, obstacles)) continue;
            }

            double ff_ij = FormFactor(p, 3, q, 3);
            if (ff_ij > 0.0) weighted_sum += area_a * ff_ij;
        }
    }

    if (total_area_A < 1e-15) return 0.0;
    return weighted_sum / total_area_A;
}


