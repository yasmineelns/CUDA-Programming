/**
 * sto_gpu.cuh — Stochastic (Monte Carlo) GPU method for view-factor computation
 *
 * OVERVIEW
 * --------
 * The stochastic method estimates F_{A→B} by firing N random rays from surface A
 * and counting the fraction that hit surface B:
 *
 *   F_AB ≈ hits / N
 *
 * Each ray is sampled in two steps:
 *   1. Origin: drawn uniformly on surface A (area-weighted across all triangles).
 *   2. Direction: drawn from the cosine-weighted hemisphere around A's normal.
 *
 * This importance sampling makes the estimator unbiased with respect to the
 * exact view-factor integrand (cos θ / π is the cosine-hemisphere PDF).
 *
 * OCCLUSION
 * ---------
 * Same as the deterministic kernel: when nObs == 0 the obstacle loop is absent
 * (Phase 1, no performance penalty); when nObs > 0 each ray is tested against
 * every obstacle triangle before checking if it hits B.
 *
 * NOTE
 * ----
 * Shared utilities (CUDA_CHECK, MY_PI, GPUSample, triangulate,
 * sampleTriangle, rayHitsTriangle) live in gpu_utils.cuh.
 */

#pragma once

#include <curand_kernel.h>
#include "gpu_utils.cuh"   // CUDA_CHECK, MY_PI, rayHitsTriangle, triangulate, …

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

/**
 * All GPU buffers needed to run the stochastic kernel.
 * Built on the host by buildStoGPUData(), freed by freeStoGPUData().
 */
struct StoGPUData {
    double*      d_facesA;    // device array: flat list of A's triangles (nFacesA × 9 doubles)
    double*      d_facesB;    // device array: flat list of B's triangles (nFacesB × 9 doubles)
    double*      d_cdfA;      // device array: cumulative-area CDF over A's triangles (nFacesA)
    double*      d_normalsA;  // device array: outward unit normals of A's triangles (nFacesA × 3)
    curandState* d_rngStates; // device array: one persistent RNG state per thread
    int          nFacesA, nFacesB;
    int          nStates;     // number of persistent RNG threads (fixed at build time)
};

// ─────────────────────────────────────────────────────────────────────────────
// Device helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Draw a uniformly random point on a triangle using barycentric coordinates.
 *
 * Given two uniform random numbers r1, r2 ∈ [0,1), the point
 *   P = (1 - √r1)·A + √r1·(1-r2)·B + √r1·r2·C
 * is uniformly distributed on the triangle.  The square-root mapping
 * corrects for the non-uniform density that a naive (r1, r2) parameterisation
 * would produce near vertex A.
 */
__device__ static void baryRandPoint(
    double ax, double ay, double az,
    double bx, double by, double bz,
    double cx, double cy, double cz,
    double r1, double r2,
    double& px, double& py, double& pz)
{
    // Fold the unit square back onto the triangle:
    // Points with r1+r2 > 1 would land in the "other" half of the unit square;
    // mirror them to stay inside the triangle.
    if (r1 + r2 > 1.0) { r1 = 1.0 - r1; r2 = 1.0 - r2; }

    double w = 1.0 - r1 - r2;   // third barycentric coordinate
    px = w*ax + r1*bx + r2*cx;
    py = w*ay + r1*by + r2*cy;
    pz = w*az + r1*bz + r2*cz;
}

/**
 * Sample a direction from the cosine-weighted hemisphere over normal (nx,ny,nz).
 *
 * Uses Malley's method: draw a cosine-weighted point on the unit disk and
 * project it up onto the hemisphere.  The tangent-frame vectors t1, t2 are
 * constructed to be orthogonal to n.  To avoid a degenerate cross product we
 * choose the reference vector (1,0,0) when n is nearly vertical (|ny| > 0.9),
 * and (0,1,0) otherwise.
 *
 * Result (dx, dy, dz) is a unit vector in the upper hemisphere.
 */
__device__ static void cosineDir(
    double nx, double ny, double nz,
    double r1, double r2,
    double& dx, double& dy, double& dz)
{
    // Map r1 to radial coordinate, r2 to azimuthal angle
    double cosT = sqrt(r1);         // cos θ — gives cosine-weighted elevation
    double sinT = sqrt(1.0 - r1);   // sin θ
    double phi  = 2.0 * MY_PI * r2; // azimuthal angle φ ∈ [0, 2π)

    // Build a local orthonormal frame (t1, t2, n)
    // Choose a reference vector not parallel to n to avoid degenerate cross product
    double ux, uy, uz;
    if (fabs(ny) > 0.9) {
        ux = 1.0; uy = 0.0; uz = 0.0;   // n is nearly vertical → use x-axis
    } else {
        ux = 0.0; uy = 1.0; uz = 0.0;   // otherwise use y-axis
    }

    // t1 = n × u  (first tangent)
    double t1x = ny*uz - nz*uy;
    double t1y = nz*ux - nx*uz;
    double t1z = nx*uy - ny*ux;
    double len = sqrt(t1x*t1x + t1y*t1y + t1z*t1z);
    t1x /= len; t1y /= len; t1z /= len;

    // t2 = n × t1  (second tangent, already unit length)
    double t2x = ny*t1z - nz*t1y;
    double t2y = nz*t1x - nx*t1z;
    double t2z = nx*t1y - ny*t1x;

    // Direction = sinT·cos(φ)·t1 + sinT·sin(φ)·t2 + cosT·n
    double cp = cos(phi), sp = sin(phi);
    dx = sinT*cp*t1x + sinT*sp*t2x + cosT*nx;
    dy = sinT*cp*t1y + sinT*sp*t2y + cosT*ny;
    dz = sinT*cp*t1z + sinT*sp*t2z + cosT*nz;
}

/**
 * Test whether the ray (ox,oy,oz) + t·(dx,dy,dz) hits any triangle in the
 * obstacle mesh.  Returns true as soon as the first intersection is found.
 */
__device__ static bool dev_rayHitsAnyTri(
    double ox, double oy, double oz,
    double dx, double dy, double dz,
    const double* obsData, int nObs)
{
    for (int o = 0; o < nObs; ++o) {
        const double* tri = obsData + o*9;
        if (rayHitsTriangle(ox, oy, oz, dx, dy, dz,
                            tri[0], tri[1], tri[2],
                            tri[3], tri[4], tri[5],
                            tri[6], tri[7], tri[8],
                            1e-6, 1e30))
            return true;
    }
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// RNG initialisation kernel — called once at build time
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Initialise one curandState per thread.  This is the only place curand_init()
 * is called; subsequent kernel launches simply load/store the state from global
 * memory, amortising the expensive init cost across all runs.
 */
__global__ void init_rng_kernel(curandState* states, int nStates,
                                 unsigned long long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nStates) return;
    // Each thread gets an independent subsequence via the sequence argument.
    curand_init(seed, /*sequence=*/tid, /*offset=*/0, &states[tid]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Stochastic kernel
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Each thread handles raysPerThread rays, reusing its persistent RNG state.
 * The state is loaded from global memory at entry and written back at exit so
 * that successive kernel launches continue from where they left off (no repeated
 * samples across launches).
 *
 * Per ray:
 *   1. Load persistent RNG state from global memory.
 *   2. Loop raysPerThread times:
 *      a. Pick a source triangle on A proportional to area (CDF lookup).
 *      b. Draw a random point on that triangle (baryRandPoint).
 *      c. Draw a cosine-weighted outgoing direction (cosineDir).
 *      d. (Phase 2 only) Test ray against obstacle mesh.
 *      e. Test ray against all triangles of B; atomically count hits.
 *   3. Store RNG state back to global memory.
 *
 * Final estimate: F_AB ≈ d_hits / (raysPerThread * activeThreads)  (see sto_GPU_run).
 */
__global__ void sto_kernel(
    const double*  __restrict__ facesA, int nFacesA,
    const double*  __restrict__ facesB, int nFacesB,
    const double*  __restrict__ cdfA,
    const double*  __restrict__ normalsA,
    const double*  __restrict__ obsData, int nObs,
    curandState*                rngStates,   // persistent states (one per thread)
    int raysPerThread,                       // rays this thread must fire
    int* d_hits)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // ── Load persistent RNG state ─────────────────────────────────────────────
    curandState rng = rngStates[tid];

    int localHits = 0;

    for (int ray = 0; ray < raysPerThread; ++ray) {

        // ── Pick a source triangle proportional to its area ───────────────────
        double u = curand_uniform_double(&rng);
        int fi = 0;
        for (int k = 0; k < nFacesA; ++k)
            if (cdfA[k] >= u) { fi = k; break; }

        // ── Draw a random point on the selected triangle ──────────────────────
        const double* tA = facesA + fi*9;
        double r1 = curand_uniform_double(&rng);
        double r2 = curand_uniform_double(&rng);
        double ox, oy, oz;
        baryRandPoint(tA[0],tA[1],tA[2],
                      tA[3],tA[4],tA[5],
                      tA[6],tA[7],tA[8],
                      r1, r2, ox, oy, oz);

        // ── Draw a cosine-weighted ray direction ──────────────────────────────
        double nx = normalsA[fi*3], ny = normalsA[fi*3+1], nz = normalsA[fi*3+2];
        double r3 = curand_uniform_double(&rng);
        double r4 = curand_uniform_double(&rng);
        double dx, dy, dz;
        cosineDir(nx, ny, nz, r3, r4, dx, dy, dz);

        // ── (Phase 2 only) Check occlusion ────────────────────────────────────
        if (nObs > 0 && dev_rayHitsAnyTri(ox, oy, oz, dx, dy, dz, obsData, nObs))
            continue;   // ray blocked by obstacle

        // ── Test whether the ray hits any triangle of B ───────────────────────
        for (int ib = 0; ib < nFacesB; ++ib) {
            const double* tB = facesB + ib*9;
            if (rayHitsTriangle(ox, oy, oz, dx, dy, dz,
                                tB[0],tB[1],tB[2],
                                tB[3],tB[4],tB[5],
                                tB[6],tB[7],tB[8],
                                1e-6, 1e30)) {
                ++localHits;
                break;   // each ray counts at most once
            }
        }
    }

    // ── Write back accumulated hits and updated RNG state ─────────────────────────
    if (localHits) atomicAdd(d_hits, localHits);
    rngStates[tid] = rng;
}

// ─────────────────────────────────────────────────────────────────────────────
// Host API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Upload surface meshes to the GPU and build the area-CDF for importance sampling.
 *
 * The CDF is computed as the normalised prefix sum of triangle areas:
 *   CDF[k] = (area[0] + … + area[k]) / total_area_A
 * so that CDF[nFacesA-1] == 1.  A thread drawing a uniform deviate and
 * searching for the first CDF entry ≥ u picks triangle k with probability
 * proportional to its area.
 */
inline StoGPUData buildStoGPUData(
    const std::vector<Triangle>& facesA,
    const std::vector<Triangle>& facesB)
{
    StoGPUData d;
    d.nFacesA = (int)facesA.size();
    d.nFacesB = (int)facesB.size();

    // ── Flatten triangles into double arrays ──────────────────────────────────
    // Layout for each triangle: [v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z]
    std::vector<double> hFacesA(d.nFacesA * 9);
    std::vector<double> hFacesB(d.nFacesB * 9);
    std::vector<double> hCdfA(d.nFacesA);
    std::vector<double> hNormalsA(d.nFacesA * 3);

    // Surface A: store vertices, compute normals and area CDF
    double cumArea = 0.0;
    for (int i = 0; i < d.nFacesA; ++i) {
        for (int v = 0; v < 3; ++v)
            for (int c = 0; c < 3; ++c)
                hFacesA[i*9 + v*3 + c] = facesA[i][v][c];

        Vec3 AB = facesA[i][1] - facesA[i][0];
        Vec3 AC = facesA[i][2] - facesA[i][0];
        Vec3 nc = cross3(AB, AC);
        Vec3 n  = normalize3(nc);
        hNormalsA[i*3+0] = n[0];
        hNormalsA[i*3+1] = n[1];
        hNormalsA[i*3+2] = n[2];
        cumArea += 0.5 * norm3(nc);
        hCdfA[i] = cumArea;   // raw prefix sum for now; normalise below
    }
    // Normalise the CDF so that CDF[nFacesA-1] == 1.0
    double totalAreaA = cumArea;
    for (auto& v : hCdfA) v /= totalAreaA;

    // Surface B: store vertices only (no normals needed on B — hit test only)
    for (int i = 0; i < d.nFacesB; ++i)
        for (int v = 0; v < 3; ++v)
            for (int c = 0; c < 3; ++c)
                hFacesB[i*9 + v*3 + c] = facesB[i][v][c];

    // ── Upload to device ──────────────────────────────────────────────────────
    CUDA_CHECK(cudaMalloc(&d.d_facesA,   hFacesA.size()   * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.d_facesB,   hFacesB.size()   * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.d_cdfA,     hCdfA.size()     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.d_normalsA, hNormalsA.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d.d_facesA,   hFacesA.data(),   hFacesA.size()   * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_facesB,   hFacesB.data(),   hFacesB.size()   * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_cdfA,     hCdfA.data(),     hCdfA.size()     * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_normalsA, hNormalsA.data(), hNormalsA.size() * sizeof(double), cudaMemcpyHostToDevice));

    // ── Pre-allocate and initialise persistent RNG states ─────────────────────
    // Fixed thread count: 256 threads/block × 1024 blocks = 262144 threads.
    // Each thread handles N/nStates rays per launch, so any N is supported.
    const int BLOCK = 256, GRID = 1024;
    d.nStates = BLOCK * GRID;
    CUDA_CHECK(cudaMalloc(&d.d_rngStates, d.nStates * sizeof(curandState)));
    init_rng_kernel<<<GRID, BLOCK>>>(d.d_rngStates, d.nStates, /*seed=*/42ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    return d;
}

/** Free all device buffers allocated by buildStoGPUData(). */
inline void freeStoGPUData(StoGPUData& d) {
    cudaFree(d.d_facesA);
    cudaFree(d.d_facesB);
    cudaFree(d.d_cdfA);
    cudaFree(d.d_normalsA);
    cudaFree(d.d_rngStates);
}

/**
 * Launch the stochastic kernel and return the view-factor estimate.
 *
 * Fires N rays total.  Pass nObs=0 (obsData=nullptr) for Phase 1.
 * d_hits is a reusable device int counter.
 */
inline double sto_GPU_run(const StoGPUData& d,
                           int* d_hits,
                           const double* obsData, int nObs,
                           int N)
{
    CUDA_CHECK(cudaMemset(d_hits, 0, sizeof(int)));

    // Use only as many threads as needed when N < nStates, so that small N
    // values actually fire fewer rays and the error scales as 1/sqrt(N).
    const int BLOCK = 256;
    int activeThreads = std::min(N, d.nStates);
    // Round up to nearest BLOCK multiple so we always fill complete warps.
    activeThreads = ((activeThreads + BLOCK - 1) / BLOCK) * BLOCK;
    activeThreads = std::min(activeThreads, d.nStates); // never exceed allocated states
    const int GRID = activeThreads / BLOCK;

    int raysPerThread = (N + activeThreads - 1) / activeThreads;

    sto_kernel<<<GRID, BLOCK>>>(
        d.d_facesA, d.nFacesA,
        d.d_facesB, d.nFacesB,
        d.d_cdfA, d.d_normalsA,
        obsData, nObs,
        d.d_rngStates, raysPerThread,
        d_hits);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_hits = 0;
    CUDA_CHECK(cudaMemcpy(&h_hits, d_hits, sizeof(int), cudaMemcpyDeviceToHost));
    // Divide by actual rays fired (raysPerThread * activeThreads), not N, to be exact.
    return (double)h_hits / (double)(raysPerThread * activeThreads);
}
