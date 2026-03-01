/**
 * det_gpu.cuh — Deterministic GPU quadrature for view-factor computation
 *
 * OVERVIEW
 * --------
 * The view factor F_{A→B} is the fraction of diffuse radiation leaving
 * surface A that arrives at surface B.  Its integral form is:
 *
 *         1    ⌠⌠   cos θ_i · cos θ_j
 *  F_AB = ── · ║║  ─────────────────── dA_i dA_j
 *         A_A  ⌡⌡        π r²
 *
 * where θ_i (θ_j) is the angle between the connecting ray and the outward
 * normal at the point on A (B), and r is the distance between the two points.
 *
 * APPROACH
 * --------
 * 1. Both surfaces are represented as a list of triangles (triangulated mesh).
 * 2. Each triangle is sampled with a stratified N×N grid of quadrature points.
 * 3. One CUDA thread is launched per (face_A, face_B) pair.
 *    Each thread loops over all sample pairs (p ∈ face_A, q ∈ face_B) and
 *    accumulates the kernel value  cos_p · cos_q / (π r²) · dA_p · dA_q.
 * 4. Thread results are merged with atomicAdd into a global accumulator.
 * 5. The final estimate is  weighted_sum / total_area_A.
 *
 * OCCLUSION
 * ---------
 * A single kernel handles both phases:
 *   nObs == 0  →  Phase 1: no obstacle, occlusion loop is fully skipped.
 *   nObs  > 0  →  Phase 2: each ray is tested against nObs obstacle triangles
 *                 using the Möller–Trumbore algorithm; blocked rays are skipped.
 */

#pragma once

#include "gpu_utils.cuh"   // CUDA_CHECK, MY_PI, GPUSample, triangulate, sampleTriangle, rayHitsTriangle

// ─────────────────────────────────────────────────────────────────────────────
// DetGPUData — device buffers for the deterministic method
// ─────────────────────────────────────────────────────────────────────────────

/**
 * All GPU buffers needed to run the deterministic kernel.
 * Built on the host by buildDetGPUData(), freed by freeDetGPUData().
 */
struct DetGPUData {
    GPUSample* d_sampA;      // device array: nFacesA × nSampPerFaceA samples
    GPUSample* d_sampB;      // device array: nFacesB × nSampPerFaceB samples
    double*    d_areasA;     // device array: area of each face of A (nFacesA)
    int nFacesA, nFacesB;
    int nSampPerFaceA, nSampPerFaceB;
};

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic kernel
// ─────────────────────────────────────────────────────────────────────────────

/**
 * One thread per (face_A, face_B) pair.
 *
 * Each thread iterates over all quadrature sample pairs (p, q) and
 * accumulates the view-factor integrand:
 *
 *   cos θ_p · cos θ_q
 *   ────────────────── · dA_p · dA_q
 *        π · r²
 *
 * where:
 *   cos θ_p = dot(n_A,  d̂)   — projection of ray onto A's normal
 *   cos θ_q = dot(n_B, -d̂)   — projection of ray onto B's normal (opposite)
 *   r        = |r⃗|             — distance between the two sample points
 *   d̂        = r⃗ / r           — unit direction from p to q
 *
 * VISIBILITY FILTER (per-sample cosine filter)
 * -----------------------------------------------------------------------
 * Back-facing sample pairs are rejected by testing cos_p > 0 and cos_q > 0
 * individually for each quadrature point pair.  This matches the sample-level
 * filtering used by exactViewFactorPolyhedra() (which filters each sub-face
 * centroid after mesh subdivision) and sto_gpu (which filters each random ray).
 *
 * When nObs > 0, each ray is additionally tested against the obstacle mesh;
 * blocked rays are skipped.  When nObs == 0, the obstacle loop is absent
 * and there is no performance penalty for Phase 1.
 *
 * Results are accumulated into d_weighted_sum (= Σ face_ff) and
 * d_total_area_A (= total area of A), so the caller can compute
 *   F_AB = d_weighted_sum / d_total_area_A.
 */
__global__ void det_kernel(
    const GPUSample* __restrict__ sampA, int nSampPerFaceA, int nFacesA,
    const GPUSample* __restrict__ sampB, int nSampPerFaceB, int nFacesB,
    const double*    __restrict__ areasA,
    const double*    __restrict__ obsData, int nObs,   // obstacle triangles (flat)
    double* d_weighted_sum,
    double* d_total_area_A)
{
    // Each thread handles one (ia, ib) face pair
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nFacesA * nFacesB) return;
    int ia = tid / nFacesB;
    int ib = tid % nFacesB;

    // Each face of A contributes its area to the denominator exactly once.
    // This must happen unconditionally — before any face-pair filter — so that
    // d_total_area_A accumulates the full surface area of A even when some
    // face pairs are rejected by the visibility test below.
    if (ib == 0) atomicAdd(d_total_area_A, areasA[ia]);

    // Pointers to this face's sample arrays
    const GPUSample* spA = sampA + ia * nSampPerFaceA;
    const GPUSample* spB = sampB + ib * nSampPerFaceB;

    // Accumulate the view-factor contribution from this face pair
    double face_ff = 0.0;

    for (int p = 0; p < nSampPerFaceA; ++p) {
        for (int q = 0; q < nSampPerFaceB; ++q) {

            // Vector from sample p (on A) to sample q (on B)
            double dx = spB[q].x - spA[p].x;
            double dy = spB[q].y - spA[p].y;
            double dz = spB[q].z - spA[p].z;
            double r2  = dx*dx + dy*dy + dz*dz;
            if (r2 < 1e-14) continue;   // samples coincide, skip
            double r = sqrt(r2);

            // Cosine of angle between ray and A's normal
            double cos_p = (spA[p].nx*dx + spA[p].ny*dy + spA[p].nz*dz) / r;
            // Cosine of angle between ray and B's normal (ray arrives from -d̂)
            double cos_q = -(spB[q].nx*dx + spB[q].ny*dy + spB[q].nz*dz) / r;

            // Per-sample visibility filter: skip back-facing sample pairs
            if (cos_p <= 0.0 || cos_q <= 0.0) continue;

            // Phase 2 only: check whether an obstacle triangle blocks the ray
            if (nObs > 0) {
                double inv = 1.0 / r;
                double dnx = dx*inv, dny = dy*inv, dnz = dz*inv;  // unit direction
                bool blocked = false;
                for (int o = 0; o < nObs && !blocked; ++o) {
                    const double* tri = obsData + o*9;
                    blocked = rayHitsTriangle(
                        spA[p].x, spA[p].y, spA[p].z,   // ray origin
                        dnx, dny, dnz,                    // ray direction
                        tri[0], tri[1], tri[2],           // obstacle triangle
                        tri[3], tri[4], tri[5],
                        tri[6], tri[7], tri[8],
                        1e-6, r - 1e-6);                  // t ∈ (ε, r-ε)
                }
                if (blocked) continue;
            }

            // Add the integrand contribution for this sample pair
            face_ff += cos_p * cos_q / (MY_PI * r2) * spA[p].w * spB[q].w;
        }
    }

    // Accumulate into global sum (atomic because multiple threads write here)
    atomicAdd(d_weighted_sum, face_ff);
}

// ─────────────────────────────────────────────────────────────────────────────
// Host API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Upload surface A and B to the GPU and pre-compute quadrature samples.
 *
 * `facesA` and `facesB` are lists of triangles (e.g. from generateIcosahedron
 * or triangulate()).  N controls the sampling density per triangle: each
 * triangle gets roughly 2·N² sample points.
 */
inline DetGPUData buildDetGPUData(
    const std::vector<Triangle>& facesA,
    const std::vector<Triangle>& facesB,
    int N)
{
    DetGPUData d;
    d.nFacesA = (int)facesA.size();
    d.nFacesB = (int)facesB.size();

    // Determine how many samples the first face of each surface produces
    // (all triangles in the same surface use the same N, so count is the same)
    { std::vector<GPUSample> tmp; sampleTriangle(facesA[0], N, tmp); d.nSampPerFaceA = (int)tmp.size(); }
    { std::vector<GPUSample> tmp; sampleTriangle(facesB[0], N, tmp); d.nSampPerFaceB = (int)tmp.size(); }

    // Allocate flat host arrays (face index × samples per face)
    std::vector<GPUSample> hostSampA(d.nFacesA * d.nSampPerFaceA);
    std::vector<GPUSample> hostSampB(d.nFacesB * d.nSampPerFaceB);
    std::vector<double>    hostAreasA(d.nFacesA);

    for (int ia = 0; ia < d.nFacesA; ++ia) {
        std::vector<GPUSample> samp;
        sampleTriangle(facesA[ia], N, samp);
        // Pad to fixed size so the GPU can index uniformly
        samp.resize(d.nSampPerFaceA, samp.empty() ? GPUSample{} : samp.back());
        for (int s = 0; s < d.nSampPerFaceA; ++s)
            hostSampA[ia * d.nSampPerFaceA + s] = samp[s];
        Vec3 AB = facesA[ia][1] - facesA[ia][0];
        Vec3 AC = facesA[ia][2] - facesA[ia][0];
        hostAreasA[ia] = 0.5 * norm3(cross3(AB, AC));
    }
    for (int ib = 0; ib < d.nFacesB; ++ib) {
        std::vector<GPUSample> samp;
        sampleTriangle(facesB[ib], N, samp);
        samp.resize(d.nSampPerFaceB, samp.empty() ? GPUSample{} : samp.back());
        for (int s = 0; s < d.nSampPerFaceB; ++s)
            hostSampB[ib * d.nSampPerFaceB + s] = samp[s];
    }

    // Upload to device
    CUDA_CHECK(cudaMalloc(&d.d_sampA,  hostSampA.size()  * sizeof(GPUSample)));
    CUDA_CHECK(cudaMalloc(&d.d_sampB,  hostSampB.size()  * sizeof(GPUSample)));
    CUDA_CHECK(cudaMalloc(&d.d_areasA, hostAreasA.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d.d_sampA,  hostSampA.data(),  hostSampA.size()  * sizeof(GPUSample), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_sampB,  hostSampB.data(),  hostSampB.size()  * sizeof(GPUSample), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_areasA, hostAreasA.data(), hostAreasA.size() * sizeof(double),    cudaMemcpyHostToDevice));
    return d;
}

/** Free all device buffers allocated by buildDetGPUData(). */
inline void freeDetGPUData(DetGPUData& d) {
    cudaFree(d.d_sampA);
    cudaFree(d.d_sampB);
    cudaFree(d.d_areasA);
}

/**
 * Launch the deterministic kernel and return the view-factor estimate.
 *
 * Pass nObs=0 (and obsData=nullptr) for Phase 1 (no obstacle).
 * Pass nObs>0 and a valid obsData pointer for Phase 2.
 *
 * d_ws and d_ta are reusable device scalars (double) for the accumulator.
 */
inline double det_GPU_run(const DetGPUData& d,
                           double* d_ws, double* d_ta,
                           const double* obsData, int nObs)
{
    // Reset accumulators
    CUDA_CHECK(cudaMemset(d_ws, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_ta, 0, sizeof(double)));

    int nPairs = d.nFacesA * d.nFacesB;
    int block  = 256;
    int grid   = (nPairs + block - 1) / block;

    det_kernel<<<grid, block>>>(
        d.d_sampA, d.nSampPerFaceA, d.nFacesA,
        d.d_sampB, d.nSampPerFaceB, d.nFacesB,
        d.d_areasA,
        obsData, nObs,
        d_ws, d_ta);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back and compute F = weighted_sum / total_area_A
    double h_ws = 0.0, h_ta = 0.0;
    CUDA_CHECK(cudaMemcpy(&h_ws, d_ws, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_ta, d_ta, sizeof(double), cudaMemcpyDeviceToHost));
    return (h_ta < 1e-15) ? 0.0 : h_ws / h_ta;
}
