/**
 * convexe.cu  –  benchmarking harness (main only)
 *
 * Geometry:
 *   - Two regular icosahedra (20 triangular faces):
 *       A: radius=1, centred at (0,0,0)
 *       B: radius=1, centred at (0,0,DIST)
 *          gap between surfaces = DIST - 2*RADIUS = 3
 *   - Phase 2: a box obstacle placed midway between them causes partial
 *     occlusion.
 *
 * Each surface is already triangulated by generateIcosahedron().
 * Both GPU methods work on any triangulated surface.
 *
 * Methods benchmarked (error + latency vs N):
 *   CUDA_Det  –  see det_gpu.cuh
 *   CUDA_Sto  –  see sto_gpu.cuh
 *
 * Ground truth:
 *   Phase 1: FormFactor() from reference_code/ffp.c (see ff_wrapper.h),
 *            applied directly on the 20-face mesh (no subdivision needed).
 *   Phase 2: FormFactor() on a subdivided mesh (subdivisions=3 → 1280
 *            sub-faces per icosahedron) with per-sub-face centroid occlusion
 *            rejection against the obstacle (see ff_wrapper.h).
 *
 * Output: results_detailed.csv
 *   Columns: N, Function, Latency_ms, Estimation, Error_abs, Error_rel
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <functional>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "geometry.h"
#include "ff_wrapper.h"
#include "det_gpu.cuh"
#include "sto_gpu.cuh"

// CUDA_CHECK is already defined in det_gpu.cuh (pulled in transitively through sto_gpu.cuh)

// All lengths are in metres (or any consistent unit).
static const double RADIUS   = 1.0;            // circumradius of both icosahedra
static const double RADIUS_B = RADIUS;         // B same size as A
static const double DIST     = 5.0;            // centre-to-centre distance; surface gap = DIST - 2*RADIUS = 3
static const double OBS_HX   = 0.3;            // obstacle box half-extents (x)
static const double OBS_HY   = 0.3;            // obstacle box half-extents (y)
static const double OBS_HZ   = 0.3;            // obstacle box half-extents (z)

// =============================================================================
// BENCHMARKING HARNESS
// =============================================================================

static void benchmarkMethod(
    std::ofstream& csv, const std::string& label, int N, double F_ref,
    double warmup_ms, double measure_ms, int max_iter,
    std::function<double()> run_fn)
{
    // Warmup: run the kernel until at least warmup_ms have elapsed.
    // This flushes GPU caches, brings the device to peak clock, and
    // amortises any one-time JIT compilation cost before actual timing.
    double t_warm = 0.0;
    while (t_warm < warmup_ms) {
        auto t0 = std::chrono::high_resolution_clock::now();
        (void)run_fn();
        t_warm += std::chrono::duration<double,std::milli>(
                      std::chrono::high_resolution_clock::now()-t0).count();
    }
    // Measurement: collect individual timed runs until the budget is spent.
    double t_total = 0.0; int iter = 0;
    while (t_total < measure_ms && iter < max_iter) {
        auto t0 = std::chrono::high_resolution_clock::now();
        double est = run_fn();
        double lat = std::chrono::duration<double,std::milli>(
                         std::chrono::high_resolution_clock::now()-t0).count();
        double err_abs = std::fabs(est - F_ref);
        double err_rel = (F_ref > 1e-15) ? err_abs/F_ref : err_abs;
        csv << N << "," << label << "," << lat << "," << est << ","
            << err_abs << "," << err_rel << "\n";
        t_total += lat; ++iter;
    }
    std::cout << "  " << label << " N=" << N << ": " << iter << " samples\n" << std::flush;
}

// =============================================================================
// MAIN
// =============================================================================

int main()
{
    // generateIcosahedron returns a vector<Triangle> (already triangulated).
    auto facesA   = generateIcosahedron(0.0, 0.0, 0.0,      RADIUS);
    auto facesB   = generateIcosahedron(0.0, 0.0, DIST,     RADIUS_B);
    auto obstacle = generateBox(0.0, 0.0, DIST/2.0, OBS_HX, OBS_HY, OBS_HZ);

    // upload obstacle triangles to GPU once
    std::vector<double> obsFlat(obstacle.size()*9);
    for (int o = 0; o < (int)obstacle.size(); ++o)
        for (int v = 0; v < 3; ++v)
            for (int c = 0; c < 3; ++c)
                obsFlat[o*9+v*3+c] = obstacle[o][v][c];
    double* d_obsData = nullptr;
    CUDA_CHECK(cudaMalloc(&d_obsData, obsFlat.size()*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_obsData, obsFlat.data(), obsFlat.size()*sizeof(double), cudaMemcpyHostToDevice));
    int nObs = (int)obstacle.size();

    // ── Reference values ──────────────────────────────────────────────────────
    std::cout << "Computing reference view factors...\n";

    // No subdivision needed for P1: FormFactor() computes the exact analytical
    // integral over each face pair, so the only approximation is the centroid-
    // level cosine visibility pre-filter — which is exact for convex geometry
    // with no obstacle (each face pair is either fully visible or fully back-
    // facing; no partial occlusion can make the centroid test misleading).
    double F_ref_p1 = exactViewFactorPolyhedra(facesA, facesB);
    std::cout << "  Phase 1 (no occlusion): F_ref = " << F_ref_p1 << "\n";
    if (F_ref_p1 < 1e-6)
        std::cerr << "  WARNING: F_ref_p1 suspiciously small – rebuild may be needed.\n";

    // For P2, subdivision is needed: the obstacle can partially occlude a face
    // pair, so the centroid-to-centroid visibility test on the original large
    // faces is a poor approximation. subdivisions=3 → 64 sub-faces each →
    // 1280×1280 = 1.6M sub-face pairs with per-sub-face centroid occlusion test.
    double F_ref_p2 = exactViewFactorPolyhedra(facesA, facesB, /*subdivisions=*/3, obstacle);
    std::cout << "  Phase 2 (with obstacle): F_ref = " << F_ref_p2 << "\n\n";

    // ── Sweep parameters ──────────────────────────────────────────────────────
    std::vector<int> N_det = {1, 2, 3, 5, 8, 12, 16};
    std::vector<int> N_sto = {1000, 10000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000, 500000000};
    const double warmup_ms = 200.0, measure_ms = 1000.0;
    const int    max_iter  = 5000;

    double* d_ws   = nullptr, *d_ta = nullptr;
    int*    d_hits = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ws,   sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ta,   sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_hits, sizeof(int)));

    std::ofstream csv("results_detailed.csv");
    csv << "N,Function,Latency_ms,Estimation,Error_abs,Error_rel\n";

    // ── Phase 1: no occlusion ─────────────────────────────────────────────────
    // Pass nullptr / 0 for the obstacle data; both kernels skip the occlusion
    // loop entirely when nObs == 0, so there is no runtime penalty versus a
    // dedicated non-occlusion kernel.
    std::cout << "=== PHASE 1: no occlusion ===\n";
    for (int N : N_det) {
        DetGPUData dgpu = buildDetGPUData(facesA, facesB, N);
        for (int w = 0; w < 3; ++w) det_GPU_run(dgpu, d_ws, d_ta, nullptr, 0);
        benchmarkMethod(csv, "P1_CUDA_Det", N, F_ref_p1, warmup_ms, measure_ms, max_iter,
            [&]{ return det_GPU_run(dgpu, d_ws, d_ta, nullptr, 0); });
        freeDetGPUData(dgpu);
    }
    {
        StoGPUData sgpu = buildStoGPUData(facesA, facesB);
        for (int w = 0; w < 3; ++w) sto_GPU_run(sgpu, d_hits, nullptr, 0, 1000);
        for (int N : N_sto)
            benchmarkMethod(csv, "P1_CUDA_Sto", N, F_ref_p1, warmup_ms, measure_ms, max_iter,
                [&]{ return sto_GPU_run(sgpu, d_hits, nullptr, 0, N); });
        freeStoGPUData(sgpu);
    }

    // ── Phase 2: with obstacle ────────────────────────────────────────────────
    // Pass d_obsData / nObs; each sample ray is now tested against the obstacle
    // triangles before being counted (stochastic) or accumulated (deterministic).
    std::cout << "\n=== PHASE 2: with obstacle ===\n";
    for (int N : N_det) {
        DetGPUData dgpu = buildDetGPUData(facesA, facesB, N);
        for (int w = 0; w < 3; ++w) det_GPU_run(dgpu, d_ws, d_ta, d_obsData, nObs);
        benchmarkMethod(csv, "P2_CUDA_Det", N, F_ref_p2, warmup_ms, measure_ms, max_iter,
            [&]{ return det_GPU_run(dgpu, d_ws, d_ta, d_obsData, nObs); });
        freeDetGPUData(dgpu);
    }
    {
        StoGPUData sgpu = buildStoGPUData(facesA, facesB);
        for (int w = 0; w < 3; ++w) sto_GPU_run(sgpu, d_hits, d_obsData, nObs, 1000);
        for (int N : N_sto)
            benchmarkMethod(csv, "P2_CUDA_Sto", N, F_ref_p2, warmup_ms, measure_ms, max_iter,
                [&]{ return sto_GPU_run(sgpu, d_hits, d_obsData, nObs, N); });
        freeStoGPUData(sgpu);
    }

    cudaFree(d_ws); cudaFree(d_ta); cudaFree(d_hits); cudaFree(d_obsData);
    csv.close();
    std::cout << "\nDone. Results in results_detailed.csv\n";
    return 0;
}
