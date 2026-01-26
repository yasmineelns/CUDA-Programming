#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

double F_ref = 0.0685896;

#define CUDA_CHECK(err)                                            \
    {                                                              \
        if (err != cudaSuccess)                                    \
        {                                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " in " << __FILE__                        \
                      << " at line " << __LINE__ << std::endl;     \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    }

// ============================================================================
// CPU Implementations (Deterministic View Factor Calculation)
// ============================================================================

double estimateViewFactor_Seq(int N, double L, double d)
{
    double dx = L / N;
    double d2 = d * d;
    vector<double> coords(N);
    for (int i = 0; i < N; ++i)
        coords[i] = -L / 2.0 + (i + 0.5) * dx;

    double sum = 0.0;
    for (int i = 0; i < N; ++i)
    {
        double xA = coords[i];
        for (int j = 0; j < N; ++j)
        {
            double yA = coords[j];
            for (int k = 0; k < N; ++k)
            {
                double dx_sq = (coords[k] - xA) * (coords[k] - xA);
                for (int l = 0; l < N; ++l)
                {
                    double r2 = dx_sq + (coords[l] - yA) * (coords[l] - yA) + d2;
                    sum += d2 / (r2 * r2);
                }
            }
        }
    }
    return (pow(dx, 4) / (M_PI * L * L)) * sum;
}

double estimateViewFactor_OMP(int N, double L, double d)
{
    double dx = L / N;
    double d2 = d * d;
    vector<double> coords(N);
    for (int i = 0; i < N; ++i)
        coords[i] = -L / 2.0 + (i + 0.5) * dx;

    double sum = 0.0;
#pragma omp parallel for reduction(+ : sum) collapse(2)
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            double xA = coords[i];
            double yA = coords[j];
            for (int k = 0; k < N; ++k)
            {
                double dx_sq = (coords[k] - xA) * (coords[k] - xA);
                for (int l = 0; l < N; ++l)
                {
                    double r2 = dx_sq + (coords[l] - yA) * (coords[l] - yA) + d2;
                    sum += d2 / (r2 * r2);
                }
            }
        }
    }
    return (pow(dx, 4) / (M_PI * L * L)) * sum;
}

// ============================================================================
// GPU Kernels (Deterministic View Factor Calculation)
// ============================================================================

__global__ void vf_kernel_naive(int N, double L, double d2, double *result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N)
        return;
    double dx = L / N;
    double xA = -L / 2.0 + (idx / N + 0.5) * dx;
    double yA = -L / 2.0 + (idx % N + 0.5) * dx;
    for (int k = 0; k < N; ++k)
    {
        for (int l = 0; l < N; ++l)
        {
            double dx2 = pow((-L / 2.0 + (k + 0.5) * dx) - xA, 2);
            double dy2 = pow((-L / 2.0 + (l + 0.5) * dx) - yA, 2);
            double r2 = dx2 + dy2 + d2;
            atomicAdd(result, d2 / (r2 * r2));
        }
    }
}

__global__ void vf_kernel_optimized(int N, double L, double d2, double *result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N)
        return;
    double dx = L / N;
    double start = -L / 2.0 + 0.5 * dx;
    double xA = start + (idx / N) * dx;
    double yA = start + (idx % N) * dx;
    double local_sum = 0.0;
    for (int k = 0; k < N; ++k)
    {
        double dx_sq = (start + k * dx - xA);
        dx_sq *= dx_sq;
        for (int l = 0; l < N; ++l)
        {
            double dy_sq = (start + l * dx - yA);
            dy_sq *= dy_sq;
            double r2 = dx_sq + dy_sq + d2;
            local_sum += d2 / (r2 * r2);
        }
    }
    atomicAdd(result, local_sum);
}

// ============================================================================
// GPU Host Functions
// ============================================================================

// Run the CUDA kernel once and return the estimated view factor
double run_cuda_once(int N, double L, double d, double *d_result, int THREADS_PER_BLOCK, bool optimized)
{
    int BLOCKS = (N * N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    double d2 = d * d;

    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(double)));

    if (optimized)
        vf_kernel_optimized<<<BLOCKS, THREADS_PER_BLOCK>>>(N, L, d2, d_result);
    else
        vf_kernel_naive<<<BLOCKS, THREADS_PER_BLOCK>>>(N, L, d2, d_result);

    CUDA_CHECK(cudaDeviceSynchronize());

    double h_result = 0.0;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));

    double dx = L / N;
    return (pow(dx, 4) / (M_PI * L * L)) * h_result;
}

double estimateViewFactor_CUDA(int N, double L, double d, double *d_result,
                               int THREADS_PER_BLOCK, bool optimized, float &kernel_ms)
{
    int BLOCKS = (N * N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    double d2 = d * d;

    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(double)));

    // Create events for accurate kernel timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    if (optimized)
        vf_kernel_optimized<<<BLOCKS, THREADS_PER_BLOCK>>>(N, L, d2, d_result);
    else
        vf_kernel_naive<<<BLOCKS, THREADS_PER_BLOCK>>>(N, L, d2, d_result);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    double h_result = 0.0;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double dx = L / N;
    return (pow(dx, 4) / (M_PI * L * L)) * h_result;
}

int main()
{
    // ----------------------------
    // Problem Parameters
    // ----------------------------
    double L = 1.0; // plate size
    double d = 2.0; // separation distance
    // we take N values that give approx. the same number of operations as the stochastic version
    vector<int> N_values = {
        6,   // 6^4 ≈ 1,296 ≈ 1k
        10,  // 10^4 = 10,000 ≈ 10k
        18,  // 18^4 ≈ 104,976 ≈ 100k
        32,  // 32^4 ≈ 1,048,576 ≈ 1M
        56,  // 56^4 ≈ 9,834,496 ≈ 10M
        100, // 100^4 = 100,000,000 = 100M
        178, // 178^4 ≈ 1,002,313,296 ≈ 1B
    };

    // ----------------------------
    // Prepare and allocate CUDA resources once
    // ----------------------------
    int max_N = *max_element(N_values.begin(), N_values.end());

    const int THREADS_PER_BLOCK = 256;

    double *d_result = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_result, sizeof(double)));

    // Warm-up kernel to avoid measuring driver initialization overhead
    vf_kernel_optimized<<<1, THREADS_PER_BLOCK>>>(10, L, d * d, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ----------------------------
    // Time-based benchmarking with detailed CSV output
    // ----------------------------
    std::ofstream detailed_file("results_detailed.csv");
    detailed_file << "N,Function,Latency_ms,Estimation,Error\n";

    const double min_duration_ms = 1000.0;   // run measurements until this total time
    const double warmup_duration_ms = 200.0; // warmup period

    // Prepare OpenMP thread counts (powers of two)
    int max_threads = omp_get_max_threads();
    std::vector<int> thread_counts;
    for (int t = 1; t <= max_threads; t <<= 1)
    {
        thread_counts.push_back(t);
        if (t == 0)
            break;
    }

    for (int N : N_values)
    {
        cout << "\n============ N = " << N << " ============\n";

        // ----- CPU Sequential -----
        cout << "  Benchmarking CPU..." << flush;
        double warmup_time = 0.0;
        while (warmup_time < warmup_duration_ms)
        {
            auto s = chrono::high_resolution_clock::now();
            (void)estimateViewFactor_Seq(N, L, d);
            auto e = chrono::high_resolution_clock::now();
            warmup_time += chrono::duration<double, std::milli>(e - s).count();
        }

        std::vector<double> seq_measurements;
        std::vector<double> seq_estimates;
        std::vector<double> seq_errors;
        double total_time = 0.0;
        int iterations = 0;
        const int max_iterations = 1000;
        while (total_time < min_duration_ms && iterations < max_iterations)
        {
            auto s = chrono::high_resolution_clock::now();
            double F_seq = estimateViewFactor_Seq(N, L, d);
            auto e = chrono::high_resolution_clock::now();
            iterations++;
            double latency = chrono::duration<double, std::milli>(e - s).count();
            double err = fabs(F_seq - F_ref);
            seq_measurements.push_back(latency);
            seq_estimates.push_back(F_seq);
            seq_errors.push_back(err);
            total_time += latency;
        }
        // write detailed CSV lines for sequential
        for (size_t i = 0; i < seq_measurements.size(); ++i)
            detailed_file << N << ",CPU," << seq_measurements[i] << "," << seq_estimates[i] << "," << seq_errors[i] << "\n";
        cout << " Done (" << seq_measurements.size() << " samples)" << endl;

        // ----- OpenMP per-thread counts -----
        for (int tcount : thread_counts)
        {
            omp_set_num_threads(tcount);
            cout << "  Benchmarking OpenMP (t=" << tcount << ")..." << flush;
            warmup_time = 0.0;
            while (warmup_time < warmup_duration_ms)
            {
                auto s = chrono::high_resolution_clock::now();
                (void)estimateViewFactor_OMP(N, L, d);
                auto e = chrono::high_resolution_clock::now();
                warmup_time += chrono::duration<double, std::milli>(e - s).count();
            }

            std::vector<double> omp_measurements;
            std::vector<double> omp_estimates;
            std::vector<double> omp_errors;
            total_time = 0.0;
            iterations = 0;
            while (total_time < min_duration_ms && iterations < max_iterations)
            {
                auto s = chrono::high_resolution_clock::now();
                double F_omp = estimateViewFactor_OMP(N, L, d);
                auto e = chrono::high_resolution_clock::now();
                iterations++;
                double latency = chrono::duration<double, std::milli>(e - s).count();
                double err = fabs(F_omp - F_ref);
                omp_measurements.push_back(latency);
                omp_estimates.push_back(F_omp);
                omp_errors.push_back(err);
                total_time += latency;
            }
            for (size_t i = 0; i < omp_measurements.size(); ++i)
                detailed_file << N << ",OpenMP_t" << tcount << "," << omp_measurements[i] << "," << omp_estimates[i] << "," << omp_errors[i] << "\n";
            cout << " Done (" << omp_measurements.size() << " samples)" << endl;
        }

        // ----- CUDA Naive (host-timed runs) -----
        cout << "  Benchmarking CUDA Naive..." << flush;
        warmup_time = 0.0;
        while (warmup_time < warmup_duration_ms)
        {
            auto s = chrono::high_resolution_clock::now();
            (void)run_cuda_once(N, L, d, d_result, THREADS_PER_BLOCK, false);
            auto e = chrono::high_resolution_clock::now();
            warmup_time += chrono::duration<double, std::milli>(e - s).count();
        }

        std::vector<double> cuda_naive_measurements;
        std::vector<double> cuda_naive_estimates;
        std::vector<double> cuda_naive_errors;
        total_time = 0.0;
        iterations = 0;
        while (total_time < min_duration_ms && iterations < max_iterations)
        {
            auto s = chrono::high_resolution_clock::now();
            double F_cuda = run_cuda_once(N, L, d, d_result, THREADS_PER_BLOCK, false);
            auto e = chrono::high_resolution_clock::now();
            iterations++;
            double latency = chrono::duration<double, std::milli>(e - s).count();
            double err = fabs(F_cuda - F_ref);
            cuda_naive_measurements.push_back(latency);
            cuda_naive_estimates.push_back(F_cuda);
            cuda_naive_errors.push_back(err);
            total_time += latency;
        }
        for (size_t i = 0; i < cuda_naive_measurements.size(); ++i)
            detailed_file << N << ",CUDA_Naive," << cuda_naive_measurements[i] << "," << cuda_naive_estimates[i] << "," << cuda_naive_errors[i] << "\n";
        cout << " Done (" << cuda_naive_measurements.size() << " samples)" << endl;

        // ----- CUDA Optimized (host-timed runs) -----
        cout << "  Benchmarking CUDA Optimized..." << flush;
        warmup_time = 0.0;
        while (warmup_time < warmup_duration_ms)
        {
            auto s = chrono::high_resolution_clock::now();
            (void)run_cuda_once(N, L, d, d_result, THREADS_PER_BLOCK, true);
            auto e = chrono::high_resolution_clock::now();
            warmup_time += chrono::duration<double, std::milli>(e - s).count();
        }

        std::vector<double> cuda_opt_measurements;
        std::vector<double> cuda_opt_estimates;
        std::vector<double> cuda_opt_errors;
        total_time = 0.0;
        iterations = 0;
        while (total_time < min_duration_ms && iterations < max_iterations)
        {
            auto s = chrono::high_resolution_clock::now();
            double F_cuda = run_cuda_once(N, L, d, d_result, THREADS_PER_BLOCK, true);
            auto e = chrono::high_resolution_clock::now();
            iterations++;
            double latency = chrono::duration<double, std::milli>(e - s).count();
            double err = fabs(F_cuda - F_ref);
            cuda_opt_measurements.push_back(latency);
            cuda_opt_estimates.push_back(F_cuda);
            cuda_opt_errors.push_back(err);
            total_time += latency;
        }
        for (size_t i = 0; i < cuda_opt_measurements.size(); ++i)
            detailed_file << N << ",CUDA_Optimized," << cuda_opt_measurements[i] << "," << cuda_opt_estimates[i] << "," << cuda_opt_errors[i] << "\n";
        cout << " Done (" << cuda_opt_measurements.size() << " samples)" << endl;
    }

    detailed_file.close();

    // ----------------------------
    // Free CUDA resources
    // ----------------------------
    cudaFree(d_result);

    cout << "\nBenchmarks terminés !\n";
    cout << "Fichier généré : results_detailed.csv\n";

    return 0;
}