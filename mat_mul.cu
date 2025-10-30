#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <omp.h>
#include <cblas.h>
#include <cublas_v2.h>

// Error checking macro
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

#define CUBLAS_CHECK(err)                                      \
    {                                                          \
        if (err != CUBLAS_STATUS_SUCCESS)                      \
        {                                                      \
            std::cerr << "CUBLAS error: " << err               \
                      << " in " << __FILE__                    \
                      << " at line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                           \
        }                                                      \
    }

// CPU implementation of matrix multiplication
void mat_mul_cpu(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C, int N)
{
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < N; ++i)
        {
            float value = 0.0f;
            for (int k = 0; k < N; ++k)
            {
                value += A[k * N + i] * B[j * N + k];
            }
            C[j * N + i] = value;
        }
    }
}

// OpenMP CPU implementation of matrix multiplication
void mat_mul_openmp(const std::vector<float> &A,
                    const std::vector<float> &B,
                    std::vector<float> &C,
                    int N)
{
#pragma omp parallel for collapse(2)
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < N; ++i)
        {
            float value = 0.0f;
            for (int k = 0; k < N; ++k)
            {
                value += A[k * N + i] * B[j * N + k];
            }
            C[j * N + i] = value;
        }
    }
}

// OpenBLAS implementation of matrix multiplication
void mat_mul_openblas(const std::vector<float> &A,
                      const std::vector<float> &B,
                      std::vector<float> &C,
                      int N)
{
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f, A.data(), N, B.data(), N, 0.0f, C.data(), N);
}

// GPU kernel for matrix multiplication
__global__ void mul_ma_kernel(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N)
    {
        float value = 0.0f;
        for (int k = 0; k < N; ++k)
        {
            value += A[k * N + i] * B[j * N + k];
        }
        C[j * N + i] = value;
    }
}

// GPU context to hold allocated memory
struct GPUContext
{
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    int N = 0;
    dim3 block;
    dim3 grid;

    void allocate(int matrix_size, const std::vector<float> &A, const std::vector<float> &B)
    {
        N = matrix_size;
        size_t bytes = N * N * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));

        // Copy data once during allocation
        CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));

        block = dim3(32, 32);
        grid = dim3((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    }

    void free()
    {
        if (d_A)
            cudaFree(d_A);
        if (d_B)
            cudaFree(d_B);
        if (d_C)
            cudaFree(d_C);
        d_A = d_B = d_C = nullptr;
    }

    ~GPUContext()
    {
        free();
    }
};

// GPU computation only (memory already allocated and copied)
void mul_mat_gpu(const std::vector<float> &A,
                 const std::vector<float> &B,
                 std::vector<float> &C,
                 GPUContext &ctx)
{
    mul_ma_kernel<<<ctx.grid, ctx.block>>>(ctx.d_A, ctx.d_B, ctx.d_C, ctx.N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// cuBLAS context to hold allocated memory and handle
struct CuBLASContext
{
    cublasHandle_t handle;
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    int N = 0;

    void allocate(int matrix_size, const std::vector<float> &A, const std::vector<float> &B)
    {
        N = matrix_size;
        CUBLAS_CHECK(cublasCreate(&handle));

        size_t bytes = N * N * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));

        // Copy data once during allocation
        CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void free()
    {
        if (d_A)
            cudaFree(d_A);
        if (d_B)
            cudaFree(d_B);
        if (d_C)
            cudaFree(d_C);
        d_A = d_B = d_C = nullptr;

        if (handle)
        {
            CUBLAS_CHECK(cublasDestroy(handle));
            handle = nullptr;
        }
    }

    ~CuBLASContext()
    {
        free();
    }
};

// cuBLAS computation only (memory already allocated and copied)
void mat_mul_cublas(const std::vector<float> &A,
                    const std::vector<float> &B,
                    std::vector<float> &C,
                    CuBLASContext &ctx)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             ctx.N, ctx.N, ctx.N, &alpha,
                             ctx.d_A, ctx.N,
                             ctx.d_B, ctx.N,
                             &beta,
                             ctx.d_C, ctx.N));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Benchmark a single function with time-based iterations
template <typename Func>
std::vector<double> benchmark_function(Func func, const std::vector<float> &A,
                                       const std::vector<float> &B,
                                       std::vector<float> &C, int N,
                                       double min_duration_ms,
                                       double warmup_duration_ms = 200.0)
{
    std::vector<double> measurements;

    // Warm-up runs (time-based)
    double warmup_time = 0.0;
    while (warmup_time < warmup_duration_ms)
    {
        auto start = std::chrono::high_resolution_clock::now();
        func(A, B, C, N);
        auto end = std::chrono::high_resolution_clock::now();
        warmup_time += std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Benchmark runs until minimum duration is reached
    double total_time = 0.0;
    while (total_time < min_duration_ms)
    {
        auto start = std::chrono::high_resolution_clock::now();
        func(A, B, C, N);
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        measurements.push_back(time_ms);
        total_time += time_ms;
    }

    return measurements;
}

// Specialized benchmark for GPU with pre-allocated context
template <typename Context, typename Func>
std::vector<double> benchmark_gpu_function(Func func, const std::vector<float> &A,
                                           const std::vector<float> &B,
                                           std::vector<float> &C, int N,
                                           double min_duration_ms,
                                           double warmup_duration_ms = 200.0)
{
    std::vector<double> measurements;

    // Allocate GPU memory once and copy data
    Context ctx;
    ctx.allocate(N, A, B);

    // Warm-up runs (time-based)
    double warmup_time = 0.0;
    while (warmup_time < warmup_duration_ms)
    {
        auto start = std::chrono::high_resolution_clock::now();
        func(A, B, C, ctx);
        auto end = std::chrono::high_resolution_clock::now();
        warmup_time += std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Benchmark runs until minimum duration is reached
    double total_time = 0.0;
    while (total_time < min_duration_ms)
    {
        auto start = std::chrono::high_resolution_clock::now();
        func(A, B, C, ctx);
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        measurements.push_back(time_ms);
        total_time += time_ms;
    }

    return measurements;
}

// Benchmark multiple sizes and export raw measurements to CSV only (no stats/analysis)
void benchmark_multiple_sizes(const std::vector<int> &sizes, double min_duration_ms = 1000.0)
{
    // Detailed raw measurements file only
    std::ofstream detailed_file("results_detailed.csv");
    detailed_file << "N,Function,Measurement_ms\n";

    for (int N : sizes)
    {
        std::cout << "\n=== Benchmarking N=" << N << " ===" << std::endl;

        std::vector<float> A(N * N), B(N * N);
        std::vector<float> C_cpu(N * N), C_openmp(N * N), C_openblas(N * N),
            C_gpu(N * N), C_cublas(N * N);

        for (int i = 0; i < N * N; ++i)
        {
            A[i] = static_cast<float>(rand() % 100);
            B[i] = static_cast<float>(rand() % 100);
        }

        // Benchmark CPU
        std::cout << "  Benchmarking CPU..." << std::flush;
        auto cpu_measurements = benchmark_function(mat_mul_cpu, A, B, C_cpu, N, min_duration_ms);
        std::cout << " Done (" << cpu_measurements.size() << " samples)" << std::endl;

        // Benchmark OpenMP (default thread setting)
        std::cout << "  Benchmarking OpenMP (default threads)..." << std::flush;
        auto openmp_measurements = benchmark_function(mat_mul_openmp, A, B, C_openmp, N, min_duration_ms);
        std::cout << " Done (" << openmp_measurements.size() << " samples)" << std::endl;

        // Benchmark OpenMP across powers-of-two thread counts up to omp_get_max_threads()
        int max_threads = omp_get_max_threads();
        std::vector<int> thread_counts;
        for (int t = 1; t <= max_threads; t <<= 1)
        {
            thread_counts.push_back(t);
            if (t == 0) // safety against overflow, though unlikely here
                break;
        }
        std::cout << "  Benchmarking OpenMP per-thread counts: ";
        for (size_t idx = 0; idx < thread_counts.size(); ++idx)
        {
            std::cout << thread_counts[idx] << (idx + 1 < thread_counts.size() ? ", " : "");
        }
        std::cout << "..." << std::flush;
        for (int t : thread_counts)
        {
            omp_set_num_threads(t);
            auto per_thread_meas = benchmark_function(mat_mul_openmp, A, B, C_openmp, N, min_duration_ms);
            // Write detailed measurements with thread-qualified label
            for (double measurement : per_thread_meas)
                detailed_file << N << ",OpenMP_t" << t << "," << measurement << "\n";
        }
        std::cout << " Done" << std::endl;

        // Benchmark OpenBLAS
        std::cout << "  Benchmarking OpenBLAS..." << std::flush;
        auto openblas_measurements = benchmark_function(mat_mul_openblas, A, B, C_openblas, N, min_duration_ms);
        std::cout << " Done (" << openblas_measurements.size() << " samples)" << std::endl;

        // Benchmark GPU (compute only, excluding allocation)
        std::cout << "  Benchmarking GPU..." << std::flush;
        auto gpu_measurements = benchmark_gpu_function<GPUContext>(mul_mat_gpu, A, B, C_gpu, N, min_duration_ms);
        std::cout << " Done (" << gpu_measurements.size() << " samples)" << std::endl;

        // Benchmark cuBLAS (compute only, excluding allocation)
        std::cout << "  Benchmarking cuBLAS..." << std::flush;
        auto cublas_measurements = benchmark_gpu_function<CuBLASContext>(mat_mul_cublas, A, B, C_cublas, N, min_duration_ms);
        std::cout << " Done (" << cublas_measurements.size() << " samples)" << std::endl;

        // Write detailed measurements (raw)
        for (double measurement : cpu_measurements)
            detailed_file << N << ",CPU," << measurement << "\n";
        for (double measurement : openmp_measurements)
            detailed_file << N << ",OpenMP," << measurement << "\n";
        for (double measurement : openblas_measurements)
            detailed_file << N << ",OpenBLAS," << measurement << "\n";
        for (double measurement : gpu_measurements)
            detailed_file << N << ",GPU," << measurement << "\n";
        for (double measurement : cublas_measurements)
            detailed_file << N << ",cuBLAS," << measurement << "\n";
    }

    detailed_file.close();
    std::cout << "\n✅ Raw benchmark measurements saved to results_detailed.csv" << std::endl;
}

int main()
{
    std::vector<int> sizes = {256, 512, 768, 1024, 1500};
    double min_duration_ms = 1000.0; // Run each function for at least 1 second

    benchmark_multiple_sizes(sizes, min_duration_ms);

    return 0;
}

// nvcc -Wno-deprecated-gpu-targets mat_mul.cu -o mat_mul_cuda -Xcompiler -fopenmp -lblas -lcublas -lcusolver && ./mat_mul_cuda