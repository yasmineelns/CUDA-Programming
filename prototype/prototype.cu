#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>
#include <omp.h>
#include <chrono>
#include <cuda_runtime.h>  // Gestion de la mémoire et de l'exécution GPU
#include <curand_kernel.h> // Générateur de nombres aléatoires sur GPU
#include <algorithm>

using namespace std;
#define M_PI 3.14159265358979323846

double F_ref = 0.0685896;

void randomPointOnSquare(double L, double &x, double &y, mt19937 &gen)
{
    uniform_real_distribution<double> dist(-L / 2.0, L / 2.0);
    x = dist(gen);
    y = dist(gen);
}

void randomCosineDirection(double &dx, double &dy, double &dz, mt19937 &gen)
{
    uniform_real_distribution<double> dist(0.0, 1.0);
    double r1 = dist(gen);
    double r2 = dist(gen);

    double phi = 2.0 * M_PI * r1;
    double z = sqrt(1.0 - r2);
    double r = sqrt(r2);

    dx = r * cos(phi);
    dy = r * sin(phi);
    dz = z;
}

bool intersectsSquareB(double Ax, double Ay, double Az,
                       double dx, double dy, double dz,
                       double L, double distZ)
{
    if (dz <= 0)
        return false;
    double t = (distZ - Az) / dz;
    if (t <= 0)
        return false;

    double X = Ax + t * dx;
    double Y = Ay + t * dy;

    return (abs(X) <= L / 2.0 && abs(Y) <= L / 2.0);
}

// --- Version CPU Séquentielle (Inchangée) ---

double estimateViewFactor_Seq(int N, double L, double d, mt19937 &gen)
{
    int hits = 0;
    for (int i = 0; i < N; i++)
    {
        double Ax, Ay, dx, dy, dz;
        randomPointOnSquare(L, Ax, Ay, gen);
        randomCosineDirection(dx, dy, dz, gen);
        if (intersectsSquareB(Ax, Ay, 0.0, dx, dy, dz, L, d))
            hits++;
    }
    return (double)hits / (double)N;
}

//  Version CPU Parallélisée avec OpenMP

double estimateViewFactor_OMP(int N, double L, double d)
{
    int hits = 0;

#pragma omp parallel reduction(+ : hits)
    {
        mt19937 gen(42 + omp_get_thread_num());
#pragma omp for
        for (int i = 0; i < N; i++)
        {
            double Ax, Ay, dx, dy, dz;
            randomPointOnSquare(L, Ax, Ay, gen);
            randomCosineDirection(dx, dy, dz, gen);

            if (intersectsSquareB(Ax, Ay, 0.0, dx, dy, dz, L, d))
                hits++;
        }
    }

    return (double)hits / (double)N;
}

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

__global__ void init_rng_states(curandState *state, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N)
    {
        curand_init(42, id, 0, &state[id]);
    }
}

__global__ void vf_kernel(int N, double L, double d, int *d_hits_out, curandState *rng_states)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N)
        return;
    curandState local_state = rng_states[id];

    // --- Génération de Point Aléatoire sur Plaque A ---
    double r_x = curand_uniform_double(&local_state);
    double r_y = curand_uniform_double(&local_state);

    double Ax = L * (r_x - 0.5);
    double Ay = L * (r_y - 0.5);
    double Az = 0.0;

    // --- Génération de Direction Lambertienne (Cosine) ---
    double r1 = curand_uniform_double(&local_state);
    double r2 = curand_uniform_double(&local_state);

    double phi = 2.0 * M_PI * r1;
    double z = sqrt(1.0 - r2);
    double r = sqrt(r2);

    double dx = r * cos(phi);
    double dy = r * sin(phi);
    double dz = z;

    // --- Test d'Intersection (Similaire à la fonction CPU) ---
    int hit = 0;
    if (dz > 0)
    {
        double t = (d - Az) / dz;
        if (t > 0)
        {
            double X = Ax + t * dx;
            double Y = Ay + t * dy;
            if (abs(X) <= L / 2.0 && abs(Y) <= L / 2.0)
            {
                hit = 1;
            }
        }
    }

    // Sauvegarde du résultat (compte atomique) et mise à jour de l'état RNG
    if (hit)
    {
        atomicAdd(d_hits_out, 1);
    }
    rng_states[id] = local_state;
}

// Run the CUDA kernel once and return the estimated view factor (host-timed caller will measure latency)
double run_cuda_once(int N, double L, double d, curandState *d_rng_states, int *d_counter, int THREADS_PER_BLOCK)
{
    int BLOCKS = (int)((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
    vf_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(N, L, d, d_counter, d_rng_states);
    CUDA_CHECK(cudaDeviceSynchronize());
    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    double F = (double)h_count / (double)N;
    return F;
}

// --- Fonction Hôte (Orchestration CUDA) ---
double estimateViewFactor_CUDA(int N, double L, double d,
                               curandState *d_rng_states, int *d_counter,
                               int THREADS_PER_BLOCK, float &kernel_ms)
{
    int BLOCKS = (int)((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // Reset device counter
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    // Create events for accurate kernel timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    vf_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(N, L, d, d_counter, d_rng_states);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    CUDA_CHECK(cudaDeviceSynchronize());

    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double F_est = (double)h_count / (double)N;
    return F_est;
}

int main()
{
    // ----------------------------
    // Paramètres du problème
    // ----------------------------
    double L = 1.0; // taille de la plaque
    double d = 2.0; // séparation demandée
    vector<int> N_values = {
        1'000,        // ~1k operations
        10'000,       // ~10k operations
        100'000,      // ~100k operations
        1'000'000,    // ~1M operations
        10'000'000,   // ~10M operations
        100'000'000,  // ~100M operations
        // 1'000'000'000 // ~1B operations
    };

    // ----------------------------
    // Générateur CPU pour séquentiel
    // ----------------------------
    mt19937 gen(42);

    // ----------------------------
    // Préparer et allouer les ressources CUDA une seule fois
    // ----------------------------
    int max_N = 0;
    for (int v : N_values)
        if (v > max_N)
            max_N = v;

    const int THREADS_PER_BLOCK = 256;
    int BLOCKS_MAX = (int)((max_N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    curandState *d_rng_states = nullptr;
    int *d_counter = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_rng_states, max_N * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc((void **)&d_counter, sizeof(int)));

    // Initialize RNG states for max_N threads
    init_rng_states<<<BLOCKS_MAX, THREADS_PER_BLOCK>>>(d_rng_states, max_N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Warm-up kernel to avoid measuring driver initialization overhead
    vf_kernel<<<1, THREADS_PER_BLOCK>>>(1, L, d, d_counter, d_rng_states);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ----------------------------
    // Boucle benchmark (time-based warm-up + repeated runs)
    // Save detailed per-run results to CSV file
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

        // ----- CPU -----
        cout << "  Benchmarking CPU..." << flush;
        double warmup_time = 0.0;
        while (warmup_time < warmup_duration_ms)
        {
            auto s = chrono::high_resolution_clock::now();
            (void)estimateViewFactor_Seq(N, L, d, gen);
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
            double F_seq = estimateViewFactor_Seq(N, L, d, gen);
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

        // ----- CUDA (host-timed runs) -----
        cout << "  Benchmarking CUDA..." << flush;
        warmup_time = 0.0;
        while (warmup_time < warmup_duration_ms)
        {
            auto s = chrono::high_resolution_clock::now();
            (void)run_cuda_once(N, L, d, d_rng_states, d_counter, THREADS_PER_BLOCK);
            auto e = chrono::high_resolution_clock::now();
            warmup_time += chrono::duration<double, std::milli>(e - s).count();
        }

        std::vector<double> cuda_measurements;
        std::vector<double> cuda_estimates;
        std::vector<double> cuda_errors;
        total_time = 0.0;
        iterations = 0;
        while (total_time < min_duration_ms && iterations < max_iterations)
        {
            auto s = chrono::high_resolution_clock::now();
            double F_cuda = run_cuda_once(N, L, d, d_rng_states, d_counter, THREADS_PER_BLOCK);
            auto e = chrono::high_resolution_clock::now();
            iterations++;
            double latency = chrono::duration<double, std::milli>(e - s).count();
            double err = fabs(F_cuda - F_ref);
            cuda_measurements.push_back(latency);
            cuda_estimates.push_back(F_cuda);
            cuda_errors.push_back(err);
            total_time += latency;
        }
        for (size_t i = 0; i < cuda_measurements.size(); ++i)
            detailed_file << N << ",CUDA," << cuda_measurements[i] << "," << cuda_estimates[i] << "," << cuda_errors[i] << "\n";
        cout << " Done (" << cuda_measurements.size() << " samples)" << endl;
    }

    detailed_file.close();

    // ----------------------------
    // Libération des ressources CUDA
    // ----------------------------
    cudaFree(d_rng_states);
    cudaFree(d_counter);

    cout << "\nBenchmarks terminés !\n";
    cout << "Fichier généré : results_detailed.csv\n";

    return 0;
}
