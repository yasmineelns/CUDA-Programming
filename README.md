# CUDA-Programming

## Reproduction

```bash
# run the cuda docker image
make run_docker

# install necessary packages (python and openblas)
make setup

# compile the benchmark code with nvcc, openmp, blas and cublas
make compile

# run the benchmark
make benchmark

# plot results
make plot
```

## Results

These results were obtained using a machine with the following hardware specifications :

- CPU : AMD EPYC 7742 (64 cores)
- GPU : NVIDIA A100-SXM4-80GB

![Benchmark Resultats](./benchmark_results.png)