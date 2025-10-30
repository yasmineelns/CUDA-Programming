run_docker:
	docker run -it --rm -v $(pwd):$(pwd) -w $(pwd) --gpus all nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

setup:
	apt-get update
	apt-get install -y python3-pip libopenblas-dev

compile:
	nvcc -Wno-deprecated-gpu-targets mat_mul.cu -o mat_mul_cuda -Xcompiler -fopenmp -lblas -lcublas -lcusolver

benchmark:
	./mat_mul_cuda

plot:
	pip install -U uv pip
	uv run plot_results.py

clean:
	rm -f mat_mul_cuda