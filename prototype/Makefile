run_docker:
	docker run -it --rm -v $(pwd):$(pwd) -w $(pwd) --gpus all nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

setup:
	apt-get update
	apt-get install -y python3-pip libopenblas-dev

compile:
	nvcc -Wno-deprecated-gpu-targets prototype.cu -o prototype -Xcompiler -fopenmp

benchmark:
	./prototype

plot:
	pip install -U uv pip
	uv run plot_latency.py
	uv run plot_error.py

clean:
	rm -f prototype