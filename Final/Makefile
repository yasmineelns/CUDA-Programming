# Makefile for convexe view-factor benchmark
#
# Builds:
#   1. libff.a   – reference analytical library (ffp.c + claussenp.c, -DDOUBLE)
#   2. convexe   – CUDA benchmark binary

REF_DIR  := ff_code
BUILD_DIR := ff_build

CC       := gcc
NVCC     := nvcc

# ── Reference library flags ──────────────────────────────────────────────────
REF_CFLAGS := -O2 -DDOUBLE -I$(REF_DIR) -fPIC -w
# -w silences old-style C warnings from the 1993 code

# ── CUDA / C++ benchmark flags ───────────────────────────────────────────────
NVCC_FLAGS := -O2 -std=c++14 \
              -I. -I$(REF_DIR) \
              -Xcompiler "-O2" \
              --generate-code arch=compute_75,code=sm_75 \
              --generate-code arch=compute_80,code=sm_80 \
              --generate-code arch=compute_86,code=sm_86 \
              --generate-code arch=compute_89,code=sm_89
LDFLAGS    := -lm -lcurand

# ── Targets ───────────────────────────────────────────────────────────────────
.PHONY: all clean run plot validate

all: convexe

# ── Step 1: compile reference objects with -DDOUBLE ──────────────────────────
$(BUILD_DIR)/claussenp.o: $(REF_DIR)/claussenp.c $(REF_DIR)/claussen.h | $(BUILD_DIR)
	$(CC) $(REF_CFLAGS) -c $< -o $@

$(BUILD_DIR)/ffp.o: $(REF_DIR)/ffp.c $(REF_DIR)/ff.h $(REF_DIR)/claussen.h | $(BUILD_DIR)
	$(CC) $(REF_CFLAGS) -c $< -o $@

# ── Step 2: archive into libff.a ─────────────────────────────────────────────
$(BUILD_DIR)/libff.a: $(BUILD_DIR)/claussenp.o $(BUILD_DIR)/ffp.o
	ar rcs $@ $^

# ── Step 3: compile and link benchmark ───────────────────────────────────────
convexe: convexe.cu geometry.h ff_wrapper.h gpu_utils.cuh det_gpu.cuh sto_gpu.cuh $(BUILD_DIR)/libff.a
	$(NVCC) $(NVCC_FLAGS) $< -L$(BUILD_DIR) -lff $(LDFLAGS) -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) convexe ff_vs_det *.csv *.png

run: convexe
	./convexe

plot: results_detailed.csv
	uv run plot_error.py
	uv run plot_latency.py
	uv run plot_error_vs_latency.py
