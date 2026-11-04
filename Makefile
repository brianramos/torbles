# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
# torbles 0.8.0
#   make            CPU engine (torbles/libalg.so) and the CPU emulation of the GPU kernels (torbles/libgpu_emu.so)
#   make gpu        CUDA executor (torbles/libgpu.so); falls back to PTX for older nvcc, and to the __popc build if
#                   nvcc rejects the b1 tensor-core MMA for this GPU
#   make test       every correctness test (CPU engine, plain baseline, DuckDB, GPU kernels: real if built, else emulated)
#   make ssb SF=1   official Star Schema Benchmark: generate, DuckDB, torbles (needs git, a C compiler, duckdb)
#   make clean
CC ?= gcc
CXX ?= g++
NVCC ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
PY ?= python3
SF ?= 1
T := torbles
export PYTHONPATH := $(CURDIR):$(CURDIR)/torbles:$(CURDIR)/tests

# the mathematical modules: every torbles/math*.c in the engine library, every torbles/math_*_gpu.cu in the GPU one
MATHC := $(sort $(wildcard $(T)/math*.c))
MATHH := $(sort $(wildcard $(T)/math*.h))
MATHCU := $(sort $(wildcard $(T)/math_*_gpu.cu))

all: $(T)/libalg.so $(T)/libgpu_emu.so $(T)/libmathgpu_emu.so

$(T)/libalg.so: $(T)/alg.c $(T)/torble.h $(T)/version.h $(T)/relationships.c $(T)/relationships_optimizer.c $(T)/relationships_system.c $(T)/relationships_dag.c $(T)/relationships_stage.c $(T)/relationships.h $(MATHC) $(MATHH)
	$(CC) -O2 -march=native -shared -fPIC $(T)/alg.c $(T)/relationships.c $(T)/relationships_optimizer.c $(T)/relationships_system.c $(T)/relationships_dag.c $(T)/relationships_stage.c $(MATHC) -o $@ -lpthread -lm

$(T)/libmathgpu_emu.so: $(MATHCU) $(MATHH)
	$(CXX) -O2 -fwrapv -DEMULATE -x c++ -shared -fPIC $(MATHCU) -o $@

$(T)/libgpu_emu.so: $(T)/gpu.cu $(T)/torble.h $(T)/version.h
	$(CXX) -O2 -fwrapv -DEMULATE -x c++ -shared -fPIC $(T)/gpu.cu -o $@

gpu: $(T)/libalg.so
	@cd $(T) && ( $(NVCC) -O3 -arch=native -shared -Xcompiler -fPIC gpu.cu -o libgpu.so 2>nvcc.err && echo "libgpu.so: native build" ) \
	 || ( $(NVCC) -O3 -gencode arch=compute_90,code=compute_90 -shared -Xcompiler -fPIC gpu.cu -o libgpu.so 2>>nvcc.err && echo "libgpu.so: PTX build (compute_90)" ) \
	 || ( $(NVCC) -O3 -gencode arch=compute_80,code=compute_80 -shared -Xcompiler -fPIC gpu.cu -o libgpu.so 2>>nvcc.err && echo "libgpu.so: PTX build (compute_80)" ) \
	 || ( $(NVCC) -O3 -DNO_B1MMA -shared -Xcompiler -fPIC gpu.cu -o libgpu.so 2>>nvcc.err && echo "libgpu.so: built without the b1 MMA" ) \
	 || ( cat nvcc.err; exit 1 )
	@cd $(T) && ( $(NVCC) -O3 -arch=native -shared -Xcompiler -fPIC $(notdir $(MATHCU)) -o libmathgpu.so 2>nvcc_math.err && echo "libmathgpu.so: native build" ) \
	 || ( $(NVCC) -O3 -gencode arch=compute_80,code=compute_80 -shared -Xcompiler -fPIC $(notdir $(MATHCU)) -o libmathgpu.so 2>>nvcc_math.err && echo "libmathgpu.so: PTX build (compute_80)" ) \
	 || ( cat nvcc_math.err; exit 1 )
	@$(PY) -c "import algq; t = algq.Table('probe', {'a': __import__('numpy').arange(1000, dtype='uint32')}); t.gpu_handle(); print('GPU:', algq.gpu_name()); algq.gpu_mm('mma'); print('tensor cores:', algq.gpu_mm()); algq.gpu_mm('word')"

test: all
	@bash tests/run_all.sh

ssb: all
	cd bench/ssb && $(PY) ssb_common.py $(SF) && $(PY) ssb_duck.py $(SF) && $(PY) ssb_torbles.py $(SF) derived

clean:
	rm -f $(T)/*.so $(T)/nvcc.err $(T)/nvcc_math.err
	find . -name __pycache__ -prune -exec rm -rf {} +

.PHONY: all gpu test ssb clean
