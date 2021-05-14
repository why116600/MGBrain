
CC=nvcc
CUDA_PATH=/usr/local/cuda-10.1/
MACRO_DEFINE=AKE_MULTI_GPU
C_PREFIX=
CUDA_PREFIX=
TEST_PATH=test

libMGPUBrain.so:sim_LIF.o msim_*.o msim_*.cpp util.o cuda_model_link.o PopGraph.cpp
	g++ ${C_PREFIX} -o $@ $^ -fopenmp -std=c++14 -fPIC -shared -I${CUDA_PATH}include -L${CUDA_PATH}lib64  -lcudadevrt -lcudart -D ${MACRO_DEFINE}

cuda_model_link.o:util.o sim_LIF.o msim_*.o
	${CC}  -dlink -o $@ $^ -std=c++14  -rdc=true --compiler-options "-fPIC -shared"  -lcudadevrt -lcudart

msim_*.o:msim_*.cu
	${CC} ${C_PREFIX} ${CUDA_PREFIX} -dc $^ -std=c++14  -rdc=true --compiler-options "-fPIC -shared" -D ${MACRO_DEFINE}

sim_LIF.o:sim_LIF.cu
	${CC}  -dc $^ -std=c++14  -rdc=true --compiler-options "-fPIC -shared"

 util.o: util.cu
	${CC}  -dc $^ -std=c++14  -rdc=true --compiler-options "-fPIC -shared"

