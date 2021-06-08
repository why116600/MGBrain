#pragma once


struct RECORDER;

__global__ void initLIF(GNLIF *node,LIF_ARG *args,SNum count);
__device__ bool simulateLIF(SNum index,SFNum input,GNLIF *node,LIF_ARG *args,SNum now,SFNum timestep,RECORDER *recorder);
//__global__ void prepareLIF(GNLIF *node,NLIF *src);
