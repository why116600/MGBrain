#pragma once

#define STEP_SIZE (1<<8)

struct RECORDER
{
	SNum maxLength;
	SNum length;
	SFNum *data;
};

__device__ void ResizeRecorder(RECORDER *pRecord,SNum len);
__global__ void HostResizeRecorder(RECORDER *pRecord,SNum len);
__device__ void AddToRecorder(RECORDER *pRecord,SFNum f);
__global__ void CopyRecorderData(RECORDER *pRecord,SFNum *data);

void GetRecorderData(RECORDER *gpu,RECORDER **host);
void FreeHostRecorder(RECORDER *recorder);
