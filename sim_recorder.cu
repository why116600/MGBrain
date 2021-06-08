
#include <stdio.h>
#include <cuda_runtime.h>

#include "util.h"
#include "sim_simulator.h"
#include "sim_recorder.h"


__device__ void ResizeRecorder(RECORDER *pRecord,SNum len)
{
	SFNum *buffer;
	SNum newLen;
	if(len<=0)
	{
		if(pRecord->data)
			delete []pRecord->data;
		pRecord->data=NULL;
		pRecord->maxLength=0;
		pRecord->length=0;
		return;
	}
	newLen=((len+STEP_SIZE-1)/STEP_SIZE)*STEP_SIZE;
	if(newLen==pRecord->maxLength)
		return;
	pRecord->maxLength=newLen;
	buffer=new SFNum[pRecord->maxLength];
	if(pRecord->data)
	{
		if(pRecord->length<len)
			memcpy(buffer,pRecord->data,pRecord->length*sizeof(SFNum));
		else
			memcpy(buffer,pRecord->data,len*sizeof(SFNum));
		delete []pRecord->data;
	}
	pRecord->length=len;
	pRecord->data=buffer;
}

__global__ void HostResizeRecorder(RECORDER *pRecord,SNum len)
{
	unsigned int index=blockIdx.x*blockDim.x+threadIdx.x;
	ResizeRecorder(&pRecord[index],len);
}

__device__ void AddToRecorder(RECORDER *pRecord,SFNum f)
{
	if(pRecord->length>=pRecord->maxLength)
		ResizeRecorder(pRecord,pRecord->length+1);
	else
		pRecord->length++;
	pRecord->data[pRecord->length-1]=f;
}

__global__ void CopyRecorderData(RECORDER *pRecord,SFNum *data)
{
	unsigned int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(pRecord->length<=index)
		return;
	data[index]=pRecord->data[index];
}

void GetRecorderData(RECORDER *gpu,RECORDER **host)
{
	SFNum *gData;
	*host=new RECORDER;
	CUDACHECK(cudaMemcpy(*host,gpu,sizeof(RECORDER),cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMalloc((void **)&gData,sizeof(SFNum)*(*host)->length));
	if((*host)->length<=0)
	{
		(*host)->data=NULL;
		return;
	}
	dim3 block(BLOCKSIZE);
	dim3 grid(GRIDSIZE((*host)->length));
	CopyRecorderData<<<grid,block>>>(gpu,gData);
	CUDACHECK(cudaGetLastError());
	CUDACHECK(cudaDeviceSynchronize());
	(*host)->data=new SFNum[(*host)->maxLength];
	CUDACHECK(cudaMemcpy((*host)->data,gData,sizeof(SFNum)*(*host)->length,cudaMemcpyDeviceToHost));
}

void FreeHostRecorder(RECORDER *recorder)
{
	if(recorder->data)
		delete[] recorder->data;
	delete recorder;
}

