#pragma once

#include <cuda_runtime.h>

#include "util.h"
//#include "msim_Simulator.h"

namespace MultiGPUBrain
{

template<class T>
class MemSchedule//显存调度类
{
private:
    T *mBuffer;//主存中的存储缓冲区，完整的数据
    T *mgBuffer;//显存中的存储缓冲区
    SLNum mLength;//整体的长度
    SLNum mGPULen;//显存可以容纳的最大长度
    SLNum mGridSize;//mGPULen将整体切分成的块数量
    SLNum mGridPos;//当前在显存中的块下标
private:
    MemSchedule();
public:
    ~MemSchedule();
static MemSchedule* AllocateMemory(SLNum length,SLNum initGrid=0);//构造一个内存调度区

    void UpdateToHost();//将当前显存中的块内容更新至主存中
    void UpdateToGPU();//将主存中的当前块内容更新至显存中
    bool SwitchToGrid(SLNum pos);//将当前的块移出显存，并将指定的块移入显存

    SLNum GetLength();
    SLNum GetGPULen();
    SLNum GetGridSize();
    SLNum GetGridPos();
    SLNum GetGridOffset();//获取当前块的首元素在整体的下标
    T * GetBuffer(bool bUpdate=false);//bUpdate表示是否先更新了主存的内容再返回
    T * GetGPUBuffer(bool bUpdate=false);//bUpdate表示是否先更新了显存的内容再返回
    T & operator[](SLNum index);//取主存中的对应元素
};


template<class T>
MemSchedule<T>::MemSchedule()
:mBuffer(NULL)
,mgBuffer(NULL)
,mLength(0)
,mGPULen(0)
,mGridSize(0)
,mGridPos(0)
{

}

template<class T>
MemSchedule<T>::~MemSchedule()
{
    if(mBuffer)
        delete []mBuffer;

    if(mgBuffer)
        cudaFree(mgBuffer);
}

#define SPLIT_COUNT 5

template<class T>
MemSchedule<T> *MemSchedule<T>::AllocateMemory(SLNum length,SLNum initGrid)
{
    MemSchedule<T> *pRet;
    cudaError_t err;
    T *gBuf=NULL;
    T *buf=NULL;
    SLNum nGridLen=length;
    SLNum nSize,nLen;

    if(initGrid>0 && initGrid<length)
        nGridLen=initGrid;

    while(nGridLen>0)//不断缩小空间大小，以试探当前可以申请的内存空间大小
    {
        for(SLNum i=SPLIT_COUNT;i>0;i--)
        {
            nLen=nGridLen*i/SPLIT_COUNT;
            nSize=nLen*sizeof(T);
            err=cudaMalloc<T>(&gBuf,nSize);
	        cudaGetLastError();
            if(err==cudaSuccess)
            {
                nGridLen=nLen;
                break;
            }
        }
        
        if(err==cudaSuccess)
            break;
        nGridLen/=SPLIT_COUNT;
    }
    if(err!=cudaSuccess)
    {
        return NULL;
    }
    if(nGridLen<length)
    {   
        buf=new T[length];
        if(!buf)
            return NULL;
    }
    pRet=new MemSchedule<T>();
    pRet->mBuffer=buf;
    pRet->mgBuffer=gBuf;
    pRet->mLength=length;
    pRet->mGPULen=nGridLen;
    pRet->mGridSize=(length+nGridLen-1)/nGridLen;
    return pRet;
}

template<class T>
void MemSchedule<T>::UpdateToHost()
{
    SLNum size=mGPULen;
    if(!mBuffer)
        return;
    if(size>(mLength-mGridPos*mGPULen))
    {
        size=mLength-mGridPos*mGPULen;
    }
    cudaMemcpy(mBuffer+GetGridOffset(),mgBuffer,sizeof(T)*size,cudaMemcpyDeviceToHost);
}

template<class T>
void MemSchedule<T>::UpdateToGPU()
{
    SLNum size=mGPULen;
    if(!mBuffer)
        return;
    if(size>(mLength-mGridPos*mGPULen))
    {
        size=mLength-mGridPos*mGPULen;
    }
    cudaMemcpy(mgBuffer,mBuffer+GetGridOffset(),sizeof(T)*size,cudaMemcpyHostToDevice);
}

template<class T>
bool MemSchedule<T>::SwitchToGrid(SLNum pos)
{
    if(pos>=mGridSize)
        return false;
    if(pos==mGridPos)
        return true;

    UpdateToHost();
    mGridPos=pos;
    UpdateToGPU();
    return true;
}

template<class T>
SLNum MemSchedule<T>::GetLength()
{
    return mLength;
}

template<class T>
SLNum MemSchedule<T>::GetGPULen()
{
    return mGPULen;
}

template<class T>
SLNum MemSchedule<T>::GetGridSize()
{
    return mGridSize;
}

template<class T>
SLNum MemSchedule<T>::GetGridPos()
{
    return mGridPos;
}

template<class T>
SLNum MemSchedule<T>::GetGridOffset()
{
    return mGridPos*mGPULen;
}

template<class T>
T * MemSchedule<T>::GetBuffer(bool bUpdate)
{
    if(bUpdate)
        UpdateToHost();
    return mBuffer;
}

template<class T>
T * MemSchedule<T>::GetGPUBuffer(bool bUpdate)
{
    if(bUpdate)
        UpdateToGPU();
    return mgBuffer;
}

template<class T>
T & MemSchedule<T>::operator[](SLNum index)
{
    return mBuffer[index];
}

}