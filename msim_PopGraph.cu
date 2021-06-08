#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "util.h"
#include "msim_schedule.h"
#include "msim_Simulator.h"
#include "PopGraph.h"

__global__ void BuildAllToAllSynapse(SYNAPSE *synapses,SNum preOffset,SNum preCount,SNum postOffset,SNum postCount,SFNum weight,SFNum delay)
{
    SNum pre=blockIdx.x*blockDim.x+threadIdx.x;
    SNum post=blockIdx.y*blockDim.y+threadIdx.y;
    if(pre>=preCount || post>=postCount)
        return;
    SNum index=pre*postCount+post;
    synapses[index].preIndex=preOffset+pre;
    synapses[index].postIndex=postOffset+post;
    synapses[index].weight=weight;
    synapses[index].delay=delay;
}

__global__ void BuildOneToOneSynapse(SYNAPSE *synapses,SNum ncount,SNum preOffset,SNum postOffset,SFNum weight,SFNum delay)
{
    SNum index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=ncount)
        return;
    synapses[index].preIndex=preOffset+index;
    synapses[index].postIndex=postOffset+index;
    synapses[index].weight=weight;
    synapses[index].delay=delay;

}

__global__ void  SelectProbaItems(SLNum *selIndex,SNum ncount,SFNum proba,unsigned long seed)
{
    SNum index=blockIdx.x*blockDim.x+threadIdx.x;
    SFNum fRnd;
    if(index>=ncount)
        return;
    curandState localState;
    curand_init(seed,index,0,&localState);
    fRnd=curand_uniform(&localState);
    if(proba<fRnd)
        selIndex[index]=1;
    else
        selIndex[index]=0;
}

#define SUM_BLOCK_SIZE 512

__global__ void prefix_sum_within_block(SLNum *buf1,SLNum *buf2,SNum length)
{
    __shared__ SLNum temp[SUM_BLOCK_SIZE*2];
    SNum pin=0,pout=1;
    SNum index=blockIdx.x*blockDim.x+threadIdx.x;
    SNum tidx=threadIdx.x;
    for(;index<length;index+=blockDim.x*gridDim.x)
    {
        temp[tidx]=buf1[index];
        pin=0;
        pout=1;
        __syncthreads();
        for(SNum step=1;step<blockDim.x;step=step<<1)
        {
            temp[pout*SUM_BLOCK_SIZE+tidx]=temp[pin*SUM_BLOCK_SIZE+tidx];
            if(tidx>=step)
                temp[pout*SUM_BLOCK_SIZE+tidx]+=temp[pin*SUM_BLOCK_SIZE+tidx-step];
            pout=pout^1;
            pin=pin^1;
            __syncthreads();
        }
        buf2[index]=temp[pin*SUM_BLOCK_SIZE+tidx];
    }
}

__global__ void gather_last_per_block(SLNum *buf1,SLNum *bufGather,SNum length)
{
    SNum index=blockIdx.x*blockDim.x+threadIdx.x;
    SNum glen=(length+SUM_BLOCK_SIZE-1)/SUM_BLOCK_SIZE;
    for(;index<glen;index+=blockDim.x*gridDim.x)
    {
        if(index<glen)
            bufGather[index]=buf1[(index+1)*SUM_BLOCK_SIZE-1];
    }
}

__global__ void spread_sum_to_block(SLNum *buf1,SLNum *bufGather,SNum length)
{
    SNum index=blockIdx.x*blockDim.x+threadIdx.x;
    SNum gidx;
    for(;index<length;index+=blockDim.x*gridDim.x)
    {
        gidx=(index/SUM_BLOCK_SIZE)-1;
        if(gidx<0)
            continue;
        buf1[index]+=bufGather[gidx];
    }
}

__global__ void check_result(SLNum *buf,SNum len)
{
    SNum index=blockIdx.x*blockDim.x+threadIdx.x;
    for(;index>0 && index<len;index+=blockDim.x*gridDim.x)
    {
        if(buf[index-1]>=buf[index])
        {
            printf("check_result wrong [%d]=%d\n",index,buf[index]);
        }
    }
}

void gpuPrefixSum(SLNum *gBuf,SNum length,SNum gridSize)
{
    SNum glen=(length+SUM_BLOCK_SIZE-1)/SUM_BLOCK_SIZE;
    SLNum *gGather,*gBuf2;
    SNum gridSize1=(gridSize+SUM_BLOCK_SIZE-1)/SUM_BLOCK_SIZE;
    CUDACHECK(cudaMalloc((void **)&gBuf2,sizeof(SLNum)*length));
    prefix_sum_within_block<<<gridSize,SUM_BLOCK_SIZE>>>(gBuf,gBuf2,length);
    if(length<SUM_BLOCK_SIZE)
    {
        CUDACHECK(cudaMemcpy(gBuf,gBuf2,sizeof(SLNum)*length,cudaMemcpyDeviceToDevice));
        cudaFree(gBuf2);
        return;
    }
    CUDACHECK(cudaMalloc((void **)&gGather,sizeof(SLNum)*glen));
    gather_last_per_block<<<gridSize1,SUM_BLOCK_SIZE>>>(gBuf2,gGather,length);
    gpuPrefixSum(gGather,glen,gridSize1);
    spread_sum_to_block<<<gridSize,SUM_BLOCK_SIZE>>>(gBuf2,gGather,length);
    CUDACHECK(cudaMemcpy(gBuf,gBuf2,sizeof(SLNum)*length,cudaMemcpyDeviceToDevice));
    cudaFree(gGather);
    cudaFree(gBuf2);

}

__global__ void  BuildProbaSynapse(SYNAPSE *synapses,SLNum *selIndex,SNum preOffset,SNum preCount,SNum postOffset,SNum postCount,SFNum weight,SFNum delay,SNum GPUID)
{
    SNum ncount=preCount*postCount;
    SNum index=blockIdx.x*blockDim.x+threadIdx.x;
    SNum sidx;
    SNum pre,post;
    if(index>=ncount)
        return;
    pre=index/postCount;
    post=index%postCount;
    if(index==0)
    printf("build propa from (%d,%d) to (%d,%d),GPU%d\n",preOffset,preCount,postOffset,postCount,GPUID);
    if(index==0 && selIndex[0]!=0)
    {
        synapses[0].preIndex=preOffset;
        synapses[0].postIndex=postOffset;
        synapses[0].weight=weight;
        synapses[0].delay=delay;
        if(postOffset==0)
        printf("GPU%d SYN0 weight:%f\n",GPUID,weight);
    }
    else if(index>0 && selIndex[index]!=selIndex[index-1])
    {
        sidx=selIndex[index-1];
        synapses[sidx].preIndex=preOffset+pre;
        synapses[sidx].postIndex=postOffset+post;
        synapses[sidx].weight=weight;
        synapses[sidx].delay=delay;
        if(synapses[sidx].postIndex==0)
        printf("GPU%d SYN%d weight:%f\n",GPUID,sidx,weight);
    }
}

bool BuildSynapse(SYNAPSE *gSyns,SNum preOffset,SNum preCount,SNum postOffset,SNum postCount,SFNum weight,SFNum delay,const CONN_INFO *pConn)
{
    dim3 grid,block;
    SNum ncount=preCount*postCount;
    SLNum *gSelIndex;
    SNum GPUID;
    cudaGetDevice(&GPUID);
    printf("build synapse from (%d,%d) to (%d,%d),GPU%d\n",preOffset,preCount,postOffset,postCount,GPUID);
    if(pConn->bOneToOne)
    {
        if(preCount!=postCount)
            return false;
        BuildOneToOneSynapse<<<GRIDSIZE(preCount),BLOCKSIZE>>>(gSyns,preCount,preOffset,postOffset,weight,delay);
        return cudaGetLastError()==cudaSuccess;
    }
    if(pConn->fPropa>=1.0)
    {
        block.x=32;
        block.y=32;
        block.z=1;
        grid.x=(preCount+block.x-1)/block.x;
        grid.y=(postCount+block.y-1)/block.y;
        grid.z=1;
        BuildAllToAllSynapse<<<grid,block>>>(gSyns,preOffset,preCount,postOffset,postCount,weight,delay);
        CUDACHECK(cudaGetLastError());
        return true;
        //return cudaGetLastError()==cudaSuccess;
    }
    //构建概率突触
    CUDACHECK(cudaMalloc((void **)&gSelIndex,sizeof(SLNum)*ncount));
    SelectProbaItems<<<GRIDSIZE(ncount),BLOCKSIZE>>>(gSelIndex,ncount,pConn->fPropa,time(NULL));
    CUDACHECK(cudaGetLastError());
    gpuPrefixSum(gSelIndex,ncount,GRIDSIZE(ncount));
    BuildProbaSynapse<<<GRIDSIZE(ncount),BLOCKSIZE>>>(gSyns,gSelIndex,preOffset,preCount,postOffset,postCount,weight,delay,GPUID);
    CUDACHECK(cudaFree(gSelIndex));
    return cudaGetLastError()==cudaSuccess;
}

//获取指定突触连接组的指定前神经元的第一个全连接突触在总连接中的下标
__device__ SLNum GetSynStartIndex(SYN_BUILD *conn,SNum preIdx)
{
    SLNum ret=(SLNum)conn->postWholeCount*((SLNum)conn->preOffsetInWhole+(SLNum)preIdx);
    ret+=(SLNum)conn->postOffsetInWhole;
    return ret;
}

//确定单个族群中各个节点所需的突触数
__global__ void gpuGetSynCountPerNode(SLNum *countPerNodes,SNum nNodeCount,SYN_BUILD *conns,SNum nConnCount,SNum nAccuracy)
{
    SNum nodeIdx=blockDim.x*blockIdx.x+threadIdx.x;
    SNum connIdx=blockDim.y*blockIdx.y+threadIdx.y;
    SLNum idx1,idx2;
    double p;
    SNum n;
    if(nodeIdx>=nNodeCount || connIdx>=nConnCount)
        return;
    if(conns[connIdx].bOneToOne)
    {
        atomicAdd(&countPerNodes[nodeIdx],1);
    }
    else if(conns[connIdx].fPropa>=1.0)
    {
        atomicAdd(&countPerNodes[nodeIdx],conns[connIdx].postCount);
    }
    else
    {
        /*
        整数时，
        按比例p/a构建突触
        在全连接中下标为i到下标为j之前的突触中，符合按比例构建的突触数为(j*p-1-(i*p+a-1)/a*a+a)/a
        浮点数时，
        按比例p构建突触
        在全连接中下标为i到下标为j之前的突触中，符合按比例构建的突触数为ip,jp向上取整求差
        */
        p=conns[connIdx].fPropa;
        idx1=GetSynStartIndex(&conns[connIdx],0);
        idx2=idx1+conns[connIdx].postCount;
        n=(SNum)(ceil(p*(idx2-1))-ceil(p*idx1));
        atomicAdd(&countPerNodes[nodeIdx],n);
    }
}

//按一定粒度分配突触空间，从而留出一定的空闲空间增加新的突触
__global__ void MeshizeSynCount(SNum nNodes,SLNum *synCounts,SNum nMeshSize)
{
    SNum nodeIdx=blockDim.x*blockIdx.x+threadIdx.x;
    SNum m;
    if(nMeshSize<=1)
    return;

    while(nodeIdx<nNodes)
    {
        m=(synCounts[nodeIdx]+nMeshSize-1)/nMeshSize;
        synCounts[nodeIdx]=m*nMeshSize;
        nodeIdx+=blockDim.x*gridDim.x;
    }
}

//计算各个节点突触在数组中的偏移
bool GetSynSection(SNum nBuild,SYN_BUILD *builds,SNum *pNodeCount,SLNum **gSections,SNum nMeshSize)
{
    SNum nNodes=0;//总共的节点数
    SNum nPop=0;
    SNum p,n;
    SNum nPopOffset[nBuild+1]={0};
    SYN_BUILD *gBuilds;
    dim3 block(BLOCK2DSIZE,BLOCK2DSIZE);
    //检查builds数组合法性
    for(SNum i=1;i<nBuild;i++)
    {
        if(builds[i-1].preOffset>builds[i].preOffset)//突触前神经元节点编号必须升序
            return false;
        if(builds[i-1].preOffset==builds[i].preOffset)
        {
            if(builds[i-1].preCount!=builds[i].preCount)//同一组突触前神经元节点不能数量不同
                return false;
            if(builds[i-1].postOffset>builds[i].postOffset)//同一组突触前神经元的突触后神经元必须升序
                return false;
        }
    }
    //统计节点数
    for(SNum i=0;i<nBuild;i++)
    {
        if(builds[i].bOneToOne && builds[i].preCount!=builds[i].postCount)
            return false;
        if(i==0 || builds[i-1].preOffset!=builds[i].preOffset)
        {
            nPopOffset[nPop++]=i;
            nNodes+=builds[i].preCount;
        }
    }
    nPopOffset[nPop]=nBuild;
    *pNodeCount=nNodes;
    CUDACHECK(cudaMalloc((void **)gSections,sizeof(SLNum)*(nNodes+1)));
    CUDACHECK(cudaMemset(*gSections,0,sizeof(SLNum)*(nNodes+1)));

    CUDACHECK(cudaMalloc((void **)&gBuilds,sizeof(SYN_BUILD)*nBuild));
    CUDACHECK(cudaMemcpy(gBuilds,builds,sizeof(SYN_BUILD)*nBuild,cudaMemcpyHostToDevice));
    //逐个族群计算各个节点的突触数
    for(SNum i=0;i<nPop;i++)
    {
        p=nPopOffset[i];
        n=nPopOffset[i+1]-p;
        dim3 grid(GRID2DSIZE(builds[p].preCount),GRID2DSIZE(n));
        gpuGetSynCountPerNode<<<grid,block>>>(*gSections+builds[p].preOffset+1,builds[p].preCount,gBuilds+p,n,10000);
    }
    if(nMeshSize>1)
        MeshizeSynCount<<<GRIDSIZE(nNodes),BLOCKSIZE>>>(nNodes,*gSections,nMeshSize);
    gpuPrefixSum(*gSections,nNodes+1,GRIDSIZE(nNodes+1));
    CUDACHECK(cudaFree(gBuilds));
    return true;
}

//将突触数据清空成无效突触
__global__ void CleanSynapseData(SLNum nSyn,SYNAPSE *syns)
{
    SLNum synIdx=blockDim.x*blockIdx.x+threadIdx.x;

    while(synIdx<nSyn)
    {
        syns[synIdx].preIndex=-1;
        syns[synIdx].postIndex=-1;
        synIdx+=blockDim.x*gridDim.x;
    }
}

__device__ SNum GetDelta(SLNum a,SLNum b)
{
    SNum ret;
    if(a>b)
    {
        ret=a-b;
        if((a-b)>0x7fffffff)
        {
            printf("Wrong delta:%d\n",ret);
        }
    }
    else
    {
        ret=b-a;
        ret=-ret;
        if((b-a)>0x7fffffff)
        {
            printf("Wrong delta:%d\n",ret);
        }
    }
    return ret;
}

//构建以单个族群为突触前神经元的突触
__global__ void gpuBuildPopSynapse(SNum nNodeCount,SNum nBuild,SYN_BUILD *builds,SLNum *sections,SYNAPSE *syns,SLNum gpuOffset,SLNum gpuLen,SNum nAccuracy,SNum GPUID)
{
    SNum nodeIdx=blockDim.x*blockIdx.x+threadIdx.x;
    SNum offset,preStart;
    SNum i,j,n;
    SLNum pOffset,pos;
    SLNum idx1,idx2,xi;
    SLNum end;//当前神经元的输出突触数组的结束位置
    double p;

    if(nBuild<=0)
        return;

    if(nodeIdx>=builds[0].preCount)
        return;


    preStart=builds[0].preOffset;
    if((preStart+nodeIdx)>=nNodeCount)
    {
        printf("node overflow!\n");
        return;
    }
    if(sections[nodeIdx+preStart]>=(gpuOffset+gpuLen) || sections[nodeIdx+preStart+1]<gpuOffset)
        return;

    /*if(gpuOffset>0 && sections[nodeIdx+preStart]<=gpuOffset)
    {
        printf("node=%d,offset=%lu,gpuOffset=%lu\n",nodeIdx,sections[nodeIdx+preStart],gpuOffset);
    }*/
    offset=GetDelta(sections[nodeIdx+preStart],gpuOffset);
    end=GetDelta(sections[nodeIdx+preStart+1],gpuOffset);
    /*if(nodeIdx==0)
    {
        for(i=0;i<nBuild;i++)
        {
            printf("GPU%d building:%d,%d->%d,%d\n",GPUID,\
            builds[i].preOffset,builds[i].preCount,builds[i].postOffset,builds[i].postCount);
        }
    }*/
    //printf("n:%d,o:%d\n",nodeIdx,offset);
    for(i=0;i<nBuild;i++)
    {
        if(offset>=end || offset>=gpuLen)
        {
            //printf("offset=%d overflow at %d!\n",offset,i);
            break;
        }
        if(builds[i].bOneToOne)
        {
            if(offset>=0)
            {
                syns[offset].preIndex=builds[i].preOffset+nodeIdx;
                syns[offset].postIndex=builds[i].postOffset+nodeIdx;
                syns[offset].weight=builds[i].weight;
                syns[offset].delay=builds[i].delay;
            }
            offset++;
        }
        else if(builds[i].fPropa>=1.0)
        {
            for(j=0;j<builds[i].postCount;j++)
            {
                if((offset+j)<0)
                    continue;
                if((offset+j)>=gpuLen || (offset+j)>=end)
                {
                    //printf("Overflow[%lu-%lu] all2all,%lu,%lu\n",offset,j,end,gpuLen);
                    break;
                }
                syns[offset+j].preIndex=builds[i].preOffset+nodeIdx;
                syns[offset+j].postIndex=builds[i].postOffset+j;
                syns[offset+j].weight=builds[i].weight;
                syns[offset+j].delay=builds[i].delay;
            }
            offset+=builds[i].postCount;
        }
        else
        {
            /*
            整数时，
            按比例p/a构建突触
            最终构建的所有突触中下标为i的突触在全连接突触中的下标x(i)=i*a/p
            按照上述算法，已知在全连接中下标为i的下一个比例突触的下标为(i*p+a-1)/a
            浮点数时，
            按比例p构建突触
            最终构建的所有突触中下标为i的突触在全连接突触中的下标x(i)=ceil(i/p)
            按照上述算法，已知在全连接中下标为i的下一个比例突触的下标为ceil(ip)
            */
            p=builds[i].fPropa;
            idx1=GetSynStartIndex(&builds[i],nodeIdx);
            idx2=idx1+builds[i].postCount;
            pOffset=(SLNum)ceil(p*idx1);
            n=(SNum)(ceil(p*(idx2-1))-ceil(p*idx1));
            for(j=0,pos=pOffset;j<n;j++,pos++)
            {
                if((offset+j)<0)
                    continue;
                xi=(SLNum)ceil(pos/p);
                if(xi<idx1 || (xi-idx1)>builds[i].postCount)
                {
                printf("Wrong propa[%d] n=%d,idx1=%llu,pOffset=%llu:%llu,xi=%llu,pos=%llu,postcount=%d\n",\
                j,n,idx1,pOffset,(SLNum)ceil(p*idx1),xi,pos,builds[i].postCount);
                //if(gpuOffset>0)
                //printf("gpu offset:%llu\n",gpuOffset);
                }
                
                if((offset+j)>=end || (offset+j)>=gpuLen)
                {
                    //printf("Overflow[%d-%d] n=%d,idx1=%lu,pOffset=%d,xi=%ld,pos=%d,postcount=%d,sec%d-%d\n",\
                    i,j,n,idx1,pOffset,xi,pos,builds[i].postCount,offset,end);
                    break;
                }
                //atomicMax(maxSyn,offset+j);
                syns[offset+j].preIndex=builds[i].preOffset+nodeIdx;
                syns[offset+j].postIndex=builds[i].postOffset+xi-idx1;
                syns[offset+j].weight=builds[i].weight;
                syns[offset+j].delay=builds[i].delay;
                //if(syns[offset+j].postIndex==0)
                //printf("0Bingo!node%d,preWOffset:%d,postWCount:%d\n",nodeIdx,builds[i].preOffsetInWhole,builds[i].postWholeCount);
            }
            //lastOffset=offset;
            offset+=n;
            /*if(offset<0)
            {
                printf("offset error,last=%d,n=%d\n",lastOffset,n);
            }*/
        }
    }
}

bool BuildSynapse(SNum nBuild,SYN_BUILD builds[],SNum *pNodeCount,SLNum **gSections,MultiGPUBrain::MemSchedule<SYNAPSE> **gSynapses,SNum *nMaxSynCount,SLNum nGridSize,SNum nMeshSize)
{
    SNum nNodes=0;//总共的节点数
    SNum nPop=0;
    SNum p,n;
    SLNum nSyn;
    SLNum *sections;
    SNum nPopOffset[nBuild+1]={0};
    SYN_BUILD *gBuilds;
    SNum *gMax;
    dim3 block(BLOCK2DSIZE,BLOCK2DSIZE);
    SNum GPUID;
    //printf("start building synapses with %d groups\n",nBuild);
    cudaGetDevice(&GPUID);
    if(nBuild<=0)
    {
        *gSynapses=NULL;
        nNodes=*pNodeCount;
        CUDACHECK(cudaMalloc((void **)gSections,sizeof(SLNum)*(nNodes+1)));
        CUDACHECK(cudaMemset(*gSections,0,sizeof(SLNum)*(nNodes+1)));
        return true;
    }
    //检查builds数组合法性
    for(SNum i=1;i<nBuild;i++)
    {
        if(builds[i-1].preOffset>builds[i].preOffset)//突触前神经元节点编号必须升序
            return false;
        if(builds[i-1].preOffset==builds[i].preOffset)
        {
            //if(builds[i-1].preCount!=builds[i].preCount)//同一组突触前神经元节点不能数量不同
                //return false;
            if(builds[i-1].postOffset>builds[i].postOffset)//同一组突触前神经元的突触后神经元必须升序
                return false;
        }
    }
    //统计节点数
    for(SNum i=0;i<nBuild;i++)
    {
        if(builds[i].postWholeCount<builds[i].postCount)
        {
            fprintf(stderr,"postWholeCount=%d data error! postCount=%d\n",\
            builds[i].postWholeCount,builds[i].postCount);
            return false;
        }
        if(builds[i].bOneToOne && builds[i].preCount!=builds[i].postCount)
        {
            fprintf(stderr,"OneToOne synapse data error!\n");
            return false;
        }
        if(i==0 || builds[i-1].preOffset!=builds[i].preOffset)
        {
            nPopOffset[nPop++]=i;
            n=builds[i].preOffset+builds[i].preCount;
            if(nNodes<n)
                nNodes=n;
            //nNodes+=builds[i].preCount;
        }
        /*if(GPUID==0)
        {
			printf("build synapses:%d,%d->%d,%d\n",builds[i].preOffset,builds[i].preCount,builds[i].postOffset,builds[i].postCount);
        }*/
    }
    //printf("Build synapses within %d-%d nodes\n",nNodes,*pNodeCount);
    nPopOffset[nPop]=nBuild;
    if(nNodes>*pNodeCount)
        *pNodeCount=nNodes;
    else
        nNodes=*pNodeCount;
    CUDACHECK(cudaMalloc((void **)gSections,sizeof(SLNum)*(nNodes+1)));
    CUDACHECK(cudaMemset(*gSections,0,sizeof(SLNum)*(nNodes+1)));

    CUDACHECK(cudaMalloc((void **)&gBuilds,sizeof(SYN_BUILD)*nBuild));
    CUDACHECK(cudaMemcpy(gBuilds,builds,sizeof(SYN_BUILD)*nBuild,cudaMemcpyHostToDevice));
    //逐个族群计算各个节点的突触数
    for(SNum i=0;i<nPop;i++)
    {
        p=nPopOffset[i];
        n=nPopOffset[i+1]-p;
        dim3 grid(GRID2DSIZE(builds[p].preCount),GRID2DSIZE(n));
        gpuGetSynCountPerNode<<<grid,block>>>(*gSections+builds[p].preOffset+1,builds[p].preCount,gBuilds+p,n,10000);
    }
    if(nMeshSize>1)
        MeshizeSynCount<<<GRIDSIZE(nNodes),BLOCKSIZE>>>(nNodes,*gSections,nMeshSize);
    sections=new SLNum[nNodes+1];
    gpuPrefixSum(*gSections,nNodes+1,GRIDSIZE(nNodes+1));
    //check_result<<<GRIDSIZE(nNodes+1),SUM_BLOCK_SIZE>>>(*gSections,nNodes+1);
    //printf("node count:%d,build count:%d\n",nNodes,nBuild);
    CUDACHECK(cudaMemcpy(sections,*gSections,sizeof(SLNum)*(nNodes+1),cudaMemcpyDeviceToHost));
    nSyn=sections[nNodes];
    //求最大输出突触数
    if(nMaxSynCount)
    {
        *nMaxSynCount=0;
        for(SNum i=0;i<nNodes;i++)
        {
            n=sections[i+1]-sections[i];
            if(n>*nMaxSynCount)
                *nMaxSynCount=n;
        }
    }
    //printf("allocate synapses' buffer[%p] with size=%llu\n",gSynapses,nSyn);
    *gSynapses=MultiGPUBrain::MemSchedule<SYNAPSE>::AllocateMemory(nSyn,nGridSize);
    CUDACHECK(cudaGetLastError());
    if(!(*gSynapses))
    {
        delete []sections;
        cudaFree(gSections);
        cudaFree(gBuilds);
        return false;
    }
    //CUDACHECK(cudaMalloc((void **)gSynapses,sizeof(SYNAPSE)*nSyn));
    CUDACHECK(cudaMalloc((void **)&gMax,sizeof(SNum)));
    for(SLNum j=0;j<(*gSynapses)->GetGridSize();j++)
    {
        (*gSynapses)->SwitchToGrid(j);
        CleanSynapseData<<<GRIDSIZE((*gSynapses)->GetGPULen()),BLOCKSIZE>>>((*gSynapses)->GetGPULen(),(*gSynapses)->GetGPUBuffer());
        //逐个族群构建突触
        for(SNum i=0;i<nPop;i++)
        {
            p=nPopOffset[i];
            n=nPopOffset[i+1]-p;
            cudaMemset(gMax,0,sizeof(SNum));
            gpuBuildPopSynapse<<<GRIDSIZE(builds[p].preCount),BLOCKSIZE>>>(nNodes,n,gBuilds+p,\
                *gSections,(*gSynapses)->GetGPUBuffer(),(*gSynapses)->GetGridOffset(),\
                (*gSynapses)->GetGPULen(),10000,GPUID);
            cudaError_t err=cudaDeviceSynchronize();
            CUDACHECK(err);
        }

    }
    CUDACHECK(cudaFree(gMax));
    delete []sections;
    CUDACHECK(cudaFree(gBuilds));
    return true;
}

#ifdef TEST_PARALLELIZE_SYNAPSE

int main(int argc,char *argv[])
{
    SNum *gSections,*sections;
    SYNAPSE *gSynapses,*synapses;
    SNum preCount=0,postOffset;
    SNum OneToOne;
    SNum ncount,scount=0,nodeCount;
    scanf("%d",&ncount);
    scanf("%d",&preCount);
    postOffset=preCount;
    SYN_BUILD builds[ncount];
    for(SNum i=0;i<ncount;i++)
    {
        builds[i].preOffset=0;
        builds[i].preCount=preCount;
        builds[i].postOffset=postOffset;
        builds[i].weight=(SFNum)i+1.0;
        builds[i].delay=10.0;
        scanf("%d",&builds[i].postCount);
        postOffset+=builds[i].postCount;
        scanf("%d",&OneToOne);
        builds[i].bOneToOne=OneToOne!=0;
        scanf("%f",&builds[i].fPropa);
    }
    
    //if(!GetSynSection(ncount,builds,&nodeCount,&gSections))
    if(!BuildSynapse(ncount,builds,&nodeCount,&gSections,&gSynapses))
    {
        printf("fuck off!\n");
        cudaFree(gSections);
        return -1;
    }
    sections=new SNum[nodeCount+1];
    CUDACHECK(cudaMemcpy(sections,gSections,sizeof(SNum)*(nodeCount+1),cudaMemcpyDeviceToHost));
    for(SNum i=0;i<=nodeCount;i++)
    {
        printf("%d ",sections[i]);
    }
    printf("\n");
    scount=sections[nodeCount];
    synapses=new SYNAPSE[scount];
    CUDACHECK(cudaMemcpy(synapses,gSynapses,sizeof(SYNAPSE)*scount,cudaMemcpyDeviceToHost));
    printf("got %d synapses\n",scount);
    for(SNum i=0;i<scount && i<10;i++)
    {
        printf("syn%d(%d->%d)\n",i,synapses[i].preIndex,synapses[i].postIndex);
    }
    delete []sections;
    cudaFree(gSections);
    delete []synapses;
    cudaFree(gSynapses);

    return 0;
}

#endif