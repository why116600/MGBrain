#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "util.h"
#include "msim_schedule.h"
#include "PopGraph.h"
#include "msim_SynManage.h"
#include "msim_network.h"
#include "msim_Simulator.h"

#include "sim_LIF.h"

/*#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif*/

void SetToGPU(void **obj,void *src,unsigned long long length)
{
	if(*obj)
	{
		cudaFree(*obj);
		*obj=NULL;
	}
	if(src && length>0)
	{
		CUDACHECK(cudaMalloc(obj,length));
		CUDACHECK(cudaMemcpy(*obj,src,length,cudaMemcpyHostToDevice));
	}
}


__device__ bool simulateSpikeGen(SNum index,GNGen *gen,SByte *spikeEvent,SNum now)
{
	if(gen[index].pos<gen[index].length && gen[index].spikes[gen[index].pos]==now)
	{
		//printf("gen %u has a spike written to %p at %d step\n",index,&spikeEvent[now],now);
		if(spikeEvent)
			spikeEvent[now]=1;
		gen[index].pos++;
		return true;
	}
	return false;
}

namespace MultiGPUBrain
{



void Simulator::UseMyGPU()
{
#ifndef FAKE_MULTI_GPU
	if(mGPUID>=0)
	{
		SNum gpuCount;
		CUDACHECK(cudaGetDeviceCount(&gpuCount));
		//CUDACHECK(cudaSetDevice(mGPUID));
		CUDACHECK(cudaSetDevice(gpuCount-1-mGPUID));
		/*int gpuid;
		cudaGetDevice(&gpuid);
		printf("Simulator[%p] uses GPU%d\n",this,gpuid);*/
	}
#endif
}

void Simulator::UpdateNodeOffset()
{
	if(!mNodeChanged)
		return;
    mNodeOffsets[0]=0;
    for(SNum i=0;i<TYPE_COUNT;i++)
		mNodeOffsets[i+1]=mNodeOffsets[i]+mNodeCounts[i];
	mNodeChanged=false;
}


void Simulator::CleanLIF()
{
    if(mgLIF)
        cudaFree(mgLIF);
    mgLIF=NULL;
    if(mgLifArg)
        cudaFree(mgLifArg);
    mgLifArg=NULL;
    mNodeCounts[TYPE_LIF]=0;
}

void Simulator::CleanSynapse()
{
    if(mgSynapses)
        delete mgSynapses;
    mgSynapses=NULL;
    if(mgSections)
        cudaFree(mgSections);
    mgSections=NULL;
}

void Simulator::CleanOutputSynapse()
{
	for(SNum i=0;i<(SNum)mOutputSyn.size();i++)
	{
		if(mOutputSynData.size()>i)
		delete mOutputSynData[i];
		//if(mOutputSyn[i].gSynapses)
			//cudaFree(mOutputSyn[i].gSynapses);
		if(mOutputSyn[i].gSections)
			cudaFree(mOutputSyn[i].gSections);
		if(mOutputSyn[i].gCurrentBuffer)
			cudaFree(mOutputSyn[i].gCurrentBuffer);
		if(mOutputSyn[i].gTargetCurrent)
		{
			Simulator *pTarget=(Simulator *)mOutputSyn[i].pTarget;
			pTarget->UseMyGPU();
			cudaFree(mOutputSyn[i].gTargetCurrent);
			UseMyGPU();
		}
		CUDACHECK(cudaStreamDestroy(mOutputSyn[i].stream));
	}
	for(SNum i=0;i<(SNum)mOutsToIn.size();i++)
	{
		delete []mOutsToIn[i].maps;
	}
	if(mgOutsToIn)
	{
		POP_MAPS pms[mOutsToIn.size()];
		CUDACHECK(cudaMemcpy(pms,mgOutsToIn,sizeof(POP_MAPS)*mOutsToIn.size(),cudaMemcpyDeviceToHost));
		for(SNum i=0;i<(SNum)mOutsToIn.size();i++)
		{
			cudaFree(pms[i].maps);
		}
		cudaFree(mgOutsToIn);
		mgOutsToIn=NULL;
	}
	mMaxGrid=0;
	mOutsToIn.clear();
	mOutputSyn.clear();
	mOutputSynData.clear();
	mInToOuts.clear();
}

void Simulator::CleanSimulData()
{
	CURRENT_ELEMENT currentEle;
	for(SNum i=0;i<(SNum)mCurrentToMerge.size();i++)
	{
		currentEle=mCurrentToMerge[i];
		delete []currentEle.currentBuffer;
	}
	mCurrentToMerge.clear();
	for(std::map<SNum,std::pair<SLNum,SYNAPSE *>>::iterator it=mTempSyn.begin();it!=mTempSyn.end();it++)
	{
		delete []it->second.second;
	}
	mTempSyn.clear();
    if(mgNetwork)
        cudaFree(mgNetwork);
	mgNetwork=NULL;

	if(mgOutputSyn)
		cudaFree(mgOutputSyn);
	mgOutputSyn=NULL;

    if(mgRecorder)
        cudaFree(mgRecorder);
	mgRecorder=NULL;
	
    if(mgCurrent)
        cudaFree(mgCurrent);
	mgCurrent=NULL;
	
    if(mgActiveCount)
        cudaFree(mgActiveCount);
	mgActiveCount=NULL;

	if(mgActiveCountByGrid)
		cudaFree(mgActiveCountByGrid);
	mgActiveCountByGrid=NULL;
	
    if(mgActiveIndex)
        cudaFree(mgActiveIndex);
    mgActiveIndex=NULL;
}

bool Simulator::SetSpikeGenerator(GNGen *gens,SNum len)
{
	mNodeCounts[TYPE_GEN]=len;
	mPopCount[TYPE_GEN]=len;
	SetToGPU((void **)&mgGens,gens,sizeof(GNGen)*len);
	return true;
}

bool Simulator::SetSpikeGenerator(SNum index,const GNGen &gens)
{
	if(!mgGens)
		return false;
	if(index<0 || index>=mNodeCounts[TYPE_GEN])
		return false;

	CUDACHECK(cudaMemcpy(mgGens+index,&gens,sizeof(GNGen),cudaMemcpyHostToDevice));
	
	return true;
}

bool Simulator::SetLIFNodes(GNLIF *lifs,SNum nLIF,LIF_ARG *lifArg,SNum nLifArg)
{
    SNum pos,index;
    //先检查节点参数是否正确
    for(SNum i=0;i<nLIF;i++)
    {
        pos=i/MAXLIF;
        index=i%MAXLIF;
        if(lifs[pos].argPos[index]>=nLifArg || lifs[pos].argPos[index]<0)
            return false;
        if(lifs[pos].argIndex[index]<0 || lifs[pos].argIndex[index]>=MAXLIF)
            return false;
    }
	CleanLIF();
	mNodeChanged=true;
	mNodeCounts[TYPE_LIF]=nLIF;
	mPopCount[TYPE_LIF]=nLifArg;
	SetToGPU((void **)&mgLIF,lifs,sizeof(GNLIF)*((nLIF+MAXLIF-1)/MAXLIF));
    SetToGPU((void **)&mgLifArg,lifArg,sizeof(LIF_ARG)*nLifArg);
    return true;
}

int StatisticSynapses(SNum index,SNum nNodes,SNum *gSections,SYNAPSE *gSynapses)
{
	int ret=0;
	int nSyn;
	SYNAPSE *syns;
	CUDACHECK(cudaMemcpy(&nSyn,gSections+nNodes,sizeof(SNum),cudaMemcpyDeviceToHost));
	syns=new SYNAPSE[nSyn];
	CUDACHECK(cudaMemcpy(syns,gSynapses,sizeof(SYNAPSE)*nSyn,cudaMemcpyDeviceToHost));
	for(int i=0;i<nSyn;i++)
	{
		if(syns[i].postIndex==index)
			ret++;
	}
	delete []syns;
	return ret;
}

SLNum GetSynCount(SNum nBuild,SYN_BUILD builds[])
{
	SLNum ret=0;
	for(SNum i=0;i<nBuild;i++)
	{
		if(builds[i].bOneToOne)
		{
			if(builds[i].preCount<builds[i].postCount)
				ret+=builds[i].preCount;
			else
				ret+=builds[i].postCount;
		}
		else
		{
			ret+=(SLNum)ceil((double)builds[i].preCount*(double)builds[i].postCount*(double)builds[i].fPropa);
		}
	}
	return ret;
}

bool Simulator::SetSynapse(SNum nBuild,SYN_BUILD builds[])
{
    UpdateNodeOffset();
	if(mMaxDelay<0 || mMinDelay<0)
    	mMaxDelay=mMinDelay=builds[0].delay;
    for(SNum i=0;i<nBuild;i++)
    {
        if(mMaxDelay<builds[i].delay)
            mMaxDelay=builds[i].delay;
        if(mMinDelay>builds[i].delay)
            mMinDelay=builds[i].delay;
		//if(mGPUID==0)
		//printf("map inner synapses:%d->%d,%d\n",builds[i].preOffset,builds[i].postOffset,builds[i].postCount);
	}
	if(mInnerBuild)
		delete []mInnerBuild;
	mInnerBuildCount=nBuild;
	mInnerBuild=new SYN_BUILD[nBuild];
	memcpy(mInnerBuild,builds,sizeof(SYN_BUILD)*nBuild);
    return true;
}

bool Simulator::BuildInnerSynapse()
{
    SNum nodeCount;
    nodeCount=mNodeOffsets[TYPE_COUNT];
	//printf("Simulator %d SetSynapse\n",mGPUID);
	if(mInnerGridSize)
		printf("Simulator %d has %llu inner synapse grid size\n",mGPUID,mInnerGridSize);
    if(!BuildSynapse(mInnerBuildCount,mInnerBuild,&nodeCount,&mgSections,&mgSynapses,NULL,mInnerGridSize,mMeshSize))
    {
        return false;
	}
	CUDACHECK(cudaGetLastError());
    if(nodeCount>mNodeOffsets[TYPE_COUNT])
    {
        CleanSynapse();
        return false;
    }
	if(mgSynapses && mgSynapses->GetGridSize()>mMaxGrid)
		mMaxGrid=mgSynapses->GetGridSize();

	mInnerBuildCount=0;
	if(mInnerBuild)
	{
		delete []mInnerBuild;
		mInnerBuild=NULL;
	}
    return true;

}

bool Simulator::SetOutputSynapse(SNum nBuild,SYN_BUILD builds[],Simulator *pTarget)
{
	POP_MAPS gpm={0};
	POP_TO_ARRAY *ptas;
	OUTPUT_SYNAPSES os={0};
	std::pair<SNum,SYN_BUILD *> buildItem;
	PopMap pm;
	SNum t;
	if(nBuild<=0)
		return true;
	UpdateNodeOffset();
	if(mMaxDelay<0 || mMinDelay<0)
    	mMaxDelay=mMinDelay=builds[0].delay;
    for(SNum i=0;i<nBuild;i++)
    {
        if(mMaxDelay<builds[i].delay)
            mMaxDelay=builds[i].delay;
        if(mMinDelay>builds[i].delay)
			mMinDelay=builds[i].delay;
			
		pm.AddPop(builds[i].postOffset,builds[i].postCount);
		//if(mGPUID==1)
			//printf("map output synase:%d,%d->%d,%d\n",builds[i].preOffset,builds[i].preCount,builds[i].postOffset,builds[i].postCount);
	}
	//由于发放外部脉冲所用的电流缓冲区只保存目标分部所涉及到的神经元输入电流，因此需要将突触下标映射
	for(SNum i=0;i<nBuild;i++)
	{
		for(SNum j=0;j<pm.GetCount();j++)
		{
			if(builds[i].postOffset==pm[j].srcOffset && builds[i].postCount==pm[j].ncount)
			{
				builds[i].postOffset=pm[j].dstOffset;
			}
		}
	}
	//for(SNum j=0;j<pm.GetCount() && mGPUID==1;j++)
		//printf("building map[%d] %d->%d size:%d\n",j,pm[j].srcOffset,pm[j].dstOffset,pm[j].ncount);
	//for(SNum i=0;i<nBuild && mGPUID==1;i++)
		//printf("build from (%d,%d) to (%d,%d)\n",builds[i].preOffset,builds[i].preCount,builds[i].postOffset,builds[i].postCount);
	os.nDstNode=pm.GetArrayLength();
	buildItem.first=nBuild;
	buildItem.second=new SYN_BUILD[nBuild];
	memcpy(buildItem.second,builds,sizeof(SYN_BUILD)*nBuild);
	//在对方建立相应的GPU内数据结构，以便对方在核函数TransferSpikes中接收
	ptas=new POP_TO_ARRAY[pm.GetCount()];
	for(SNum i=0;i<pm.GetCount();i++)
	{
		ptas[i]=pm[i];
		t=ptas[i].srcOffset;
		ptas[i].srcOffset=ptas[i].dstOffset;
		ptas[i].dstOffset=t;
		gpm.nodeCount+=ptas[i].ncount;
		if(ptas[i].ncount>gpm.nMaxNodeCount)
			gpm.nMaxNodeCount=ptas[i].ncount;
	}
	gpm.ncount=pm.GetCount();
	gpm.maps=ptas;
	os.nDstMap=(SNum)pTarget->mOutsToIn.size();
	pTarget->mOutsToIn.push_back(gpm);

	os.pTarget=pTarget;
	//检测当前GPU到目标GPU是否可以直接点对点通信
#ifndef FAKE_MULTI_GPU
	CUDACHECK(cudaDeviceCanAccessPeer(&os.nAccessEnable,mGPUID,pTarget->mGPUID));
	if(os.nAccessEnable)
	{
		//printf("communication from GPU%d to GPU%d enable!\n",mGPUID,pTarget->mGPUID);
		CUDACHECK(cudaDeviceEnablePeerAccess(pTarget->mGPUID,0));
		CUDACHECK(cudaStreamCreate(&os.stream));
		//pTarget->UseMyGPU();
		//UseMyGPU();
	}
#endif
	mOutterBuilds.push_back(buildItem);
	mOutputSyn.push_back(os);
	mInToOuts.push_back(pm);
    return true;
}

bool Simulator::BuildOutterSynapse()
{
	SNum nodeCount;
	SNum nBuild;
	SYN_BUILD *builds;
	OUTPUT_SYNAPSES os={0};
	OUTPUT_SYNAPSES *pOS;
	MemSchedule<SYNAPSE> *pMS;
	for(SNum i=0;i<(SNum)mOutterBuilds.size();i++)
	{
		nBuild=mOutterBuilds[i].first;
		builds=mOutterBuilds[i].second;
		os=mOutputSyn[i];
		if(os.nGridSize)
			printf("Outter synapses from %d to %d has grid size:%llu\n",\
			mGPUID,((Simulator *)os.pTarget)->mGPUID,os.nGridSize);
		pOS=new OUTPUT_SYNAPSES;
		if(!BuildSynapse(nBuild,builds,&nodeCount,&pOS->gSections,&pMS,&os.nMaxSynCount,os.nGridSize))
		{
			printf("Build synapse failed!\n");
			delete pOS;
			CleanOutterBuilds();
			return false;
		}
		os.gSections=pOS->gSections;
		os.gSynapses=pMS->GetGPUBuffer();//pOS->gSynapses;
		os.nGridSize=pMS->GetGPULen();
		mOutputSyn[i]=os;
		delete pOS;
	
		if(pMS->GetGridSize()>mMaxGrid)
			mMaxGrid=pMS->GetGridSize();
		mOutputSynData.push_back(pMS);
	}
	CleanOutterBuilds();
	return true;
}

bool Simulator::PrepareSynapseSize()
{
	size_t a,t;
	SNum n;
	OUTPUT_SYNAPSES os;
	SFNum rate;
	SLNum wholeSynSize,s;
	SLNum innerSynSize;
	std::vector<SLNum> outterSynSize;
	SLNum avail;
	SLNum runtimeSize=0;
	SLNum nodeSize=0;
	mInnerGridSize=0;
	//计算节点所需要的显存空间大小
	nodeSize+=sizeof(GNGen)*mNodeCounts[TYPE_GEN];
	nodeSize+=sizeof(LIF_ARG)*mPopCount[TYPE_LIF];
	nodeSize+=sizeof(GNLIF)*((mNodeCounts[TYPE_LIF]+MAXLIF-1)/MAXLIF);
	nodeSize+=sizeof(SLNum)*(mNodeOffsets[TYPE_COUNT]+1);
	//计算仿真过程中需要的运行时显存空间大小
	runtimeSize+=sizeof(NETWORK_DATA);
	for(SNum i=0;i<(SNum)mOutputSyn.size();i++)
	{
		os=mOutputSyn[i];
		mOutputSyn[i].nGridSize=0;
		runtimeSize+=sizeof(SFNum)*os.nDstNode*(mGroupLength*mCurrentGroupCount+mMaxTSDelay);
	}
	runtimeSize+=sizeof(NSPIKE_GEN)*mNodeOffsets[TYPE_COUNT];
	runtimeSize+=sizeof(SFNum)*(mCurrentGroupCount*mGroupLength+mMaxTSDelay)*mNodeOffsets[TYPE_COUNT];
	runtimeSize+=sizeof(SNum)*mCurrentGroupCount;
	runtimeSize+=sizeof(SNum)*mGroupLength*mNodeOffsets[TYPE_COUNT];

	n=(SNum)mOutsToIn.size();
	runtimeSize+=sizeof(POP_MAPS)*n;
	for(SNum i=0;i<n;i++)
		runtimeSize+=sizeof(POP_TO_ARRAY)*mOutsToIn[i].ncount;

	CUDACHECK(cudaMemGetInfo(&a,&t));
	avail=(SLNum)a;
	if(avail<=(nodeSize+runtimeSize-sizeof(SYNAPSE)*(1+mOutterBuilds.size())))
		return false;
	avail-=(nodeSize+runtimeSize);
	avail=(SLNum)((SFNum)avail*0.8f);
	innerSynSize=sizeof(SYNAPSE)*GetSynCount(mInnerBuildCount,mInnerBuild);
	wholeSynSize=innerSynSize;
	for(SNum i=0;i<(SNum)mOutterBuilds.size();i++)
	{
		s=sizeof(SYNAPSE)*GetSynCount(mOutterBuilds[i].first,mOutterBuilds[i].second);
		outterSynSize.push_back(s);
		wholeSynSize+=s;
	}
	printf("Part %d needs %llu bytes to save nodes, %llu bytes to run, %llu bytes to save synapses\n",\
	mGPUID,nodeSize,runtimeSize,wholeSynSize);
	if(wholeSynSize<=avail)
		return true;
	printf("Part %d needs synapse swope,available memory size:%llu\n",mGPUID,avail);
	rate=(SFNum)avail/(SFNum)wholeSynSize;
	innerSynSize=(SLNum)((SFNum)innerSynSize*rate);
	mInnerGridSize=innerSynSize/sizeof(SYNAPSE);
	if(mInnerGridSize<=0)
		mInnerGridSize=1;

	for(SNum i=0;i<(SNum)mOutputSyn.size();i++)
	{
		s=outterSynSize[i];
		s=(SLNum)((SFNum)s*rate);
		mOutputSyn[i].nGridSize=s/sizeof(SYNAPSE);
		if(mOutputSyn[i].nGridSize<=0)
			mOutputSyn[i].nGridSize=1;
	}
	
	return true;
}

bool Simulator::Prepare(SFNum timestep,SNum nTSGroupLen,SNum nTSMaxDelay,SNum nBlockSize)
{
    NETWORK_DATA network={{0}};
	OUTPUT_SYNAPSES os;
    SLNum section[mNodeOffsets[TYPE_COUNT]+1];
	SNum d,n;
	cudaDeviceProp deviceProp;
	mExtraSyn=new SynManager(mNodeOffsets[TYPE_COUNT],mMeshSize);
	CUDACHECK(cudaGetDeviceProperties(&deviceProp,0));
	mBlockSize=nBlockSize;
	mMaxBlock=deviceProp.maxThreadsPerMultiProcessor*deviceProp.multiProcessorCount/mBlockSize;
	mTimestep=timestep;
	if(nTSGroupLen>0)
		mGroupLength=nTSGroupLen;
	else
    	mGroupLength=(SNum)(mMinDelay/timestep);//以最小延迟作为时间片组的长度
	mNowTSGroup=0;
	if(mMaxDelay<=0)
		mCurrentGroupCount=nTSGroupLen;
	else
		mCurrentGroupCount=(SNum)(mMaxDelay/timestep);
	if(nTSMaxDelay>0)
		mMaxTSDelay=nTSMaxDelay;
	else
		mMaxTSDelay=(SNum)(mMaxDelay/timestep);
	UpdateNodeOffset();
	if(!PrepareSynapseSize())
		return false;
	//正式构建网络
	if(!BuildInnerSynapse())
		return false;
	if(!BuildOutterSynapse())
		return false;
    memcpy(network.offset,mNodeOffsets,(TYPE_COUNT+1)*sizeof(SNum));
    if(mgSections)
    {
        CUDACHECK(cudaMemcpy(section,mgSections,sizeof(SLNum)*(mNodeOffsets[TYPE_COUNT]+1),cudaMemcpyDeviceToHost));
        for(SNum i=0;i<mNodeOffsets[TYPE_COUNT];i++)
        {
            d=section[i+1]-section[i];
            if(d>network.maxSynapseCount)
                network.maxSynapseCount=d;
		}
		//printf("This simulator has %llu synapses\n",section[mNodeOffsets[TYPE_COUNT]]);
	}
	network.maxWholeSynCount=network.maxSynapseCount;
	if(mgSynapses)
	{
		network.synapses=mgSynapses->GetGPUBuffer();
		network.nGridSize=mgSynapses->GetGPULen();
	}
	network.section=mgSections;
	network.gen=mgGens;
    network.LIF=mgLIF;
	network.LIFArgs=mgLifArg;
	network.nOutput=(SNum)mOutputSyn.size();
	network.meshSize=mMeshSize;
	CUDACHECK(cudaMalloc(&mgOutputSyn,sizeof(OUTPUT_SYNAPSES)*network.nOutput));
	for(SNum i=0;i<network.nOutput;i++)
	{
		os=mOutputSyn[i];
		if(os.nMaxSynCount>network.maxSynapseCount)
			network.maxWholeSynCount=os.nMaxSynCount;

		//printf("output %d's buffer length:%d\n",i,os.nDstNode*(mGroupLength*mCurrentGroupCount+mMaxTSDelay));
		CUDACHECK(cudaMalloc((LPVOID *)&os.gCurrentBuffer,sizeof(SFNum)*os.nDstNode*(mGroupLength*mCurrentGroupCount+mMaxTSDelay)));
		CUDACHECK(cudaMemcpy(&mgOutputSyn[i],&os,sizeof(OUTPUT_SYNAPSES),cudaMemcpyHostToDevice));
		mOutputSyn[i]=os;
	}
	network.outputs=mgOutputSyn;

    CUDACHECK(cudaMalloc(&mgNetwork,sizeof(NETWORK_DATA)));
    CUDACHECK(cudaMemcpy(mgNetwork,&network,sizeof(NETWORK_DATA),cudaMemcpyHostToDevice));

    CUDACHECK(cudaMalloc(&mgRecorder,sizeof(NSPIKE_GEN)*mNodeOffsets[TYPE_COUNT]));
    CUDACHECK(cudaMemset(mgRecorder,0,sizeof(NSPIKE_GEN)*mNodeOffsets[TYPE_COUNT]));
    CUDACHECK(cudaMalloc(&mgCurrent,sizeof(SFNum)*(mCurrentGroupCount*mGroupLength+mMaxTSDelay)*mNodeOffsets[TYPE_COUNT]));
    CUDACHECK(cudaMemset(mgCurrent,0,sizeof(SFNum)*(mCurrentGroupCount*mGroupLength+mMaxTSDelay)*mNodeOffsets[TYPE_COUNT]));
    CUDACHECK(cudaMalloc(&mgActiveCount,sizeof(SNum)*mCurrentGroupCount));
	CUDACHECK(cudaMemset(mgActiveCount,0,sizeof(SNum)*mCurrentGroupCount));
    CUDACHECK(cudaMalloc(&mgActiveCountByGrid,sizeof(SNum)*mMaxGrid));
	CUDACHECK(cudaMemset(mgActiveCountByGrid,0,sizeof(SNum)*mMaxGrid));
    CUDACHECK(cudaMalloc(&mgActiveIndex,sizeof(SNum)*mGroupLength*mNodeOffsets[TYPE_COUNT]));
	CUDACHECK(cudaMemset(mgActiveIndex,0,sizeof(SNum)*mGroupLength*mNodeOffsets[TYPE_COUNT]));
	
	//将向内映射数据传入GPU内
	n=(SNum)mOutsToIn.size();
	POP_MAPS pms[n];
	POP_TO_ARRAY *ptas;
	CUDACHECK(cudaMalloc((LPVOID *)&mgOutsToIn,sizeof(POP_MAPS)*n));
	for(SNum i=0;i<n;i++)
	{
		pms[i]=mOutsToIn[i];
		CUDACHECK(cudaMalloc((LPVOID *)&ptas,sizeof(POP_TO_ARRAY)*pms[i].ncount));
		CUDACHECK(cudaMemcpy(ptas,pms[i].maps,sizeof(POP_TO_ARRAY)*pms[i].ncount,cudaMemcpyHostToDevice));
		pms[i].maps=ptas;
	}
	CUDACHECK(cudaMemcpy(mgOutsToIn,pms,sizeof(POP_MAPS)*n,cudaMemcpyHostToDevice));

	n=0;
    return true;
}

__global__ void simulateAndPush(NETWORK_DATA *network,NSPIKE_GEN *neuronSpikes,SFNum *currentBuf,\
	SNum *activeGridNum,SNum *activeNum,SNum *activePreIndex,SNum nGroupLength,SNum now,SNum currentPos,\
	SFNum timestep,bool bSelectActiveNeuron,SNum GPUID,SLNum synGridPos)
{
	//unsigned int threadID=blockIdx.x*blockDim.x+threadIdx.x;
	SLNum synStart,synEnd,osynStart,osynEnd;
	SNum thID=blockIdx.x*blockDim.x+threadIdx.x;
	SNum index=thID;
	SNum threadNum=blockDim.x*gridDim.x;
	SNum i;
	SNum pos;
	SNum neuronCount=network->offset[TYPE_COUNT];
	SLNum j,k;
	SLNum startS,endS;
	SNum bufOffset=index;
	SNum delay,postTime,postPos;
	SFNum nowCurrent;
	SNum mesh;
	bool bFired;
	synStart=synGridPos*network->nGridSize;
	synEnd=synStart+network->nGridSize;
	while(index<neuronCount)
	{
		bufOffset=index;
		startS=network->section[index];
		endS=network->section[index+1];
		if(startS<synStart)
			startS=synStart;
		if(endS>synEnd)
			endS=synEnd;
		for(i=0;i<nGroupLength ;i++,bufOffset+=neuronCount)
		{
			bFired=false;
			nowCurrent=currentBuf[bufOffset];
			/*if(GPUID==0 && index==0)
			{
				if(nowCurrent!=0.0)
				printf("got input current%p[%d]:%d-%f\n",&currentBuf[bufOffset],bufOffset,now,nowCurrent);
			}*/
			if(index>=network->offset[TYPE_LIF])
			{
				if(simulateLIF(index-network->offset[TYPE_LIF],nowCurrent,network->LIF,network->LIFArgs,i,timestep,NULL))
				{
					//if(endS>startS || network->nOutput>0)
					//if(index==network->offset[TYPE_LIF])
					//printf("fired at %d-%d with %d output synapses\n",now,i,network->nOutput);
					network->lastFired=i;
					bFired=true;
				}
			}
			else if(index>=network->offset[TYPE_GEN])
			{
				if(simulateSpikeGen(index,network->gen,NULL,now*nGroupLength+i))
				{
					//printf("input spike at %d-%d with %d output synapses\n",now,i,network->nOutput);
					network->lastFired=i;
					bFired=true;
				}
			}
			// push spike to post-synaptic neuron
			if(bFired)
			{
				if(neuronSpikes && neuronSpikes[index].length<MAX_SPIKE_COUNT)
					neuronSpikes[index].spikes[neuronSpikes[index].length++]=(SFNum)(now*nGroupLength+i)*timestep;
				atomicAdd(&activeGridNum[startS/network->nGridSize],1);
				atomicAdd(&activeGridNum[(endS-1)/network->nGridSize],1);
				for(j=0;j<network->nOutput;j++)
				{
					atomicAdd(&activeGridNum[network->outputs[j].gSections[index]/network->outputs[j].nGridSize],1);
					atomicAdd(&activeGridNum[(network->outputs[j].gSections[index+1]-1)/network->outputs[j].nGridSize],1);
				}
				if(bSelectActiveNeuron)
				{
					pos=atomicAdd(activeNum,1);
					activePreIndex[pos]=index+i*neuronCount;
				}
				else
				{
					//向内推送脉冲
					for(j=startS-synStart;j<(endS-synStart);j++)
					{
						if(network->synapses[j].postIndex<0)
							continue;
						delay=(SNum)(network->synapses[j].delay/timestep);
						postTime=i+delay;
						postPos=postTime*neuronCount+network->synapses[j].postIndex;
						atomicAdd(&currentBuf[postPos],network->synapses[j].weight);
						/*if(network->synapses[j].postIndex==0 && GPUID==0 && network->synapses[j].weight!=0.0)
						{
							printf("N%d add %f to [%d]=%f\n",index,network->synapses[j].weight,\
							postPos,currentBuf[postPos]);
						}*/
					}
					//对新增的突触进行脉冲发放
					if(network->node2Syn)
					{
						mesh=network->node2Syn[index];
						while(mesh>=0)
						{
							for(j=network->meshSize*mesh;j<(network->meshSize*mesh+network->meshSize);j++)
							{
								if(network->extraSynapses[j].postIndex<0)
									continue;
								delay=(SNum)(network->extraSynapses[j].delay/timestep);
								postTime=i+delay;
								postPos=postTime*neuronCount+network->extraSynapses[j].postIndex;
								atomicAdd(&currentBuf[postPos],network->extraSynapses[j].weight);
							}
							mesh=network->linkTable[mesh];
						}
					}
					//向外推送脉冲
					for(j=0;j<network->nOutput;j++)
					{
						osynStart=network->outputs[j].nGridSize*synGridPos;
						osynEnd=osynStart+network->outputs[j].nGridSize;
						for(k=network->outputs[j].gSections[index];k<network->outputs[j].gSections[index+1];k++)
						{
							if(k<osynStart || k>=osynEnd || network->outputs[j].gSynapses[k-osynStart].postIndex<0)
								continue;
							delay=(SNum)(network->outputs[j].gSynapses[k-osynStart].delay/timestep);
							postTime=i+delay;
							postPos=postTime*network->outputs[j].nDstNode+network->outputs[j].gSynapses[k-osynStart].postIndex;
							//if(thID==0 && GPUID==0)
							//printf("postTime=%d,postPos=%d,currentPos=%d,nDstNode=%d,grouplength=%d\n",\
							postTime,postPos,currentPos,network->outputs[j].nDstNode,nGroupLength);
							//if(GPUID==1)
							//printf("%d push outter spikes through %d to %d\n",index,network->outputs[j].gSynapses[k-osynStart].postIndex,postPos+currentPos*network->outputs[j].nDstNode*nGroupLength);
							//if(network->outputs[j].gSynapses[k-osynStart].postIndex==0 && GPUID==1 && network->outputs[j].gSynapses[k-osynStart].weight!=0.0)
							//printf("neuron %d push outer spike %f to %d\n",index,network->outputs[j].gSynapses[k-osynStart].weight,postPos+currentPos*network->outputs[j].nDstNode*nGroupLength);
							atomicAdd(&network->outputs[j].gCurrentBuffer[postPos+currentPos*network->outputs[j].nDstNode*nGroupLength],network->outputs[j].gSynapses[k-osynStart].weight);
						}
					}
				}
			}
			
		}
		index+=threadNum;
	}
}

__global__ void push_spike(NETWORK_DATA *network,SFNum *currentBuf,SNum *activeNum,SNum *activePreIndex,SNum nGroupLength,SNum currentPos,SFNum timestep,SLNum synGridPos,SNum GPUID)
{
	unsigned long long pos=blockIdx.x*blockDim.x+threadIdx.x;
	SNum index;
	SLNum startS,endS;
	SNum delay,postTime,postPos;
	SNum i,k,mesh;
	SNum neuron;
	SLNum syn,j;
	SLNum synStart,synEnd;
	//SNum bufOffset=0;
	SNum neuronCount=network->offset[TYPE_COUNT];
	unsigned long long maxSyn=network->maxWholeSynCount>network->maxExtraCount?network->maxWholeSynCount:network->maxExtraCount;
	unsigned long long activeSynCount=(unsigned long long)(*activeNum)*maxSyn;
	while(pos<activeSynCount)
	{
		index=pos/maxSyn;
		if(index>=(*activeNum))
		{
			break;
		}
		neuron=activePreIndex[index]%neuronCount;
		syn=pos%maxSyn;
		if(network->maxSynapseCount>0)//发放内部预定突触的脉冲
		{
			synStart=network->nGridSize*synGridPos;
			synEnd=synStart+network->nGridSize;
			startS=network->section[neuron];
			endS=network->section[neuron+1];
			if(endS>synStart && startS<=synEnd)//当前的突触区间与当前的显存块必须有交集，才发放脉冲
			{
				if(startS<synStart)
					startS=synStart;
				if(endS>synEnd)
					endS=synEnd;
				if(startS<endS && syn<(endS-startS))
				{
					i=activePreIndex[index]/neuronCount;
					j=startS+syn-synStart;
					if(network->synapses[j].postIndex>=0)
					{
						delay=(SNum)(network->synapses[j].delay/timestep);
						postTime=i+delay;
						postPos=postTime*neuronCount+network->synapses[j].postIndex;
						atomicAdd(&currentBuf[postPos],network->synapses[j].weight);
					}
				}
			}
		}
		if(network->node2Syn)//发放内部额外突触的脉冲
		{
			//index=pos/network->maxExtraCount;
			//neuron=activePreIndex[index]%neuronCount;
			//syn=pos%network->maxExtraCount;
			mesh=network->node2Syn[neuron];
			for(i=0;mesh>=0 && i<syn/network->meshSize;i++)
			{
				mesh=network->linkTable[mesh];
			}
			if(mesh>=0)
			{
				i=activePreIndex[index]/neuronCount;
				j=mesh*network->meshSize+(syn%network->meshSize);
				if(network->extraSynapses[j].postIndex>=0)
				{
					delay=(SNum)(network->extraSynapses[j].delay/timestep);
					postTime=i+delay;
					postPos=postTime*neuronCount+network->extraSynapses[j].postIndex;
					atomicAdd(&currentBuf[postPos],network->extraSynapses[j].weight);
				}
			}
		}
		for(i=0;i<network->nOutput;i++)//发放外部突触的脉冲
		{
			if(network->outputs[i].nMaxSynCount<=0)
				continue;
			//index=pos/network->outputs[i].nMaxSynCount;
			//neuron=activePreIndex[index]%neuronCount;
			//syn=pos%network->outputs[i].nMaxSynCount;
			synStart=network->outputs[i].nGridSize*synGridPos;
			synEnd=synStart+network->outputs[i].nGridSize;
			startS=network->outputs[i].gSections[neuron];
			endS=network->outputs[i].gSections[neuron+1];
			if(endS<=synStart || startS>synEnd)//如果当前的突触区间与当前的显存块完全没有交集，则跳过
				continue;
			if(startS<synStart)
				startS=synStart;
			if(endS>synEnd)
				endS=synEnd;
			if(startS>=endS || syn>=(endS-startS))
				continue;
			k=activePreIndex[index]/neuronCount;
			j=startS+syn-synStart;
			if(network->outputs[i].gSynapses[j].postIndex<0)
				continue;
			delay=(SNum)(network->outputs[i].gSynapses[j].delay/timestep);
			postTime=k+delay;
			postPos=postTime*network->outputs[i].nDstNode+network->outputs[i].gSynapses[j].postIndex;
			k=postPos+currentPos*network->outputs[i].nDstNode*nGroupLength;
			atomicAdd(&network->outputs[i].gCurrentBuffer[k],network->outputs[i].gSynapses[j].weight);
		}
		pos+=blockDim.x * gridDim.x;
		//bufOffset+=neuronCount;
    }

}
    
bool Simulator::Simulate(bool bSelectActiveNeuron)
{
	OUTPUT_SYNAPSES os;
	SNum currentPos;
	SNum blockSize;
	SNum lastGrid;
	SNum activeGridNum[mMaxGrid];
	std::vector<SLNum> gridToHandle;//将要调度来做脉冲发放的格子
	CUDACHECK(cudaGetLastError());
	if(mCurrentGroupCount>0)
		currentPos=mNowTSGroup%mCurrentGroupCount;
	if(currentPos==0 && mNowTSGroup>0 && mCurrentGroupCount>0)//重置仿真过程的数据
	{
		//CUDACHECK(cudaDeviceSynchronize());
		CUDACHECK(cudaMemset(mgActiveCount,0,sizeof(SNum)*mCurrentGroupCount));
		CUDACHECK(cudaMemcpy(mgCurrent,&mgCurrent[mNodeOffsets[TYPE_COUNT]*mCurrentGroupCount*mGroupLength],\
			sizeof(SFNum)*mNodeOffsets[TYPE_COUNT]*mMaxTSDelay,cudaMemcpyDeviceToDevice));
		CUDACHECK(cudaMemset(&mgCurrent[mNodeOffsets[TYPE_COUNT]*mMaxTSDelay],0,sizeof(SFNum)*mNodeOffsets[TYPE_COUNT]*mGroupLength*mCurrentGroupCount));
		for(SNum i=0;i<(SNum)mOutputSyn.size();i++)
		{
			os=mOutputSyn[i];
			CUDACHECK(cudaMemcpy(os.gCurrentBuffer,&os.gCurrentBuffer[os.nDstNode*mGroupLength*mCurrentGroupCount],\
				sizeof(SFNum)*os.nDstNode*mMaxTSDelay,cudaMemcpyDeviceToDevice));
			CUDACHECK(cudaMemset(&os.gCurrentBuffer[os.nDstNode*mMaxTSDelay],0,sizeof(SFNum)*os.nDstNode*mGroupLength*mCurrentGroupCount));
		}
	}
	if(mMaxGrid>1)//如果需要调度，则需要使用每个调度格子的记录情况，在仿真之前需要清空
		CUDACHECK(cudaMemset(mgActiveCountByGrid,0,sizeof(SNum)*mMaxGrid));
	blockSize=MYGRID_SIZE(mNodeOffsets[TYPE_COUNT],mBlockSize);
	if(blockSize>mMaxBlock)
		blockSize=mMaxBlock;
	
	if(mgSynapses)
		mgSynapses->SwitchToGrid(0);
	simulateAndPush<<<blockSize,mBlockSize>>>(mgNetwork,mgRecorder,&mgCurrent[currentPos*mNodeOffsets[TYPE_COUNT]*mGroupLength],\
		mgActiveCountByGrid,&mgActiveCount[currentPos],mgActiveIndex,mGroupLength,mNowTSGroup,currentPos,mTimestep,bSelectActiveNeuron,mGPUID,0);
	cudaError_t err=cudaGetLastError();
	CUDACHECK(err);
	if(mMaxGrid>1)//如果需要调度，则需要判断当前是否存在要发放脉冲的神经元
	{
		err=cudaDeviceSynchronize();
		CUDACHECK(err);
		CUDACHECK(cudaMemcpy(activeGridNum,mgActiveCountByGrid,sizeof(SNum)*mMaxGrid,cudaMemcpyDeviceToHost));
		//CUDACHECK(cudaMemcpy(&activeNum,&mgActiveCount[currentPos],sizeof(SNum),cudaMemcpyDeviceToHost));
	}
	else
	{
		activeGridNum[0]=1;
	}
	//先确定有哪些格子需要调度
	lastGrid=mgSynapses->GetGridPos();
	gridToHandle.push_back(lastGrid);//先完成之前的调度格子
	for(SNum i=0;i<mMaxGrid;i++)
	{
		if(activeGridNum[i]<=0 || i==lastGrid)
			continue;
	}
	for(SNum ii=0;ii<(SNum)gridToHandle.size();ii++)
	{
		SLNum i=gridToHandle[ii];
		if(mgSynapses)
			mgSynapses->SwitchToGrid(i);
		for(SNum j=0;j<mOutputSynData.size();j++)
		{
			if(mOutputSynData[j])
				mOutputSynData[j]->SwitchToGrid(i);
		}
		if(bSelectActiveNeuron)
		{
			push_spike<<<blockSize,mBlockSize>>>(mgNetwork,&mgCurrent[currentPos*mNodeOffsets[TYPE_COUNT]*mGroupLength],\
				&mgActiveCount[currentPos],mgActiveIndex,mGroupLength,currentPos,mTimestep,i,mGPUID);
		}
		else if(i>0)
		{
			simulateAndPush<<<blockSize,mBlockSize>>>(mgNetwork,mgRecorder,&mgCurrent[currentPos*mNodeOffsets[TYPE_COUNT]*mGroupLength],\
				mgActiveCountByGrid,&mgActiveCount[currentPos],mgActiveIndex,mGroupLength,mNowTSGroup,currentPos,mTimestep,bSelectActiveNeuron,mGPUID,\
				i);
		}
		err=cudaGetLastError();
		if(err!=cudaSuccess)
			printf("GPU%d had error!current grid:%llu,max grid:%llu\n",mGPUID,i,mMaxGrid);
		CUDACHECK(err);

	}
	err=cudaDeviceSynchronize();
	CUDACHECK(err);
	mNowTSGroup++;
    return true;
}

//按照映射转移脉冲电流
__global__ void TransferSpikesToTarget(POP_MAPS *maps,SFNum *srcCurrent,SFNum *dstCurrent,SNum nodeCount,SNum copyLen,SNum GPUID)
{
	unsigned long long pos=blockIdx.x*blockDim.x+threadIdx.x;
	SNum nodePerMap=maps->nMaxNodeCount;
	SNum nWhole=nodePerMap*maps->ncount;
	SNum iMap,iNode;
	SNum src,dst;
	while(pos<nWhole)
	{
		iMap=pos/nodePerMap;
		iNode=pos%nodePerMap;
		if(iNode<maps->maps[iMap].ncount)
		{
			src=maps->maps[iMap].srcOffset+iNode;
			dst=maps->maps[iMap].dstOffset+iNode;
			if(src<nodeCount && dst<nodeCount)
			{
				for(SNum i=0;i<copyLen;i++)
				{
					atomicAdd(&dstCurrent[i*nodeCount+dst],srcCurrent[i*maps->nodeCount+src]);
					/*if(GPUID==0 && dst==0 && srcCurrent[i*maps->nodeCount+src]!=0.0f)
					{
						printf("transfer spike,i=%d,pos=%llu,nWhole=%d,nodePerMap=%d,maps-count=%d\n",i,pos,nWhole,nodePerMap,maps->ncount);
						printf("map[%d]:%d->%d\n",iMap,maps->maps[iMap].srcOffset,maps->maps[iMap].dstOffset);
						printf("got outter spike[%f] from %p to %p,now current:%f\n",srcCurrent[i*maps->nodeCount+src],&srcCurrent[i*maps->nodeCount+src],&dstCurrent[i*nodeCount+dst],dstCurrent[i*nodeCount+dst]);
					}*/
				}
			}
		}
		pos+=blockDim.x * gridDim.x;
	}
}

bool Simulator::TransferSpikes()
{
	CURRENT_ELEMENT currentEle={0};
	//SNum blockSize;
	SNum currentPos;
	OUTPUT_SYNAPSES os;
	SNum i,copyLen;
	SFNum *pSrcBuf;
	Simulator *pTarget;
	//SFNum **allCurrents;
	//SFNum *gCurrent;
	if(mOutputSyn.size()<=0)
		return true;
	currentPos=mNowTSGroup%mCurrentGroupCount;
	copyLen=mMaxTSDelay;
	//allCurrents=new SFNum*[mOutputSyn.size()];
	//先将当前所有的脉冲电流数据取出来
	UseMyGPU();
	for(i=0;i<(SNum)mOutputSyn.size();i++)
	{
		os=mOutputSyn[i];
		currentEle.nDstNode=os.nDstNode;
		currentEle.nDstMap=os.nDstMap;
		pTarget=(Simulator *)os.pTarget;
		pSrcBuf=&os.gCurrentBuffer[currentPos*os.nDstNode*mGroupLength];
		if(os.nAccessEnable)
		{
			if(!os.gTargetCurrent)
			{
				pTarget->UseMyGPU();
				CUDACHECK(cudaMalloc((void **)&os.gTargetCurrent,copyLen*os.nDstNode*sizeof(SFNum)));
				mOutputSyn[i].gTargetCurrent=os.gTargetCurrent;
				UseMyGPU();
			}
			CUDACHECK(cudaMemcpyPeerAsync(os.gTargetCurrent,pTarget->mGPUID,pSrcBuf,mGPUID,\
				sizeof(SFNum)*os.nDstNode*copyLen,os.stream));
			CUDACHECK(cudaStreamSynchronize(os.stream));
			//CUDACHECK(cudaMemcpy(os.gTargetCurrent,&os.gCurrentBuffer[currentPos*os.nDstNode*mGroupLength],\
				sizeof(SFNum)*os.nDstNode*copyLen,cudaMemcpyDeviceToDevice));
			currentEle.gTargetCurrent=os.gTargetCurrent;
		}
		else
		{
			currentEle.currentBuffer=new SFNum[os.nDstNode*copyLen];
			CUDACHECK(cudaMemcpy(currentEle.currentBuffer,&os.gCurrentBuffer[currentPos*os.nDstNode*mGroupLength],sizeof(SFNum)*os.nDstNode*copyLen,cudaMemcpyDeviceToHost));
		}
		pTarget->Lock();
		pTarget->mCurrentToMerge.push_back(currentEle);
		pTarget->Unlock();
	}
	//再将脉冲电流数据按照映射关系发放到目标分部
	/*for(i=0;i<(SNum)mOutputSyn.size();i++)
	{
		os=mOutputSyn[i];
		pTarget=(Simulator *)os.pTarget;
		if(!pTarget)
			continue;
		pTarget->UseMyGPU();
		blockSize=MYGRID_SIZE(pTarget->mOutsToIn[os.nDstMap].nMaxNodeCount*pTarget->mOutsToIn[os.nDstMap].ncount,mBlockSize);
		if(blockSize>mMaxBlock)
			blockSize=mMaxBlock;
		CUDACHECK(cudaMalloc((LPVOID *)&gCurrent,sizeof(SFNum)*os.nDstNode*copyLen));
		CUDACHECK(cudaMemcpy(gCurrent,allCurrents[i],sizeof(SFNum)*os.nDstNode*copyLen,cudaMemcpyHostToDevice));
		TransferSpikesToTarget<<<blockSize,mBlockSize>>>(&pTarget->mgOutsToIn[os.nDstMap],gCurrent,\
			&pTarget->mgCurrent[currentPos*pTarget->mNodeOffsets[TYPE_COUNT]*pTarget->mGroupLength],pTarget->mNodeOffsets[TYPE_COUNT],copyLen);
		CUDACHECK(cudaGetLastError());
		CUDACHECK(cudaFree(gCurrent));
	}
	CUDACHECK(cudaDeviceSynchronize());*/
	/*for(i=0;i<(SNum)mOutputSyn.size();i++)
	{
		delete allCurrents[i];
	}
	delete []allCurrents;*/
	return true;
}

void Simulator::MergeSpikes()
{
	SNum i;
	SNum blockSize;
	SFNum *gCurrent;
	CURRENT_ELEMENT currentEle;
	SNum currentPos,copyLen;
	currentPos=mNowTSGroup%mCurrentGroupCount;
	copyLen=mMaxTSDelay;
	for(i=0;i<(SNum)mCurrentToMerge.size();i++)
	{
		currentEle=mCurrentToMerge[i];
		blockSize=MYGRID_SIZE(mOutsToIn[currentEle.nDstMap].nMaxNodeCount*mOutsToIn[currentEle.nDstMap].ncount,mBlockSize);
		if(blockSize>mMaxBlock)
			blockSize=mMaxBlock;
		if(currentEle.gTargetCurrent)
		{
			gCurrent=currentEle.gTargetCurrent;
		}
		else
		{
			CUDACHECK(cudaMalloc((LPVOID *)&gCurrent,sizeof(SFNum)*currentEle.nDstNode*copyLen));
			CUDACHECK(cudaMemcpy(gCurrent,currentEle.currentBuffer,sizeof(SFNum)*currentEle.nDstNode*copyLen,cudaMemcpyHostToDevice));
		}
		//if(i>0)
		//printf("Merge %d\n",i);
		TransferSpikesToTarget<<<blockSize,mBlockSize>>>(&mgOutsToIn[currentEle.nDstMap],gCurrent,\
			&mgCurrent[currentPos*mNodeOffsets[TYPE_COUNT]*mGroupLength],mNodeOffsets[TYPE_COUNT],copyLen,mGPUID);
		CUDACHECK(cudaGetLastError());
		if(!currentEle.gTargetCurrent)
			CUDACHECK(cudaFree(gCurrent));
		if(currentEle.currentBuffer)
			delete []currentEle.currentBuffer;
	}
	mCurrentToMerge.clear();
}

bool Simulator::GetNeuronSpikes(SNum neuronIndex,NSPIKE_GEN *spikes,bool bClean)
{
	if(neuronIndex<0 || neuronIndex>=mNodeOffsets[TYPE_COUNT])
		return false;
	CUDACHECK(cudaMemcpy(spikes,&mgRecorder[neuronIndex],sizeof(NSPIKE_GEN),cudaMemcpyDeviceToHost));
	if(bClean)
		CUDACHECK(cudaMemset(&mgRecorder[neuronIndex],0,sizeof(NSPIKE_GEN)));
	return true;
}

void Simulator::Reset()
{
	mNowTSGroup=0;
}

bool Simulator::MapSynapses(SNum nPreNode)
{
	if(mTempSyn.count(nPreNode))
		return true;
	SNum nNodes=mNodeOffsets[TYPE_COUNT];
	SLNum nSyn;
	std::pair<SLNum,SYNAPSE *> syns;
	if(nPreNode<0 || nPreNode>=nNodes)
		return NULL;
	SYNAPSE *pSyn;
	SLNum *nSections=new SLNum[nNodes+1];
	CUDACHECK(cudaMemcpy(nSections,mgSections,sizeof(SLNum)*(nNodes+1),cudaMemcpyDeviceToHost));
	nSyn=nSections[nPreNode+1]-nSections[nPreNode];
	if(nSyn<=0 || !mgSynapses)
	{
		delete []nSections;
		return false;
	}
	pSyn=new SYNAPSE[nSyn];
	CUDACHECK(cudaMemcpy(pSyn,mgSynapses->GetGPUBuffer()+nSections[nPreNode],sizeof(SYNAPSE)*nSyn,cudaMemcpyDeviceToHost));
	syns.first=nSyn;
	syns.second=pSyn;
	mTempSyn[nPreNode]=syns;
	delete []nSections;
	return true;
}

SYNAPSE * Simulator::GetSynapses(SNum preIndex,SNum postIndex)
{
	SNum nNodes=mNodeOffsets[TYPE_COUNT];
	SLNum nSyn;
	std::pair<SLNum,SYNAPSE *> syns;
	if(preIndex<0 || preIndex>=nNodes || postIndex<0 || postIndex>=nNodes)
		return NULL;
	SYNAPSE *pSyn;
	if(!MapSynapses(preIndex))
		return NULL;
	syns=mTempSyn[preIndex];
	nSyn=syns.first;
	pSyn=syns.second;
	for(SNum i=0;i<nSyn;i++)
	{
		if(pSyn[i].postIndex==postIndex)
			return &pSyn[i];
	}
	
	return NULL;
}

bool Simulator::AddNewSynapse(SNum preIndex,SNum postIndex,SFNum weight,SFNum delay)
{
	if(delay<mMinDelay || delay>mMaxDelay || postIndex<0 || postIndex>=mNodeOffsets[TYPE_COUNT])
		return false;
	std::pair<SLNum,SYNAPSE *> syns;
	SLNum nSyn;
	SYNAPSE *pSyn;
	if(MapSynapses(preIndex))
	{
		syns=mTempSyn[preIndex];
		nSyn=syns.first;
		pSyn=syns.second;
		for(SNum i=0;i<nSyn;i++)
		{
			if(pSyn[i].postIndex<0)
			{
				pSyn[i].preIndex=preIndex;
				pSyn[i].postIndex=postIndex;
				pSyn[i].weight=weight;
				pSyn[i].delay=delay;
				return true;
			}
		}
	}
	
	return mExtraSyn->AddNewSynapse(preIndex,postIndex,weight,delay,mgNetwork);
}

bool Simulator::RemoveSynapse(SNum preIndex,SNum postIndex)
{
	std::pair<SLNum,SYNAPSE *> syns;
	SNum nSyn;
	SYNAPSE *pSyn;
	if(MapSynapses(preIndex))
	{
		syns=mTempSyn[preIndex];
		nSyn=syns.first;
		pSyn=syns.second;
		for(SNum i=0;i<nSyn;i++)
		{
			if(pSyn[i].postIndex==postIndex)
			{
				pSyn[i].postIndex=-1;
				return true;
			}
		}

	}
	return mExtraSyn->RemoveSynapse(preIndex,postIndex);
}

bool Simulator::SetSynapse(SNum preIndex,SNum postIndex,SFNum weight,SFNum delay)
{
	SYNAPSE *pSyn;
	if(delay<mMinDelay || delay>mMaxDelay || postIndex<0 || postIndex>=mNodeOffsets[TYPE_COUNT])
		return false;
	pSyn=GetSynapses(preIndex,postIndex);
	if(pSyn)
	{
		pSyn->weight=weight;
		pSyn->delay=delay;
		return true;
	}
	return mExtraSyn->SetSynapse(preIndex,postIndex,weight,delay);
}

bool Simulator::SubmitSynapseChange()
{
	SNum nNodes=mNodeOffsets[TYPE_COUNT];
	//if(mTempSyn.size()<=0)
		//return true;
	std::pair<SLNum,SYNAPSE *> syns;
	SNum nSyn;
	SLNum synPos,splitN;
	SNum preIndex;
	SYNAPSE *pSyn;
	SLNum *nSections=new SLNum[nNodes+1];
	CUDACHECK(cudaMemcpy(nSections,mgSections,sizeof(SLNum)*(nNodes+1),cudaMemcpyDeviceToHost));
	
	for(std::map<SNum,std::pair<SLNum,SYNAPSE *>>::iterator it=mTempSyn.begin();it!=mTempSyn.end() && mgSynapses;it++)
	{
		preIndex=it->first;
		syns=it->second;
		nSyn=syns.first;
		pSyn=syns.second;
		synPos=nSections[preIndex]/mgSynapses->GetGPULen();
		if(synPos!=mgSynapses->GetGridPos())
		{
			mgSynapses->SwitchToGrid(synPos);
		}
		if((nSections[preIndex]+nSyn)>(mgSynapses->GetGridOffset()+mgSynapses->GetGPULen()))
		{
			splitN=mgSynapses->GetGridOffset()+mgSynapses->GetGPULen()-nSections[preIndex];
			CUDACHECK(cudaMemcpy(mgSynapses->GetGPUBuffer()+nSections[preIndex]-mgSynapses->GetGridOffset(),\
			pSyn,sizeof(SYNAPSE)*splitN,cudaMemcpyHostToDevice));
			if(mgSynapses->SwitchToGrid(synPos+1))
			CUDACHECK(cudaMemcpy(mgSynapses->GetGPUBuffer()+nSections[preIndex]+splitN-mgSynapses->GetGridOffset(),\
			pSyn,sizeof(SYNAPSE)*(nSyn-splitN),cudaMemcpyHostToDevice));

		}
		else
		{
			CUDACHECK(cudaMemcpy(mgSynapses->GetGPUBuffer()+nSections[preIndex]-mgSynapses->GetGridOffset(),\
			pSyn,sizeof(SYNAPSE)*nSyn,cudaMemcpyHostToDevice));
		}
		delete []pSyn;
	}
	mTempSyn.clear();
	delete []nSections;
	mExtraSyn->SubmitChange();
	return true;
}

}