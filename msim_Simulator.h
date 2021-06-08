#pragma once
#include <stdio.h>
#include <vector>
#include <map>
#include <omp.h>

#include "msim_schedule.h"

inline void CheckCall(cudaError_t err,const char *file,int line)
{
	const cudaError_t error = err;
	if(error!=cudaSuccess)
	{
		printf("Error:%s.Line %d,",file,line);
		printf("code:%d, reason:%s\n",error,cudaGetErrorString(error));
		exit(1);
	}
}



#define CUDACHECK(x) CheckCall(x,__FILE__,__LINE__)

void SetToGPU(void **obj,void *src,unsigned long long length);

struct GNLIF;
struct RECORDER;
struct NEXTRA_INFO;

struct NETWORK_DATA;
struct _SYN_BUILD;
typedef struct _SYN_BUILD SYN_BUILD;


bool BuildSynapse(SNum nBuild,SYN_BUILD builds[],SNum *pNodeCount,SLNum **gSections,MultiGPUBrain::MemSchedule<SYNAPSE> **gSynapses,SNum *nMaxSynCount=NULL,SLNum nGridSize=0,SNum nMeshSize=1);

#define MYGRID_SIZE(a,b) ((a+b-1)/b)

namespace MultiGPUBrain
{
class PopMap;
class SynManager;
struct POP_MAPS;

class Simulator//对应一个GPU内的仿真器,只负责单个时间片组的仿真
{
public:
    Simulator(SNum nMeshSize=1);
    ~Simulator();
    void Lock();
    void Unlock();
    bool SetSpikeGenerator(GNGen *gens,SNum len);
    bool SetLIFNodes(GNLIF *lifs,SNum nLIF,LIF_ARG *lifArg,SNum nLifArg);//一次性设置LIF节点,nLIF代表总的节点数,nLifArg代表LIF_ARG结构体数量
    bool SetSynapse(SNum nBuild,SYN_BUILD builds[]);
    bool SetOutputSynapse(SNum nBuild,SYN_BUILD builds[],Simulator *pTarget);//设置输出到其它分部的突触
    bool Prepare(SFNum timestep,SNum nTSGroupLen=-1,SNum nTSMaxDelay=-1,SNum nBlockSize=BLOCKSIZE);//nTSGroupLen为正表示指定时间片组的长度,nTSMaxDelay表示最大延迟对应的时间片长度
    bool Simulate(bool bSelectActiveNeuron);//继续下一个时间片组的模拟
    bool TransferSpikes();//多线程并行将上一次时间片组模拟的脉冲传输给目标分部
    void MergeSpikes();//多线程并行吸收来自其它分部的脉冲电流
    bool GetNeuronSpikes(SNum neuronIndex,NSPIKE_GEN *spikes,bool bClean=true);//获取指定神经元的脉冲
    SYNAPSE * GetSynapses(SNum preIndex,SNum postIndex);//获取指定的突触
    bool AddNewSynapse(SNum preIndex,SNum postIndex,SFNum weight,SFNum delay);//仿真过程中添加新的突触
    bool RemoveSynapse(SNum preIndex,SNum postIndex);
    bool SetSynapse(SNum preIndex,SNum postIndex,SFNum weight,SFNum delay);
    bool SubmitSynapseChange();//仿真过程中提交突触修改
    bool SetSpikeGenerator(SNum index,const GNGen &gens);//修改某脉冲生成器的输入脉冲序列
    void UseMyGPU();//使用本仿真器的GPU
    void Reset();//重置网络状态，以便从头开始仿真
private:
    bool PrepareSynapseSize();//仿真运行时存放突触所需要的显存
    bool BuildInnerSynapse();//真正开始构建内部突触数据
    bool BuildOutterSynapse();//真正开始构建外部突触数据
    void CleanOutterBuilds();
    void CleanLIF();
    void CleanSynapse();
    void CleanOutputSynapse();
    void CleanSimulData();
    void UpdateNodeOffset();
    bool MapSynapses(SNum nPreNode);//将GPU中的突触映射到主存中
private:
    SNum mInnerBuildCount,mOutterBuildCount;//待构建的突触构建信息结构体数
    SYN_BUILD *mInnerBuild,*mOutterBuild;//待构建的突触构建信息结构体
    std::vector<std::pair<SNum,SYN_BUILD *>> mOutterBuilds;//待构建的外部突触信息结构体
    omp_lock_t mLock;
    SNum mBlockSize;//线程块大小
    bool mNodeChanged;
    SNum mPopCount[TYPE_COUNT];//各类族群的数量
    SNum mNodeCounts[TYPE_COUNT];//各类节点的数量
    SNum mNodeOffsets[TYPE_COUNT+1];//各类节点的偏移量
    SFNum mTimestep;
    SNum mMaxTSDelay;//最大延迟所对应的时间片个数
    SNum mGroupLength;//一个时间片组的长度
    SNum mCurrentGroupCount;//输入电流数组的占用的时间片组的个数
    SNum mMaxBlock;//最大线程块数
    SNum mNowTSGroup;//当前准备要模拟的时间片组下标
    SNum mMeshSize;//该GPU仿真器存储突触的粒度大小
    std::vector<CURRENT_ELEMENT> mCurrentToMerge;//需要融合进当前分部的脉冲电流
    std::map<SNum,std::pair<SLNum,SYNAPSE *>> mTempSyn;//神经元->突触数，所有输出突触，突触在主存的临时存放处
    SynManager *mExtraSyn;//管理运行过程中扩充的突触
private:
    GNGen *mgGens;//各个分部的脉冲生成器
    
    GNLIF *mgLIF;
    LIF_ARG *mgLifArg;
    MemSchedule<SYNAPSE> *mgSynapses;
    SLNum *mgSections;
    std::vector<OUTPUT_SYNAPSES> mOutputSyn;//输出到其他分部的突触
    std::vector<MemSchedule<SYNAPSE> *> mOutputSynData;//输出到其他分部的完整突触数据
    SLNum mMaxGrid;//所有的突触数据中的最大块数
    SLNum mInnerGridSize;//内部突触的块大小
    std::vector<PopMap> mInToOuts;//向外的族群映射，在向外输出脉冲时使用，与mOutputSyn一一对应
    std::vector<POP_MAPS> mOutsToIn;//向内的族群映射，后续需要将其传入到GPU再使用

    OUTPUT_SYNAPSES *mgOutputSyn;
    NSPIKE_GEN *mgRecorder;//记录各个脉冲
    NETWORK_DATA *mgNetwork;
    SFNum *mgCurrent;//记录一个时间片组内各个时间片下各个神经元的输入电流
    SNum *mgActiveCount;//记录一个时间片组内各个时间片下活动神经元的个数
    SNum *mgActiveIndex;//记录一个时间片组内各个时间片下活动神经元的下标
    SNum *mgActiveCountByGrid;//每个调度格子下的处于激活状态的神经元数
    POP_MAPS *mgOutsToIn;//向内的族群映射，与mOutsToIn大小相同，只是在GPU中
public:
    SNum mGPUID;
    SFNum mMinDelay;
    SFNum mMaxDelay;
};


}