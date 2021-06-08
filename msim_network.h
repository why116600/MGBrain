#pragma once

#include <map>
#include <string>
#include <vector>

#include "util.h"
//#include "msim_Simulator.h"

namespace MultiGPUBrain
{
class Simulator;

struct POP_IN_PART//描述一个族群的节点在不同分部下的分布情况
{
    SNum popOffset;//所在族群的首节点下标
    SNum offset;//所在分部逻辑大数组的首节点下标
    SNum count;
    SNum part;
};

enum PARTITION_MODEL//划分模式
{
    LoadBalance,//在负载均衡的情况下尽可能降低不同GPU的通信量
    FIFP,//先进先划分
    Average//按GPU平均分
};

#define BEFORE_PART(a,b) ((a.part<b.part) || (a.part==b.part && a.offset<b.offset))

class Population
{
public:
    SNum mCount;
    std::string mType;
    std::map<std::string,SFNum> mArgs;
private:
    std::vector<POP_IN_PART> mParts;
public:
    Population(const char *szType,SNum nCount,const std::map<std::string,SFNum> &args)
    {
        mType=szType;
        mCount=nCount;
        mArgs=args;
    }

    Population &operator =(const Population& pop)
    {
        mCount=pop.mCount;
        mType=pop.mType;
        mArgs=pop.mArgs;
        mParts=pop.mParts;
        return *this;
    }

    bool SendArg(const char *szArg,SFNum *fDst,SFNum defValue)//发送参数到fDst，如果没有，就按照defValue
    {
        if(!mArgs.count(szArg))
        {
            *fDst=defValue;
            return false;
        }
        *fDst=mArgs[szArg];
        return true;
    }

    bool InsertPart(SNum offset,SNum count,SNum part);//按照顺序插入
    bool GetPart(SNum index,POP_IN_PART &pp);//根据节点在族群的下标找到对应的分布
};

struct POP_TO_ARRAY
{
    SNum srcOffset,dstOffset;
    SNum ncount;
};

//一组节点映射，相当于PopMap的结构体版本
struct POP_MAPS
{
    SNum ncount;//映射个数
    SNum nodeCount;//参与映射的所有节点的个数
    SNum nMaxNodeCount;//单个映射最大的节点数
    POP_TO_ARRAY *maps;
};

class PopMap//将若干段编号不连续的族群节点段映射到一个统一的连续的数组
{
private:
    SNum mArrayLen=0;
    std::vector<POP_TO_ARRAY> mPopToArrays;
    std::map<std::pair<SNum,SNum>,std::pair<SNum,SNum>> mSecToPop;//源区段到目标区段的映射
public:
    SNum AddPop(SNum offset,SNum ncount)
    {
        POP_TO_ARRAY pta;
        std::pair<SNum,SNum> sec,dst;
        sec.first=offset;
        sec.second=ncount;
        pta.srcOffset=offset;
        pta.ncount=ncount;
        if(mSecToPop.count(sec))//如果某个源区段之前已经映射过了，则按照之前的映射
        {
            pta.dstOffset=mSecToPop[sec].first;
        }
        else//否则就建立新的映射
        {
            pta.dstOffset=mArrayLen;
            dst.first=mArrayLen;
            dst.second=ncount;
            mSecToPop[sec]=dst;
            mArrayLen+=ncount;
            mPopToArrays.push_back(pta);
        }
        return pta.dstOffset;
    }
    void Clear()
    {
        mArrayLen=0;
        mPopToArrays.clear();
    }
    SNum GetArrayLength()
    {
        return mArrayLen;
    }
    SNum GetCount()
    {
        return (SNum)mPopToArrays.size();
    }
    POP_TO_ARRAY &operator [](SNum index)
    {
        return mPopToArrays[index];
    }
    PopMap &operator =(const PopMap &obj)
    {
        mArrayLen=obj.mArrayLen;
        mPopToArrays=obj.mPopToArrays;
        return *this;
    }

};

class Network
{
private:
	SFNum mTimestep;
	std::map<std::string,SNum> mTypeMap;
    //原始的神经网络结构数据
    std::vector<Population> mPops[TYPE_COUNT];
    std::vector<POP_CONN> mSynapses;
    //编译过程中的数据
    SNum mOffsets[TYPE_COUNT+1];//各个类别的节点
    //编译后用于仿真的数据
    std::map<SNum,std::vector<SFNum>> *mNeuronToSpikes;//神经元下标->脉冲序列，用于记录指定神经元的脉冲序列
    //神经元下标->(进度，脉冲序列)，用于记录输入脉冲序列
    std::map<SNum,std::pair<SNum,std::vector<SFNum>>> *mGenSpikeTrains;
    SFNum mMinDelay;//最小延迟，作为时间片组长度
    SFNum mMaxDelay;//最大延迟
    SNum mPartCount;//部分数
    Simulator *mSimulators;//真正执行仿真程序的类

private:
    void CopyNodesToSimulator(Simulator *pSim,std::vector<Population> pops[TYPE_COUNT]);
    bool LocateNeuron(SNum popID,SNum nIndex,std::pair<SNum,SNum> &ret);//定位某节点所在分部及其分部内的下标

public:
    Network(SFNum timestep);
    void clean();
    SNum GetGPUCount();
    SNum CreateSpikeGenerator();
    SNum CreatePopulation(const char *szType,SNum nCount,const std::map<std::string,SFNum> &args);
    bool Connect(SNum preID,SNum postID,SFNum weight,SFNum delay,bool bOneToOne,double fProba);
    bool Compile(SNum nPart,PARTITION_MODEL pm=LoadBalance,SNum meshSize=1,SFNum minDelay=1.0,SNum nblockSize=BLOCKSIZE);//将网络分为nPart个分部进行编译
    //以下是完成编译后的操作
    bool SetSpikeTrain(SNum genID,const std::vector<SFNum> &spikes);
    bool WatchNeuron(SNum popID,SNum index);//监视指定的神经元，从而获取完整的脉冲序列
    bool Simulate(SFNum simulTime);//开始仿真
    bool GetNeuronSpikes(SNum popID,SNum index,std::vector<SFNum> &times);//获取指定族群，指定
    bool GetSynapseInfo(SNum preID,SNum preIndex,SNum postID,SNum postIndex,SYNAPSE *pSyn);//获取指定突触的信息
    bool Connect(SNum preID,SNum preIndex,SNum postID,SNum postIndex,SFNum weight,SFNum delay);//修改或增加指定突触
    bool Disconnect(SNum preID,SNum preIndex,SNum postID,SNum postIndex);//断开突触
};

}