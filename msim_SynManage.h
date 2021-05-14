#pragma once

#include <vector>
#include <map>

namespace MultiGPUBrain
{

struct SYN_SEC
{
    SNum nSize;//当前突触块的大小
    SNum nOffset;//当前突触块在整个突触数组的偏移
    SYNAPSE *pSyn;
};

//专门管理GPU内的突触数据，方便按神经元随时增删改查
class SynManager
{
public:
    SynManager(SNum nodeCount,SNum meshSize);
    ~SynManager();
    void clear();
    bool AddNewSynapse(SNum preIndex,SNum postIndex,SFNum weight,SFNum delay,NETWORK_DATA *gNetwork);//添加新的突触，如果需要扩充数组空间，则顺便将gNetwork中的相应数据更改
    bool RemoveSynapse(SNum preIndex,SNum postIndex);
    bool SetSynapse(SNum preIndex,SNum postIndex,SFNum weight,SFNum delay);
    void SubmitChange();

private:
    bool PushForward(SNum preIndex,SYN_SEC *pSS);//按照mTempSyn中的进度向前遍历一个突触块，将其内容存入pSS
    //SNum ExpandSynapseBuf();//扩充一个突触块

private:
    SNum mMeshSize;//突触块包含的突触个数
    SNum mNodeCount;
    SNum mMeshCount;//当前突触数组已分配的按粒度数的大小
    std::vector<SNum> mLinkTable;//每个突触块到下一个突触块的下标，相当于链表的下一指针
    SNum * mNode2Syn;//节点到首个突触块下标的映射
    SNum mMaxMeshCount;//单个神经元所占用的最大突触块

    std::map<SNum,std::vector<SYN_SEC>> mTempSyn;//当前正在编辑的突触

private:
    SNum * mgNode2Syn;
    SNum *mgLinkTable;
    SYNAPSE *mgSynapses;
};

}