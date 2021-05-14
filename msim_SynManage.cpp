
#include <cuda_runtime.h>

#include "util.h"

#include "msim_Simulator.h"
#include "msim_SynManage.h"

namespace MultiGPUBrain
{

SynManager::SynManager(SNum nodeCount,SNum meshSize)
:mNodeCount(nodeCount)
,mMeshSize(meshSize)
,mgNode2Syn(NULL)
,mgLinkTable(NULL)
,mgSynapses(NULL)
,mMeshCount(0)
,mMaxMeshCount(0)
{
    mNode2Syn=new SNum[nodeCount];
    for(SNum i=0;i<nodeCount;i++)
    {
        mNode2Syn[i]=-1;
    }
    CUDACHECK(cudaMalloc((void **)&mgNode2Syn,sizeof(SNum)*nodeCount));
    CUDACHECK(cudaMemcpy(mgNode2Syn,mNode2Syn,sizeof(SNum)*nodeCount,cudaMemcpyHostToDevice));
}

SynManager::~SynManager()
{
    cudaFree(mgSynapses);
    cudaFree(mgLinkTable);
    cudaFree(mgNode2Syn);
    delete []mNode2Syn;
    clear();

}

void SynManager::clear()
{
    SNum n;
    for(std::map<SNum,std::vector<SYN_SEC>>::iterator it=mTempSyn.begin();it!=mTempSyn.end();it++)
    {
        n=(SNum)it->second.size();
        for(SNum i=0;i<n;i++)
        {
            delete []it->second[i].pSyn;
        }
    }
    mTempSyn.clear();
}

bool SynManager::PushForward(SNum preIndex,SYN_SEC *pSS)
{
    SNum nextSec;
    SYN_SEC ss;
    if(preIndex<0 || preIndex>=mNodeCount || mNode2Syn[preIndex]<0)
        return false;
    
    if(mTempSyn.count(preIndex) && mTempSyn[preIndex].size()>0)
    {
        nextSec=mLinkTable[mTempSyn[preIndex].end()->nOffset/mMeshSize];
    }
    else
    {
        nextSec=mNode2Syn[preIndex];
    }
    if(nextSec<0)
        return false;
    ss.nOffset=nextSec*mMeshSize;
    ss.nSize=mMeshSize;
    ss.pSyn=new SYNAPSE[mMeshSize];
    CUDACHECK(cudaMemcpy(ss.pSyn,mgSynapses+ss.nOffset,sizeof(SYNAPSE)*ss.nSize,cudaMemcpyDeviceToHost));
    mTempSyn[preIndex].push_back(ss);
    *pSS=ss;
    return true;
}

bool SynManager::AddNewSynapse(SNum preIndex,SNum postIndex,SFNum weight,SFNum delay,NETWORK_DATA *gNetwork)
{
    if(preIndex<0 || preIndex>=mNodeCount || postIndex<0 || postIndex>=mNodeCount)
        return false;
        
    SNum i;
    SYNAPSE *gSyn;
    SYNAPSE syn={0};
    SNum * gLinkTable;
    SYN_SEC ss={0};
    NETWORK_DATA nd;
    //先在正处于编辑状态的空间内查找
    if(mTempSyn.count(preIndex))
    {
        for(i=0;i<mTempSyn[preIndex].size();i++)
        {
            ss=mTempSyn[preIndex][i];
            for(SNum j=0;j<ss.nSize;j++)
            {
                if(ss.pSyn[i].postIndex<0)//如果找到了空位，就直接填写
                {
                    ss.pSyn[i].preIndex=preIndex;
                    ss.pSyn[i].postIndex=postIndex;
                    ss.pSyn[i].weight=weight;
                    ss.pSyn[i].delay=delay;
                    return true;
                }
            }
        }
    }
    while(PushForward(preIndex,&ss))//然后在已有的空间内寻找空闲的突触结构
    {
        for(i=0;i<ss.nSize;i++)
        {
            if(ss.pSyn[i].postIndex<0)//如果找到了空位，就直接填写
            {
                ss.pSyn[i].preIndex=preIndex;
                ss.pSyn[i].postIndex=postIndex;
                ss.pSyn[i].weight=weight;
                ss.pSyn[i].delay=delay;
                return true;
            }
        }
    }
    //未找到则开辟新空间
    mLinkTable.push_back(-1);
    CUDACHECK(cudaMalloc((void **)&gSyn,sizeof(SYNAPSE)*(mMeshCount+1)*mMeshSize));
    CUDACHECK(cudaMalloc((void **)&gLinkTable,sizeof(SNum)*(mMeshCount+1)));
    if(mMeshCount>0)
    {
        CUDACHECK(cudaMemcpy(gSyn,mgSynapses,sizeof(SYNAPSE)*mMeshCount*mMeshSize,cudaMemcpyDeviceToDevice));
        CUDACHECK(cudaMemcpy(gLinkTable,mgLinkTable,sizeof(SNum)*mMeshCount,cudaMemcpyDeviceToDevice));
        cudaFree(mgSynapses);
        cudaFree(mgLinkTable);
    }
    
    if(mNode2Syn[preIndex]<0)
    {
        CUDACHECK(cudaMemcpy(mgNode2Syn+preIndex,&mMeshCount,sizeof(mMeshCount),cudaMemcpyHostToDevice));
        mNode2Syn[preIndex]=mMeshCount;
    }
    
    mgSynapses=gSyn;
    mgLinkTable=gLinkTable;
    //如果之前已经有分配空间，则将之前的空间链表的结尾指向新开辟的空间
    if(ss.nSize>0)
    {
        i=mMeshCount;
        CUDACHECK(cudaMemcpy(gLinkTable+ss.nOffset/mMeshSize,&i,sizeof(SNum),cudaMemcpyHostToDevice));
    }
    i=-1;
    CUDACHECK(cudaMemcpy(gLinkTable+mMeshCount,&i,sizeof(SNum),cudaMemcpyHostToDevice));
    //将新的突触填入新的待编辑主存空间
    ss.nOffset=mMeshSize*mMeshCount;
    ss.nSize=mMeshSize;
    ss.pSyn=new SYNAPSE[mMeshSize];
    ss.pSyn[0].preIndex=preIndex;
    ss.pSyn[0].postIndex=postIndex;
    ss.pSyn[0].weight=weight;
    ss.pSyn[0].delay=delay;
    mTempSyn[preIndex].push_back(ss);
    if(mTempSyn[preIndex].size()>mMaxMeshCount)
        mMaxMeshCount=mTempSyn[preIndex].size();
    syn.preIndex=preIndex;
    syn.postIndex=-1;
    for(i=0;i<mMeshSize;i++)
    {
        CUDACHECK(cudaMemcpy(gSyn+mMeshCount*mMeshSize+i,&syn,sizeof(SYNAPSE),cudaMemcpyHostToDevice));
    }

    //将gNetwork中的相应数据更改
    CUDACHECK(cudaMemcpy(&nd,gNetwork,sizeof(NETWORK_DATA),cudaMemcpyDeviceToHost));
    nd.node2Syn=mgNode2Syn;
    nd.linkTable=mgLinkTable;
    nd.extraSynapses=mgSynapses;
    nd.maxExtraCount=mMaxMeshCount*mMeshSize;
    CUDACHECK(cudaMemcpy(gNetwork,&nd,sizeof(NETWORK_DATA),cudaMemcpyHostToDevice));

    mMeshCount++;
    return true;
}

bool SynManager::RemoveSynapse(SNum preIndex,SNum postIndex)
{
    if(preIndex<0 || preIndex>=mNodeCount || postIndex<0 || postIndex>=mNodeCount)
        return false;
        
    SYN_SEC ss={0};
    //先在正处于编辑状态的空间内查找
    if(mTempSyn.count(preIndex))
    {
        for(SNum i=0;i<mTempSyn[preIndex].size();i++)
        {
            ss=mTempSyn[preIndex][i];
            for(SNum j=0;j<ss.nSize;j++)
            {
                if(ss.pSyn[i].postIndex==postIndex)//如果找到了空位，就直接填写
                {
                    ss.pSyn[i].postIndex=-1;
                    return true;
                }
            }
        }
    }
    while(PushForward(preIndex,&ss))//然后在已有的空间内寻找空闲的突触结构
    {
        for(SNum i=0;i<ss.nSize;i++)
        {
            if(ss.pSyn[i].postIndex==postIndex)//如果找到了空位，就直接填写
            {
                ss.pSyn[i].postIndex=-1;
                return true;
            }
        }
    }
    return false;
}

bool SynManager::SetSynapse(SNum preIndex,SNum postIndex,SFNum weight,SFNum delay)
{
    if(preIndex<0 || preIndex>=mNodeCount || postIndex<0 || postIndex>=mNodeCount)
        return false;
        
    SYN_SEC ss={0};
    //先在正处于编辑状态的空间内查找
    if(mTempSyn.count(preIndex))
    {
        for(SNum i=0;i<mTempSyn[preIndex].size();i++)
        {
            ss=mTempSyn[preIndex][i];
            for(SNum j=0;j<ss.nSize;j++)
            {
                if(ss.pSyn[i].postIndex==postIndex)//如果找到了空位，就直接填写
                {
                    ss.pSyn[i].weight=weight;
                    return true;
                }
            }
        }
    }
    while(PushForward(preIndex,&ss))//然后在已有的空间内寻找空闲的突触结构
    {
        for(SNum i=0;i<ss.nSize;i++)
        {
            if(ss.pSyn[i].postIndex==postIndex)//如果找到了空位，就直接填写
            {
                ss.pSyn[i].weight=weight;
                return true;
            }
        }
    }
    return false;
}

void SynManager::SubmitChange()
{
    SNum n;
    SYN_SEC ss={0};
    for(std::map<SNum,std::vector<SYN_SEC>>::iterator it=mTempSyn.begin();it!=mTempSyn.end();it++)
    {
        n=(SNum)it->second.size();
        for(SNum i=0;i<n;i++)
        {
            ss=it->second[i];
            CUDACHECK(cudaMemcpy(mgSynapses+ss.nOffset,ss.pSyn,sizeof(SYNAPSE)*ss.nSize,cudaMemcpyHostToDevice));
        }
    }
    clear();
}

}