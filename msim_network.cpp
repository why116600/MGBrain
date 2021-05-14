#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "util.h"

#include "msim_Simulator.h"
#include "msim_network.h"
#include "msim_order.h"
#include "PopGraph.h"


namespace MultiGPUBrain
{

Network::Network(SFNum timestep)
:mTimestep(timestep)
,mSimulators(NULL)
,mNeuronToSpikes(NULL)
,mGenSpikeTrains(NULL)
{
	mTypeMap.insert(std::pair<std::string,SNum>("LIF",TYPE_LIF));
}

SNum Network::GetGPUCount()
{
    SNum gpuCount;
	CUDACHECK(cudaGetDeviceCount(&gpuCount));
    return gpuCount;
}

void Network::clean()
{
    if(mGenSpikeTrains)
        delete []mGenSpikeTrains;
    if(mNeuronToSpikes)
        delete []mNeuronToSpikes;
    mNeuronToSpikes=NULL;
    if(mSimulators)
        delete []mSimulators;
    mSimulators=NULL;
}

SNum Network::CreateSpikeGenerator()
{
    SNum nNo;
    std::map<std::string,SFNum>  params;
    Population pop("GEN",1,params);
    nNo=(SNum)mPops[TYPE_GEN].size();
    mPops[TYPE_GEN].push_back(pop);
    return GETID(TYPE_GEN,nNo);
}

SNum Network::CreatePopulation(const char *szType,SNum nCount,const std::map<std::string,SFNum> &args)
{
    std::string type=szType;
    std::map<std::string,SNum>::iterator it;
    SNum nType,nNo;
    it=mTypeMap.find(type);
    if(it==mTypeMap.end())
    {
        return -1;
    }
    Population pop(szType,nCount,args);
    nType=it->second;
    nNo=(SNum)mPops[nType].size();
    mPops[nType].push_back(pop);
    return GETID(nType,nNo);
}

bool Network::Connect(SNum preID,SNum postID,SFNum weight,SFNum delay,bool bOneToOne,double fProba)
{
    SNum preCount,postCount;
    SNum type,index;
    POP_CONN syn;
    type=GETTYPE(preID);
    index=GETNO(preID);
    if(type>=TYPE_COUNT || type<0)
        return false;
    preCount=(SNum)mPops[type].size();
    if(index>=preCount || index<0)
        return false;
    type=GETTYPE(postID);
    index=GETNO(postID);
    if(type>=TYPE_COUNT || type<0)
        return false;
    postCount=(SNum)mPops[type].size();
    if(index>=postCount || index<0)
        return false;
    if(bOneToOne && preCount!=postCount)
        return false;
    syn.preID=preID;
    syn.postID=postID;
    syn.weight=weight;
    syn.delay=delay;
    syn.bOneToOne=bOneToOne;
    syn.fProba=fProba;
    index=-1;
    for(SNum i=0;i<(SNum)mSynapses.size();i++)
    {
        if(syn.preID<mSynapses[i].preID || (syn.preID==mSynapses[i].preID && syn.postID<mSynapses[i].postID))
        {
            index=i;
            break;
        }
    }
    if(index<0)
        mSynapses.push_back(syn);
    else
        mSynapses.insert(mSynapses.begin()+index,syn);
    
    return true;
}

void Network::CopyNodesToSimulator(Simulator *pSim,std::vector<Population> pops[TYPE_COUNT])
{
    SNum offset,i,j,pos,index,nodeCount=0,len;
    LIF_ARG *pLifArg;
    GNLIF *pLIF;
    offset=0;
    //构建LIF节点数据
    pLifArg=new LIF_ARG[(pops[TYPE_LIF].size()+MAXLIF-1)/MAXLIF];
    for(i=0;i<(SNum)pops[TYPE_LIF].size();i++)//计算LIF节点总数,并复制参数
    {
        pos=i/MAXLIF;
        index=i%MAXLIF;
        nodeCount+=pops[TYPE_LIF][i].mCount;
        pops[TYPE_LIF][i].SendArg("V_init",&pLifArg[pos].V_init[index],-70.0);
        pops[TYPE_LIF][i].SendArg("V_reset",&pLifArg[pos].V_reset[index],-70.0);
        pops[TYPE_LIF][i].SendArg("V_th",&pLifArg[pos].V_th[index],-55.0);
        pops[TYPE_LIF][i].SendArg("Tau_m",&pLifArg[pos].Tau_m[index],10.0);
        pops[TYPE_LIF][i].SendArg("C_m",&pLifArg[pos].C_m[index],250.0);
        pops[TYPE_LIF][i].SendArg("I_e",&pLifArg[pos].I_e[index],376.0);
        pops[TYPE_LIF][i].SendArg("T_ref",&pLifArg[pos].T_ref[index],2.0);
        pops[TYPE_LIF][i].SendArg("tau_ex_",&pLifArg[pos].tau_ex_[index],2.0);
        pops[TYPE_LIF][i].SendArg("tau_in_",&pLifArg[pos].tau_in_[index],2.0);
    }
    len=(nodeCount+MAXLIF-1)/MAXLIF;
    pLIF=new GNLIF[len];
    memset(pLIF,0,sizeof(GNLIF)*len);
    for(i=0;i<(SNum)pops[TYPE_LIF].size();i++)//建立节点到参数的映射
    {
        for(j=0;j<pops[TYPE_LIF][i].mCount;j++)
        {
            pos=(offset+j)/MAXLIF;
            index=(offset+j)%MAXLIF;
            pLIF[pos].argPos[index]=i/MAXLIF;
            pLIF[pos].argIndex[index]=i%MAXLIF;
            pops[TYPE_LIF][i].SendArg("V_init",&pLIF[pos].MP[index],-70.0);
        }
        offset+=pops[TYPE_LIF][i].mCount;
    }
    pSim->SetLIFNodes(pLIF,nodeCount,pLifArg,(pops[TYPE_LIF].size()+MAXLIF-1)/MAXLIF);
    delete []pLifArg;
    delete []pLIF;
    //构建脉冲生成器节点
    if(pops[TYPE_GEN].size()>0)
    {
        GNGen *gens=new GNGen[pops[TYPE_GEN].size()];
        memset(gens,0,pops[TYPE_GEN].size()*sizeof(GNGen));
        pSim->SetSpikeGenerator(gens,(SNum)pops[TYPE_GEN].size());
        delete []gens;
    }
}

bool Network::Compile(SNum nPart,SNum meshSize,SFNum minDelay,SNum nBlockSize)
{
    SNum i,j,type,pos,index,popIndex,preIndex,postIndex,len,offset;
    SNum synCount=0,nodeCount=0;
    SNum nodeCounts[TYPE_COUNT]={0};
    POP_CONN conn;
    std::vector<Population> PopsByType[TYPE_COUNT];
    std::vector<POP_INFO> pops;
    LIF_ARG *pLifArg=NULL;
    GNLIF *pLIF=NULL;
    bool bRet=false;
    mOffsets[0]=0;
#ifndef FAKE_MULTI_GPU
    if(GetGPUCount()<nPart)
        return false;
#endif
    for(i=1;i<=TYPE_COUNT;i++)
    {
        mOffsets[i]=mOffsets[i-1]+(SNum)mPops[i-1].size();
    }
    SNum popCounts[mOffsets[TYPE_COUNT]];//各个族群的节点数
    SNum popOffsets[mOffsets[TYPE_COUNT]+1]={0};//各个族群的首节点在逻辑大数组中的偏移
    //将突触数据按前神经元的编号排序
    for(i=(SNum)mSynapses.size()-1;i>0;i--)
    {
        for(j=0;j<i;j++)
        {
            if(GETTYPE(mSynapses[j].preID)>GETTYPE(mSynapses[j+1].preID) ||
            (GETTYPE(mSynapses[j].preID)==GETTYPE(mSynapses[j+1].preID) && GETNO(mSynapses[j].preID)>GETNO(mSynapses[j+1].preID) ))
            {
                conn=mSynapses[j];
                mSynapses[j]=mSynapses[j+1];
                mSynapses[j+1]=conn;
            }
        }
    }
    //求最小延迟
    if(minDelay>0.0)
    {
        mMinDelay=minDelay;
        mMaxDelay=minDelay;
    }
    else
    {
        mMinDelay=mSynapses[0].delay;
        mMaxDelay=mSynapses[0].delay;
    }
    
    for(i=0;i<(SNum)mSynapses.size();i++)
    {
        if(mMinDelay>mSynapses[i].delay)
            mMinDelay=mSynapses[i].delay;
        if(mMaxDelay<mSynapses[i].delay)
            mMaxDelay=mSynapses[i].delay;
    }
    //统计族群信息
    for(i=0;i<TYPE_COUNT;i++)
    {
        for(j=0;j<(SNum)mPops[i].size();j++)
        {
            popCounts[mOffsets[i]+j]=mPops[i][j].mCount;
            popOffsets[mOffsets[i]+j+1]=popOffsets[mOffsets[i]+j]+mPops[i][j].mCount;
        }
    }
    if(nPart>1)//需要划分
    {
        //安排外部突触处理顺序的类
        OrderArrange oa(nPart);
        std::vector<SNum> orders;
        //将拓扑结构数据输入划分算法中
        PopGraph pg(mOffsets[TYPE_COUNT],popCounts);
        SYN_BUILD builds[nPart][mSynapses.size()];
        if(!oa.Arrange())
            goto compile_end;
        for(i=0;i<(SNum)mSynapses.size();i++)
        {
            type=GETTYPE(mSynapses[i].preID);
            index=GETNO(mSynapses[i].preID);
            preIndex=mOffsets[type]+index;
            type=GETTYPE(mSynapses[i].postID);
            index=GETNO(mSynapses[i].postID);
            postIndex=mOffsets[type]+index;
            pg.AddConn(preIndex,postIndex,mSynapses[i].bOneToOne,mSynapses[i].fProba,mSynapses[i].weight,mSynapses[i].delay);
        }
        //统计总节点数
        for(i=0;i<(SNum)mPops[TYPE_LIF].size();i++)//计算节点总数
        {
            nodeCount+=mPops[TYPE_LIF][i].mCount;
        }
        srand(time(NULL));
        pg.StartPartition(nPart);
        pg.Partition(nodeCount/10,0.1);
        mSimulators=new Simulator[nPart];
        for(i=0;i<nPart;i++)
        {
            mSimulators[i].mGPUID=i;//分配GPU
            mSimulators[i].mMaxDelay=mMaxDelay;
            mSimulators[i].mMinDelay=mMinDelay;
        }
        for(i=0;i<nPart;i++)//逐个分部构建
        //#pragma omp parallel num_threads(nPart)
        {
            mSimulators[i].UseMyGPU();
            //处理神经元数据
            if(!pg.GetPopsInPart(i,pops))
            {
                goto compile_end;
            }
            offset=0;
            //根据节点类型进行不同的处理
            pLifArg=new LIF_ARG[(mPops[TYPE_LIF].size()+MAXLIF-1)/MAXLIF];
            for(j=0;j<(SNum)pops.size();j++)
            {
                if(pops[j].nPopIndex>=mOffsets[TYPE_COUNT])
                {
                    goto compile_end;
                }
                if(pops[j].nPopIndex>=mOffsets[TYPE_LIF])
                {
                    popIndex=pops[j].nPopIndex-mOffsets[TYPE_LIF];
                    PopsByType[TYPE_LIF].push_back(mPops[TYPE_LIF][popIndex]);
                    PopsByType[TYPE_LIF][PopsByType[TYPE_LIF].size()-1].mCount=pops[j].nNodeCount;
                    mPops[TYPE_LIF][popIndex].InsertPart(offset,pops[j].nNodeCount,i);
                    offset+=pops[j].nNodeCount;
                }
                else if(pops[j].nPopIndex>=mOffsets[TYPE_GEN])
                {
                    popIndex=pops[j].nPopIndex-mOffsets[TYPE_GEN];
                    PopsByType[TYPE_GEN].push_back(mPops[TYPE_GEN][popIndex]);
                    mPops[TYPE_GEN][popIndex].InsertPart(offset,pops[j].nNodeCount,i);
                    offset+=pops[j].nNodeCount;
                }
            }
            CopyNodesToSimulator(&mSimulators[i],PopsByType);
            //处理分部内部突触
            synCount=pg.GetInnerConn(i,builds[i],(SNum)mSynapses.size());
            if(!mSimulators[i].SetSynapse(synCount,builds[i]))
            {
                goto compile_end;
            }
            //处理外部突触
            oa.GetOrder(i,orders);
            for(j=0;j<(SNum)orders.size();j++)
            {
                int index=orders[j];
                if(i==index)
                    continue;
                synCount=pg.GetOutterConn(i,index,builds[i],(SNum)mSynapses.size());
                if(synCount<0)
                {
                    fprintf(stderr,"Wrong outter synapse data!\n");
                    goto compile_end;
                }
                if(!mSimulators[i].SetOutputSynapse(synCount,builds[i],&mSimulators[index]))
                {
                    fprintf(stderr,"outter[%d] synapse failed!\n",i);
                    goto compile_end;
                }
            }
            for(j=0;j<TYPE_COUNT;j++)
            {
                PopsByType[j].clear();
            }
            pops.clear();
        }
        for(i=0;i<nPart;i++)//逐个分部做最后的处理
        {
            mSimulators[i].UseMyGPU();
            if(!mSimulators[i].Prepare(mTimestep,(SNum)(mMinDelay/mTimestep),(SNum)(mMaxDelay/mTimestep),nBlockSize))
            {
                goto compile_end;
            }
        }
        bRet=true;
    }
    else//不需要划分
    {
        mSimulators=new Simulator[1]{meshSize};
        mSimulators->mGPUID=0;
        mSimulators->mMaxDelay=mMaxDelay;
        mSimulators->mMinDelay=mMinDelay;
        SYN_BUILD builds[mSynapses.size()];
        for(i=0;i<(SNum)mSynapses.size();i++)
        {
            builds[i].bOneToOne=mSynapses[i].bOneToOne;
            builds[i].weight=mSynapses[i].weight;
            builds[i].delay=mSynapses[i].delay;
            builds[i].fPropa=mSynapses[i].fProba;
            type=GETTYPE(mSynapses[i].preID);
            index=GETNO(mSynapses[i].preID);
            preIndex=mOffsets[type]+index;
            type=GETTYPE(mSynapses[i].postID);
            index=GETNO(mSynapses[i].postID);
            postIndex=mOffsets[type]+index;
            builds[i].preOffset=popOffsets[preIndex];
            builds[i].preCount=popOffsets[preIndex+1]-popOffsets[preIndex];
            builds[i].postOffset=popOffsets[postIndex];
            builds[i].postCount=popOffsets[postIndex+1]-popOffsets[postIndex];

            builds[i].preOffsetInWhole=0;
            builds[i].postOffsetInWhole=0;
            builds[i].postWholeCount=builds[i].postCount;
        }
        offset=0;
        for(i=0;i<TYPE_COUNT;i++)
        {
            for(j=0;j<(SNum)mPops[i].size();j++)
            {
                    mPops[i][j].InsertPart(offset,mPops[i][j].mCount,0);
                    offset+=mPops[i][j].mCount;
            }
        }
        CopyNodesToSimulator(mSimulators,mPops);
	    CUDACHECK(cudaGetLastError());
        if(!mSimulators[0].SetSynapse((SNum)mSynapses.size(),builds))
        {
            goto compile_end;
        }
	    CUDACHECK(cudaGetLastError());
        if(!mSimulators[0].Prepare(mTimestep,-1,-1,nBlockSize))
        {
            goto compile_end;
        }
	    CUDACHECK(cudaGetLastError());
        bRet=true;
    }
    mNeuronToSpikes=new std::map<SNum,std::vector<SFNum>> [nPart];
    mGenSpikeTrains=new std::map<SNum,std::pair<SNum,std::vector<SFNum>>>[nPart];
    mPartCount=nPart;
compile_end:
    if(!bRet)
    {
        clean();
    }
    if(pLIF)
        delete []pLIF;
    if(pLifArg)
        delete []pLifArg;
    return bRet;
}

 bool Network::Simulate(SFNum simulTime)
 {
#ifdef FAKE_MULTI_GPU
     SFNum now=0.0;
     while(now<simulTime)
     {
         for(SNum i=0;i<mPartCount;i++)
         {
            mSimulators[i].Simulate(false);
         }
         /*for(SNum i=0;i<mPartCount;i++)
         {
             mSimulators[i].TransferSpikes();
         }*/
         now+=mMinDelay;
     }
#else
    SFNum now[mPartCount]={0.0};
    printf("Now:");
    char szDigit[100]={0};
    int len=0;
    #pragma omp parallel num_threads(mPartCount)
    {
        SNum i=omp_get_thread_num();
        NSPIKE_GEN gen;
        GNGen ngen;
	    CUDACHECK(cudaGetLastError());
        mSimulators[i].UseMyGPU();
	    CUDACHECK(cudaGetLastError());
        mSimulators[i].SubmitSynapseChange();
	    CUDACHECK(cudaGetLastError());
        mSimulators[i].Reset();
	    CUDACHECK(cudaGetLastError());
        for(std::map<SNum,std::pair<SNum,std::vector<SFNum>>>::iterator it=mGenSpikeTrains[i].begin();
        it!=mGenSpikeTrains[i].end();it++)
        {
            it->second.first=0;
        }
        while(now[i]<simulTime)
        {
            if(i==(mPartCount-1))
            {
                for(int j=0;j<len;j++)
                printf("\b");
                sprintf(szDigit,"%f",now[i]);
                len=strlen(szDigit);
                printf("%s",szDigit);
            }
            for(std::map<SNum,std::pair<SNum,std::vector<SFNum>>>::iterator it=mGenSpikeTrains[i].begin();
            it!=mGenSpikeTrains[i].end();it++)
            {
                if(it->second.first< it->second.second.size() && \
                it->second.second[it->second.first]>now[i])
                {
                    ngen.pos=0;
                    ngen.length=0;
                    for(SNum j=0;j<MAX_SPIKE_COUNT && (it->second.first+j)<it->second.second.size();j++)
                    {
                        ngen.spikes[j]=it->second.second[it->second.first+j];
                        ngen.length++;
                    }
                    it->second.first+=ngen.length;
                    mSimulators[i].SetSpikeGenerator(it->first,ngen);
                }
            }
	        CUDACHECK(cudaGetLastError());
            mSimulators[i].Simulate(true);
            //if(i==0)
            //printf("End one timestep group\n");
            for(std::map<SNum,std::vector<SFNum>> ::iterator it=mNeuronToSpikes[i].begin();it!=mNeuronToSpikes[i].end();it++)
            {
                mSimulators[i].GetNeuronSpikes(it->first,&gen,true);
                for(SNum j=0;j<gen.length;j++)
                {
                    it->second.push_back(gen.spikes[j]);
                }
            }
            now[i]+=mMinDelay;
            //if(i==1)
            //printf("End spike collecting\n");
            #pragma omp barrier
            if(mPartCount>1)
            {
                mSimulators[i].TransferSpikes();
                #pragma omp barrier
                mSimulators[i].MergeSpikes();
            }
        }
    }
    printf("\nThe end!\n");
#endif
     return true;
 }

bool Network::LocateNeuron(SNum popID,SNum nIndex,std::pair<SNum,SNum> &ret)
{
    POP_IN_PART pip;
    SNum type,popIndex;
    type=GETTYPE(popID);
    popIndex=GETNO(popID);
    if(type<0 || type>=TYPE_COUNT)
        return false;
    if(popIndex<0 || popIndex>=(SNum)mPops[type].size())
        return false;
    if(!mPops[type][popIndex].GetPart(nIndex,pip))
        return false;
    ret.first=pip.part;
    ret.second=pip.offset+nIndex-pip.popOffset;
    return true;
}

bool Network::SetSpikeTrain(SNum genID,const std::vector<SFNum> &spikes)
{
    std::pair<SNum,SNum> neuron;
    std::pair<SNum,std::vector<SFNum>> PosAndSpikes;
    SNum type,popIndex;
    //GNGen gen;
    type=GETTYPE(genID);
    popIndex=GETNO(genID);
    if(type!=TYPE_GEN)
        return false;
    if(!LocateNeuron(genID,popIndex,neuron))
        return false;

    /*gen.length=spikes.size()<MAX_SPIKE_COUNT?(SNum)spikes.size():MAX_SPIKE_COUNT;
    gen.pos=0;
    for(SNum i=0;i<gen.length;i++)
    gen.spikes[i]=spikes[i];*/

    PosAndSpikes.first=0;
    PosAndSpikes.second=spikes;
    mGenSpikeTrains[neuron.first][neuron.second]=PosAndSpikes;

    return true;
    /*if(mPartCount>1)
        mSimulators[neuron.first].UseMyGPU();

    return mSimulators[neuron.first].SetSpikeGenerator(neuron.second,gen);*/
}

bool Network::WatchNeuron(SNum popID,SNum index)
{
    std::pair<SNum,SNum> neuron;
    if(!mNeuronToSpikes)
        return false;
    if(!LocateNeuron(popID,index,neuron))
        return false;
    mNeuronToSpikes[neuron.first][neuron.second].size();
    return true;
}

bool Network::GetNeuronSpikes(SNum popID,SNum index,std::vector<SFNum> &times)
{
    POP_IN_PART pip;
    SNum type,popIndex,partIndex;
    NSPIKE_GEN gen;
    type=GETTYPE(popID);
    popIndex=GETNO(popID);
    if(type<0 || type>=TYPE_COUNT)
        return false;
    if(popIndex<0 || popIndex>=(SNum)mPops[type].size())
        return false;
    if(!mPops[type][popIndex].GetPart(index,pip))
        return false;

    partIndex=pip.offset+index-pip.popOffset;
    if(mNeuronToSpikes[pip.part].count(partIndex))
    {
        times=mNeuronToSpikes[pip.part][partIndex];
        mNeuronToSpikes[pip.part][partIndex].clear();
        return true;
    }
    mSimulators[pip.part].UseMyGPU();
    mSimulators[pip.part].GetNeuronSpikes(partIndex,&gen,true);
    //printf("Get spike train from neuron %d of GPU%d\n",pip.offset+index-pip.popOffset,pip.part);
    for(SNum i=0;i<gen.length;i++)
    {
        times.push_back(gen.spikes[i]);
    }
    return true;
}

bool Network::Connect(SNum preID,SNum preIndex,SNum postID,SNum postIndex,SFNum weight,SFNum delay)//修改或增加指定突触
{
    std::pair<SNum,SNum> pre,post;
    if(!mSimulators)
        return false;

    if(!LocateNeuron(preID,preIndex,pre))
        return false;

    if(!LocateNeuron(postID,postIndex,post))
        return false;

    if(pre.first!=post.first)//目前只支持分部内的修改
        return false;

    if(!mSimulators[pre.first].SetSynapse(pre.second,post.second,weight,delay))
    {
        if(!mSimulators[pre.first].AddNewSynapse(pre.second,post.second,weight,delay))
            return false;
    }
    return true;
}

bool Network::Disconnect(SNum preID,SNum preIndex,SNum postID,SNum postIndex)//断开突触
{
    std::pair<SNum,SNum> pre,post;
    if(!mSimulators)
        return false;

    if(!LocateNeuron(preID,preIndex,pre))
        return false;

    if(!LocateNeuron(postID,postIndex,post))
        return false;

    if(pre.first!=post.first)//目前只支持分部内的修改
        return false;

    return mSimulators[pre.first].RemoveSynapse(pre.second,post.second);
}

bool Population::InsertPart(SNum offset,SNum count,SNum part)//按照顺序插入
{
    POP_IN_PART pp={0,offset,count,part};
    for(SNum i=1;i<(SNum)mParts.size();i++)
    {
        if(BEFORE_PART(mParts[i-1],pp) && BEFORE_PART(pp,mParts[i]))
        {
            mParts.insert(mParts.begin()+i,pp);
            for(i++;i<(SNum)mParts.size();i++)
            {
                mParts[i].popOffset+=count;
            }
            return true;
        }
    }
    if(mParts.size()>0 && BEFORE_PART(pp,mParts[0]))
    {
        mParts.insert(mParts.begin(),pp);
        for(SNum i=1;i<(SNum)mParts.size();i++)
        {
            mParts[i].popOffset+=count;
        }
    }
    else
    {
        if(mParts.size()>0)
        {
            pp.popOffset=mParts.rbegin()->popOffset+mParts.rbegin()->count;
        }
        mParts.push_back(pp);
    }
    return true;
}

bool Population::GetPart(SNum index,POP_IN_PART &pp)//根据节点在族群的下标找到对应的分布
{
    if(index<0)
        return false;
    for(SNum i=0;i<(SNum)mParts.size();i++)
    {
        if(mParts[i].count>index)
        {
            pp=mParts[i];
            return true;
        }
        index-=mParts[i].count;
    }
    return false;
}

}