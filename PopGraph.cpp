#include <stdio.h>
#include <string.h>
#include <math.h>

#include "util.h"
#include "PopGraph.h"


PopGraph::PopGraph(SNum nPop,SNum nPopCounts[])
    :mPopCount(nPop)
    ,mConnCountPerPop(NULL)
    ,mConnPerPop(NULL)
    ,mPartCount(0)
    ,mFirstPop(NULL)
    ,mNodeCount(0)
    ,mNodePerPop(NULL)
    ,mBestPop(NULL)
    ,mUpdated(true)
    ,mPopIDToArray(NULL)
{
    mPops=new SNum[nPop];
    for(SNum i=0;i<nPop;i++)
    {
        mPops[i]=nPopCounts[i];
        mNodeCount+=nPopCounts[i];
    }
    mNodePerPop=new PopNode*[nPop];
    memset(mNodePerPop,0,sizeof(PopNode *)*nPop);
    mLastBalance.Count=-1;
}

PopGraph::~PopGraph()
{
    delete []mPops;
    if(mConnCountPerPop)
        delete []mConnCountPerPop;
    if(mConnPerPop)
    {
        for(SNum i=0;i<mPopCount;i++)
        {
            if(mConnPerPop[i])
                delete []mConnPerPop[i];
        }
        delete []mConnPerPop;
    }
    if(mNodePerPop)
        delete []mNodePerPop;
    if(mFirstPop)
    {
        PopNode::DeleteWholeLink(mFirstPop);
    }
    if(mBestPop)
    {
        PopNode::DeleteWholeLink(mBestPop);
    }
    if(mPopIDToArray)
        delete []mPopIDToArray;
}

bool PopGraph::AddConn(SNum pop1,SNum pop2,bool bOneToOne,double fPropa,SFNum weight,SFNum delay)
{
    SNum t;
    if(mPartCount>0)//划分开始，就不能再添加新的边
        return false;
    if(pop1<0 || pop1>=mPopCount || pop2<0 || pop2>=mPopCount)
        return false;
    if(pop1==pop2 && bOneToOne)//对于自连接且一对一的连接，不予处理
        return false;
    if(bOneToOne && mPops[pop1]!=mPops[pop2])//一对一连接需要节点数相等
        return false;
    CONN_INFO conn={pop1,pop2,bOneToOne,fPropa,weight,delay};
//按照顺序插入
    t=-1;
    for(SNum i=0;i<mConns.size();i++)
    {
        if(conn.pop1<mConns[i].pop1 || (conn.pop1==mConns[i].pop1 && conn.pop2<mConns[i].pop2))
        {
            t=i;
            break;
        }
    }
    if(t>=0)
        mConns.insert(mConns.begin()+t,conn);
    else
        mConns.push_back(conn);
    return true;
}

bool PopGraph::StartPartition(SNum nPart,bool bAverage)
{
    PopNode *pNode,*pLast=NULL;
    SNum i,j,p,q,n;
    SNum nConn;
    std::pair<SNum,SNum> edge;
    if(nPart<=0)
        return false;
    SNum curPop=0;//当前所划分到的族群
    SNum nodePerPart=(mNodeCount+nPart-1)/nPart;//每个部分初始的节点数
    SNum progPerPop[mPopCount]={0};//每个族群的分配连接信息的进度
    SNum countPerPop[mPopCount]={0};//每个族群剩余的节点数
    memcpy(countPerPop,mPops,sizeof(SNum)*mPopCount);
    mCarveCount=0;
    mMoveTime=0;
    mConnCountPerPop=new SNum[mPopCount];
    mConnPerPop=new CONN_INFO*[mPopCount];
    mCountPerPart=new SNum[nPart];
    memset(mCountPerPart,0,sizeof(SNum)*nPart);
    //统计各个节点的边数
    memset(mConnCountPerPop,0,sizeof(SNum)*mPopCount);
    SNum nEdge=(SNum)mConns.size();
    mPartCount=nPart;
    for(i=0;i<nEdge;i++)
    {
        mConnCountPerPop[mConns[i].pop1]++;
        mConnCountPerPop[mConns[i].pop2]++;
    }
    //将边的信息分配到各个顶点对应的数组中
    for(i=0;i<mPopCount;i++)
    {
        nConn=mConnCountPerPop[i];
        mConnPerPop[i]=new CONN_INFO[nConn];
    }
    for(i=0;i<nEdge;i++)
    {
        edge.first=mConns[i].pop1;
        edge.second=mConns[i].pop2;
        mNeighbor[edge].push_back(i);
        p=mConns[i].pop1;
        q=progPerPop[p]++;
        mConnPerPop[p][q]=mConns[i];

        edge.first=mConns[i].pop2;
        edge.second=mConns[i].pop1;
        mNeighbor[edge].push_back(i);
        p=mConns[i].pop2;
        q=progPerPop[p]++;
        mConnPerPop[p][q]=mConns[i];
    }
    //初始化划分链表并开始初步划分
    for(i=0;i<mPopCount && bAverage;i++)//逐个总族群进行平均分
    {
        n=countPerPop[i]/nPart;
        for(j=0;j<nPart;j++)
        {
            if(j==(nPart-1))
            {
                if(countPerPop[i]<=0)
                    continue;
                pNode=new PopNode(this,i,countPerPop[i],j);
                if(!mNodePerPop[i])
                    mNodePerPop[i]=pNode;
                mCountPerPart[j]+=countPerPop[i];
                countPerPop[i]=0;
            }
            else
            {
                if(n<=0)
                    continue;
                pNode=new PopNode(this,i,n,j);
                if(!mNodePerPop[i])
                    mNodePerPop[i]=pNode;
                countPerPop[i]-=n;
                mCountPerPart[j]+=n;
            }
            if(pLast)
            {
                pLast->Insert(pNode);
                pLast=pNode;
            }
            else
            {
                mFirstPop=pLast=pNode;
            }
            
        }
    }
    for(i=0;i<nPart && !bAverage;i++)//逐个部分进行划分
    {
        while(curPop<mPopCount && mCountPerPart[i]<nodePerPart)
        {
            if((countPerPop[curPop]+mCountPerPart[i])<=nodePerPart)//当前部分还够分，则将整个族群直接放入
            {
                pNode=new PopNode(this,curPop,countPerPop[curPop],i);
                if(!mNodePerPop[curPop])
                    mNodePerPop[curPop]=pNode;
                mCountPerPart[i]+=countPerPop[curPop];
                countPerPop[curPop]=0;
                curPop++;
            }
            else//如果不够分了，就拆分后再分
            {
                n=nodePerPart-mCountPerPart[i];
                pNode=new PopNode(this,curPop,n,i);
                if(!mNodePerPop[curPop])
                    mNodePerPop[curPop]=pNode;
                countPerPop[curPop]-=n;
                mCountPerPart[i]+=n;
            }
            if(pLast)
            {
                pLast->Insert(pNode);
                pLast=pNode;
            }
            else
            {
                mFirstPop=pLast=pNode;
            }
            
        }
    }
    UpdateGains();
    mBestPop=mFirstPop->CloneList();
    mMinCarveCount=mCarveCount;
    if(mPopIDToArray)
        delete []mPopIDToArray;
    mPopIDToArray=new NodeMap[mPartCount];
    return true;
}

bool PopGraph::UpdateGains()
{
    if(!mPartCount)
        return false;
    PopNode *pNode=mFirstPop;
    mCarveCount=0.0;
    mInternalNode.clear();
    do
    {
        pNode->UpdateGains();
        if(pNode->GetCarveCount()<=0.0)
            mInternalNode.push_back(pNode);
        mCarveCount+=pNode->GetCarveCount();
        pNode=pNode->Next();
    } while (pNode && pNode!=mFirstPop);
    mUpdated=true;
    return true;
}

SNum PopGraph::MovePop(PopGraph::PopNode *pNode,SNum ncount,SNum dstPart)
{
    SNum srcPart=pNode->GetPart();
    PopNode *pSrc=pNode;
    if(ncount<=0)
        ncount=pNode->GetCount();
    pNode=pNode->MoveTo(dstPart,ncount);
    mCountPerPart[srcPart]-=ncount;
    mCountPerPart[dstPart]+=ncount;
    if(pNode!=pSrc)
    {
        if(mNodePerPop[pNode->GetPopID()]==pSrc)
            mNodePerPop[pNode->GetPopID()]=pNode;
        if(pSrc==mFirstPop)
            mFirstPop=pNode;
        delete pSrc;
    }
    mUpdated=true;
    return ncount;
}

bool PopGraph::GreedyMove()
{
    SNum ncount;
    SFNum fMaxGain=0.0,fGain;
    SNum nMinNode=mNodeCount;
    PopGain *pMaxGain=NULL,*pGain;
    if(!mPartCount)
        return false;
    PopNode *pNode=mFirstPop;
    //先找出最大收益
    do
    {
        pGain=pNode->GetMaxGain(-1,true);
        if(pGain && (!pGain->EqualRecord(mLastBalance) || pGain->mParent->GetPart()!=mLastBalance.DstPart))
        {
            ncount=pGain->mParent->GetCount();
            fGain=pGain->GetGain(-1);
            if(!pMaxGain || (fGain>fMaxGain || (fGain==fMaxGain && ncount<nMinNode)))
            {
                pMaxGain=pGain;
                fMaxGain=fGain;
                nMinNode=ncount;
            }
        }
        pNode=pNode->Next();
    } while (pNode && pNode!=mFirstPop);
    if(!pMaxGain)//未找到，说明收敛了
        return false;
    MovePop(pMaxGain->mParent,-1,pMaxGain->mPart);
    mMoveTime++;
    UpdateGains();
    
    return true;
}

SNum PopGraph::MaximizeMove(SNum ncount,SNum tolerance,SNum srcPart,SNum dstPart)
{
    CONN_INFO conn;
    SNum moved=0,n,m,i;
    SNum n121;
    SNum bound=ncount-tolerance/2;
    SNum pop;
    std::pair<SNum,SNum> edge;
    SFNum fMaxGain=0.0,fGain,f2MaxGain;
    PopGain *pMaxGain=NULL,*pGain,*p2MaxGain;
    PopNode *pNode,*pMaxNode,*p2MaxNode;
    while(moved<ncount)
    {
        n=ncount-moved;
        pNode=mFirstPop;
        pMaxGain=NULL;
        fMaxGain=0.0;
        do
        {
            if(pNode->GetPart()!=srcPart)// || ((mMoveTime-pNode->GetLastMove())<=mPartCount && pNode->GetLastMove()>0))
            {
                pNode=pNode->Next();
                continue;
            }
            pGain=pNode->GetPartGain(dstPart);//pNode->GetMaxGain(n,false);
            if(pGain)
            {
                fGain=pGain->GetGain(1);
                if((moved<bound || fGain>0.0) && (!pMaxGain || fGain>fMaxGain))
                {
                    pMaxGain=pGain;
                    fMaxGain=fGain;
                }
            }
            pNode=pNode->Next();
        } while (pNode && pNode!=mFirstPop);
        if(!pMaxGain)
            break;
        pMaxNode=pMaxGain->mParent;
        //如果要移动的节点数大于1，则需要确定合适移动的节点数
        //寻找与当前最大收益子族群邻接且在同一分部的最大收益子族群
        p2MaxGain=NULL;
        f2MaxGain=0.0;
        for(pNode=mFirstPop,i=0;pNode && (pNode!=mFirstPop || i<=0) && n>1;pNode=pNode->Next(),i++)
        {
            if(pNode->GetPart()!=srcPart || pNode==pMaxGain->mParent)
                continue;
            edge.first=pMaxGain->mParent->GetPopID();
            edge.second=pNode->GetPopID();
            if(!mNeighbor.count(edge))
                continue;
            pGain=pNode->GetPartGain(dstPart);//pNode->GetMaxGain(n,false);
            if(pGain)
            {
                fGain=pGain->GetGain(1);
                if((moved<bound || fGain>0.0) && (!p2MaxGain || fGain>f2MaxGain))
                {
                    p2MaxNode=pGain->mParent;
                    p2MaxGain=pGain;
                    f2MaxGain=fGain;
                }
            }
        }
        if(p2MaxGain)//如果找到合适的最大相邻子族群，需要计算出当前子族群移动到目标分部的合适节点数目
        {
            if(fMaxGain==f2MaxGain)//如果发现双方收益一样，则双方各移动一半的节点
            {
                m=n/2;
                if(pMaxNode->GetCount()<m)
                    m=pMaxNode->GetCount();
                if(p2MaxNode->GetCount()<m)
                    m=p2MaxNode->GetCount();
                
                mLastBalance.Pop=p2MaxNode->GetPopID();
                mLastBalance.Count=m;
                mLastBalance.SrcPart=p2MaxNode->GetPart();
                mLastBalance.DstPart=dstPart;
                MovePop(pMaxNode,m,dstPart);
                UpdateGains();
                MovePop(p2MaxNode,m,dstPart);
                UpdateGains();
                moved+=2*m;
                continue;
            }
            edge.first=pMaxNode->GetPopID();
            edge.second=p2MaxNode->GetPopID();
            fGain=0.0;
            n121=0;
            for(SNum i=0;i<(SNum)mNeighbor[edge].size();i++)
            {
                conn=mConns[mNeighbor[edge][i]];
                if(conn.bOneToOne)
                {
                    if(pMaxNode->GetCount()>p2MaxNode->GetCount())
                    {
                        //对于一对一的连接，如果要移动的子族群节点数大于最大相邻子族群
                        //则移动后不会影响最大相邻子族群的收益
                        //此时需要将最大移动节点数限制在两者的差值，否则一旦超出会导致计算错误
                        n121=pMaxNode->GetCount()-p2MaxNode->GetCount();
                    }
                    else
                    {
                        fGain+=1.0;
                    }
                }
                else
                {
                    fGain+=conn.fPropa;
                }
            }
            if(fGain>0.0)
            {
                m=(SNum)((fMaxGain-f2MaxGain)/fGain);
                if(m>n121 && n121>0)
                    m=n121;
                if(m<n)
                    n=m;
                if(n<=0)
                    n=1;
            }
            

        }//如果找不到合适的最大相邻子族群，就直接将当前子族群尽数移动到目标分部
        if(n>pMaxGain->mParent->GetCount())
            n=pMaxGain->mParent->GetCount();
        mLastBalance.Pop=pMaxGain->mParent->GetPopID();
        mLastBalance.Count=n;
        mLastBalance.SrcPart=pMaxGain->mParent->GetPart();
        mLastBalance.DstPart=dstPart;
        
        MovePop(pMaxGain->mParent,n,dstPart);
        UpdateGains();
        moved+=n;
    }
    mMoveTime++;
    return moved;
}

SNum PopGraph::Turbulence(SNum ncount,SNum srcPart,SNum dstPart)
{
    PopNode *pNode;
    SNum i,len,r,delta;
    SNum n=0,ret=0;
    SNum *nodeCounts;//各个族群要转移的节点数
    std::vector<PopNode *> IntNodes;//属于srcPart的内部族群
    //先筛选出属于srcPart的内部族群，并统计可扰动的节点数
    len=(SNum)mInternalNode.size();
    for(i=0;i<len;i++)
    {
        pNode=mInternalNode[i];
        if(pNode->GetPart()==srcPart)
        {
            IntNodes.push_back(pNode);
            n+=pNode->GetCount();
        }
    }
    len=(SNum)IntNodes.size();
    if(n<ncount)//不够，则直接全体挪到对面
    {
        for(i=0;i<len;i++)
        {
            pNode=IntNodes[i];
            MovePop(pNode,-1,dstPart);
        }
        UpdateGains();
        return n;
    }
    nodeCounts=new SNum[len];
    for(i=0;i<len;i++)
    {
        nodeCounts[i]=IntNodes[i]->GetCount()*ncount/n;
        delta=IntNodes[i]->GetCount()-nodeCounts[i];
        r=(rand()%delta*2)-delta;
        nodeCounts[i]+=r;
        if(nodeCounts[i]<0)
            nodeCounts[i]=0;
    }
    for(i=0;i<len;i++)
    {
        ret+=MovePop(IntNodes[i],nodeCounts[i],dstPart);
    }
    delete []nodeCounts;
    UpdateGains();
    mMoveTime++;
    return ret;
}

void PopGraph::Partition(SNum turbCount,SFNum turbDecay)
{
    SNum maxi,mini,maxn,minn,n,i,delta,p,q;
    SNum tolerance;
    SNum NonImprove=0;//在真正优化之前完成一次优化动作的次数
    SNum minCount=mMinCarveCount;
    if(!mPartCount)
        return;
    tolerance=mNodeCount/(mPartCount*10);
    printf("first partition\n");
    PrintPartition();
    do
    {
        while(NonImprove<mNodeCount)
        {
            //贪心移动
            if(!GreedyMove())
                break;
            //printf("greedy move\n");
            //PrintPartition();
            //求最大部分和最小部分
            while(true)
            {
                maxi=0;
                mini=0;
                minn=maxn=mCountPerPart[0];
                for(i=1;i<mPartCount;i++)
                {
                    n=mCountPerPart[i];
                    if(n>maxn)
                    {
                        maxi=i;
                        maxn=n;
                    }
                    if(n<minn)
                    {
                        mini=i;
                        minn=n;
                    }
                }
                //补偿移动
                if(mini==maxi || (maxn-minn)<=tolerance)
                    break;
                delta=maxn-minn;
                MaximizeMove(delta/2,tolerance,maxi,mini);
            }
            //printf("balance move\n");
            //PrintPartition();
            NonImprove++;
            if(mCarveCount<minCount)
            {
                minCount=mCarveCount;
                NonImprove=0;
            }
        }
        if(mCarveCount<mMinCarveCount)
        {
            if(mBestPop)
                PopNode::DeleteWholeLink(mBestPop);
            mBestPop=mFirstPop->CloneList();
            mMinCarveCount=mCarveCount;
        }
        
        if(turbCount>0)
        {
            p=rand()%mPartCount;
            while((q=(rand()%mPartCount))==p);
            if(Turbulence(turbCount,p,q)<=0)
                break;
            turbCount=(SNum)((SFNum)turbCount*turbDecay);
            //printf("After turbulence:\n");
            //PrintPartition();
        }
    }while(turbCount>0);
    printf("Final partition:\n");
    PrintPartition(true);
}

SNum PopGraph::GetPartCount()
{
    return mPartCount;
}

SNum PopGraph::GetPartNodeCount(SNum index)
{
    if(index<0 || index>=mPartCount)
        return 0;
    return mCountPerPart[index];
}

void PopGraph::PrintPartition(bool bPrintBest)
{
    PopNode *pNode=bPrintBest?mBestPop:mFirstPop;
    PopNode *pSrc=pNode;
    if(!pNode)
        return;
    do
    {
        pNode->PrintToScreen();
        pNode=pNode->Next();
    } while (pNode && pNode!=pSrc);
    if(bPrintBest)
        printf("carve edge count:%.1f\n",mMinCarveCount);
    else
        printf("carve edge count:%.1f\n",mCarveCount);
}


SNum PopGraph::GetPopCountInPart(SNum pop,SNum part)
{
    if(!mPopIDToArray[part].count(pop))
        return 0;
    return mPopIDToArray[part][pop].second;
}

void PopGraph::UpdatePopNode()
{
    if(!mPopIDToArray || !mBestPop)
        return;
    SNum NodeCount;
    std::vector<PopNode *> PopPerPart[mPartCount];
    PopNode *pNode=mBestPop;
    PopNode *pSrc=pNode;
    SNum part,ncount;
    SNum dstOffsets[mPartCount]={0};
    SNum srcOffset;
    SNum dstPop1,dstPop2,srcPop1,srcPop2;
    SNum buildCount;
    std::pair<SNum,SNum> pair;
    //将各个分部的子族群按分部存放
    do
    {
        part=pNode->GetPart();
        PopPerPart[part].push_back(pNode);
        pNode=pNode->Next();
    } while (pNode && pNode!=pSrc);
    //将各个分部的子族群排序，然后开始分配节点空间
    for(part=0;part<mPartCount;part++)
    {
        for(SNum i=(SNum)PopPerPart[part].size()-1;i>0;i--)
        {
            for(SNum j=0;j<i;j++)
            {
                if(PopPerPart[part][j]->GetPopID()>PopPerPart[part][j+1]->GetPopID())
                {
                    pNode=PopPerPart[part][j];
                    PopPerPart[part][j]=PopPerPart[part][j+1];
                    PopPerPart[part][j+1]=pNode;
                }
            }
        }
        NodeCount=0;
        for(SNum i=0;i<PopPerPart[part].size();i++)
        {
            pNode=PopPerPart[part][i];
            pair.first=NodeCount;
            pair.second=pNode->GetCount();
            mPopIDToArray[part][pNode->GetPopID()]=pair;
            NodeCount+=pNode->GetCount();
        }
    }
    //计算一对一连接的外部突触分配
    for(SNum i=0;i<mConns.size();i++)
    {
        if(!mConns[i].bOneToOne)
            continue;
        std::map<std::pair<SNum,SNum>,SYN_BUILD> results;
        std::pair<SNum,SNum> p2p;//分部到分部
        SYN_BUILD sb;
        results.clear();
        for(SNum j=0;j<mPartCount;j++)//先计算目标族群在每个分部中可以内部消化的部分，作为构建外部连接的起始
        {
            dstOffsets[j]=0;
            if(!mPopIDToArray[j].count(mConns[i].pop1) || !mPopIDToArray[j].count(mConns[i].pop2))
                continue;
            if(mPopIDToArray[j][mConns[i].pop1].second<mPopIDToArray[j][mConns[i].pop2].second)
                dstOffsets[j]=mPopIDToArray[j][mConns[i].pop1].second;
        }
        for(p2p.first=0;p2p.first<mPartCount;p2p.first++)
        {
            srcPop1=GetPopCountInPart(mConns[i].pop1,p2p.first);
            if(srcPop1<=0)
                continue;
            srcPop2=GetPopCountInPart(mConns[i].pop2,p2p.first);
            if(srcPop1<=srcPop2)//足够内部消化，不需要构建外部连接
                continue;
            srcOffset=srcPop2;
            for(p2p.second=0;p2p.second<mPartCount;p2p.second++)
            {
                if(p2p.first==p2p.second)
                    continue;
                dstPop2=GetPopCountInPart(mConns[i].pop2,p2p.second);
                if(dstPop2<=0)
                    continue;
                if(dstOffsets[p2p.second]>=dstPop2)//已经消化完毕，不需要构建外部连接
                    continue;
                buildCount=srcPop1-srcOffset;
                if(buildCount>(dstPop2-dstOffsets[p2p.second]))
                    buildCount=dstPop2-dstOffsets[p2p.second];
                if(buildCount<=0)
                    continue;
                sb.bOneToOne=true;
                sb.postCount=sb.preCount=buildCount;
                sb.preOffset=mPopIDToArray[p2p.first][mConns[i].pop1].first+srcOffset;
                sb.postOffset=mPopIDToArray[p2p.second][mConns[i].pop2].first+dstOffsets[p2p.second];
                srcOffset+=buildCount;
                dstOffsets[p2p.second]+=buildCount;
                results[p2p]=sb;
            }
        }
        m121ConnBuilds[i]=results;
    }
    
    mUpdated=false;
}

SNum PopGraph::GetSubConnOffset(SNum connIdx,SNum part1,SNum part2)
{
    if(connIdx<0 || connIdx>=(SNum)mConns.size())
        return 0;
    SNum ret=0;
    CONN_INFO conn=mConns[connIdx];
    //先统计在之前分部的边数
    for(SNum i=0;i<part1;i++)
    {
        if(!mPopIDToArray[i].count(conn.pop1))
            continue;
        ret+=mPopIDToArray[i][conn.pop1].second*mPops[conn.pop2];
    }
    //再统计以当前的前节点分部到后节点分部的边数
    for(SNum i=0;i<part2;i++)
    {
        if(!mPopIDToArray[part1].count(conn.pop1) || !mPopIDToArray[i].count(conn.pop2))
            continue;
        ret+=mPopIDToArray[part1][conn.pop1].second*mPopIDToArray[i][conn.pop2].second;
    }
    return ret;
}

SNum PopGraph::GetSubPopOffset(SNum part,SNum pop)
{
    SNum ret=0;
    //统计在之前分部的神经元数
    for(SNum i=0;i<part;i++)
    {
        if(!mPopIDToArray[i].count(pop))
            continue;
        ret+=mPopIDToArray[i][pop].second;
    }
    return ret;
}

SNum PopGraph::GetInnerConn(SNum nPart,SYN_BUILD *builds,SNum nBuild)
{
    SNum ncount=0;
    SNum pop1,pop2;
    if(mUpdated)
        UpdatePopNode();
    
    for(SNum i=0;i<(SNum)mConns.size() && (ncount<nBuild || nBuild<=0);i++)
    {
        pop1=mConns[i].pop1;
        pop2=mConns[i].pop2;
        if(!mPopIDToArray[nPart].count(pop1) || !mPopIDToArray[nPart].count(pop2))
            continue;
        if(ncount<nBuild && builds)
        {
            builds[ncount].preOffset=mPopIDToArray[nPart][pop1].first;
            builds[ncount].preCount=mPopIDToArray[nPart][pop1].second;
            builds[ncount].postOffset=mPopIDToArray[nPart][pop2].first;
            builds[ncount].postCount=mPopIDToArray[nPart][pop2].second;
            if(builds[ncount].preCount<=0 || builds[ncount].postCount<=0)
                continue;
            if(mConns[i].bOneToOne)
            {
                if(builds[ncount].preCount>builds[ncount].postCount)
                {
                    builds[ncount].preCount=builds[ncount].postCount;
                }
                else
                {
                    builds[ncount].postCount=builds[ncount].preCount;
                }
                
            }
            builds[ncount].bOneToOne=mConns[i].bOneToOne;
            builds[ncount].fPropa=mConns[i].fPropa;
            builds[ncount].weight=mConns[i].fWeight;
            builds[ncount].delay=mConns[i].fDelay;
            builds[ncount].preOffsetInWhole=GetSubPopOffset(nPart,pop1);
            builds[ncount].postOffsetInWhole=GetSubPopOffset(nPart,pop2);
            builds[ncount].postWholeCount=mPops[pop2];
            ncount++;
        }
    }
    return ncount;
}

SNum PopGraph::GetOutterConn(SNum nSrcPart,SNum nDstPart,SYN_BUILD *builds,SNum nBuild)
{
    SNum ncount=0;
    SNum pop1,pop2;
    std::pair<SNum,SNum> p2p;
    if(mUpdated)
        UpdatePopNode();
    
    for(SNum i=0;i<(SNum)mConns.size() && (ncount<nBuild || nBuild<=0);i++)
    {
        pop1=mConns[i].pop1;
        pop2=mConns[i].pop2;
        if(!mPopIDToArray[nSrcPart].count(pop1) || !mPopIDToArray[nDstPart].count(pop2))
            continue;
        if(ncount<nBuild && builds)
        {
            if(mConns[i].bOneToOne)
            {
                p2p.first=nSrcPart;
                p2p.second=nDstPart;
                if(!m121ConnBuilds.count(i) || !m121ConnBuilds[i].count(p2p))
                    continue;
                builds[ncount]=m121ConnBuilds[i][p2p];
            }
            else
            {
                builds[ncount].preOffset=mPopIDToArray[nSrcPart][pop1].first;
                builds[ncount].preCount=mPopIDToArray[nSrcPart][pop1].second;
                builds[ncount].postOffset=mPopIDToArray[nDstPart][pop2].first;
                builds[ncount].postCount=mPopIDToArray[nDstPart][pop2].second;
            }
            builds[ncount].bOneToOne=mConns[i].bOneToOne;
            builds[ncount].fPropa=mConns[i].fPropa;
            builds[ncount].weight=mConns[i].fWeight;
            builds[ncount].delay=mConns[i].fDelay;
            builds[ncount].preOffsetInWhole=GetSubPopOffset(nSrcPart,pop1);
            builds[ncount].postOffsetInWhole=GetSubPopOffset(nDstPart,pop2);
            builds[ncount].postWholeCount=mPops[pop2];
            ncount++;
        }
    }
    return ncount;
}

bool PopGraph::GetPopsInPart(SNum nPart,std::vector<POP_INFO> &Pops)
{
    if(mUpdated)
        UpdatePopNode();
    if(!mPopIDToArray || !mBestPop)
        return false;
    PopNode *pNode=mBestPop;
    PopNode *pSrc=pNode;
    POP_INFO pi;
    do
    {
        if(pNode->GetPart()==nPart)
        {
            pi.nNodeCount=pNode->GetCount();
            pi.nPopIndex=pNode->GetPopID();
            Pops.push_back(pi);
        }
        pNode=pNode->Next();
    } while (pNode && pNode!=pSrc);
    
    return true;
}

PopGraph::PopNode::PopNode(PopGraph *parent,SNum popID,SNum popN,SNum part)
    :mParent(parent)
    ,mPopID(popID)
    ,mCount(popN)
    ,mConnCount(parent->mConnCountPerPop[popID])
    ,mConns(parent->mConnPerPop[popID])
    ,mPartCount(parent->mPartCount)
    ,mPart(part)
    ,mGains(NULL)
    ,mCarveCount(0)
    ,mLastMove(parent->mMoveTime)
{
    mpPrev=this;
    mpNext=this;
    mpPopPrev=this;
    mpPopNext=this;
}

PopGraph::PopNode::~PopNode()
{
    if(mGains)
        delete []mGains;
}

void PopGraph::PopNode::Insert(PopNode *pNode)
{
    if(!pNode)
        return;
    if(mpNext)
    {
        mpNext->mpPrev=pNode;
        pNode->mpNext=mpNext;
    }
    mpNext=pNode;
    pNode->mpPrev=this;
    if(mPopID==pNode->mPopID)
    {
        if(mpPopNext)
        {
            mpPopNext->mpPopPrev=pNode;
            pNode->mpPopNext=mpPopNext;
        }
        mpPopNext=pNode;
        pNode->mpPopPrev=this;
    }
}

void PopGraph::PopNode::Delete()
{
    if(mpPrev && mpPrev!=this)
        mpPrev->mpNext=mpNext;
    if(mpNext && mpPrev!=this)
        mpNext->mpPrev=mpPrev;
    mpPrev=this;
    mpNext=this;

    if(mpPopPrev && mpPopPrev!=this)
        mpPopPrev->mpPopNext=mpPopNext;
    if(mpPopNext && mpPopNext!=this)
        mpPopNext->mpPopPrev=mpPopPrev;
    mpPopPrev=this;
    mpPopNext=this;
}

PopGraph::PopNode * PopGraph::PopNode::MoveTo(SNum part,SNum ncount)
{
    if(ncount>mCount)
        return this;
    if(mPart==part)//如果目标部分与当前部分一样，则不需要分裂
        return this;
    PopNode *pNode;
    mLastMove=mParent->mMoveTime;
    if(ncount<=0 || ncount==mCount)//整体搬迁
    {
        pNode=mpPopNext;
        //先寻找当前族群所在总族群中是否有其他族群在目标部分中，如果有，就合并
        while(pNode && pNode!=this)//往后面找
        {
            if(pNode->mPart==part)
                break;
            pNode=pNode->mpPopNext;
        }
        if(pNode && pNode!=this)//如果找到了，就直接合并
        {
            pNode->mCount+=mCount;
            pNode->mLastMove=mParent->mMoveTime;
            Delete();
            return pNode;
        }
        //没找到，就直接挪过去
        pNode->mPart=part;
    }
    else//切割后分出一部分到目标部分
    {
        pNode=mpPopNext;
        //先寻找当前族群所在总族群中是否有其他族群在目标部分中，如果有，就合并
        while(pNode && pNode!=this)//往后面找
        {
            if(pNode->mPart==part)
                break;
            pNode=pNode->mpPopNext;
        }
        if(pNode && pNode!=this)//如果找到了，就直接合并
        {
            pNode->mCount+=ncount;
            pNode->mLastMove=mParent->mMoveTime;
        }
        else//没找到，就新建一个子族群插入
        {
            pNode=new PopNode(mParent,mPopID,ncount,part);
            Insert(pNode);
        }
        mCount-=ncount;
    }
    return this;
}

void PopGraph::PopNode::UpdateGains()
{
    PopNode *pNode,*pDstNode;
    SFNum internal=0.0,selfPropa=0.0;
    SNum dstPop;
    SNum nDelta[mPartCount];
    mCarveCount=0.0;
    if(!mGains)
    {
        mGains=new PopGain[mPartCount];
    }
    for(SNum i=0;i<mPartCount;i++)
    {
        mGains[i].Reset();
        mGains[i].mParent=this;
        mGains[i].mPart=i;
        mGains[i].mSelfCount=(SFNum)mCount;
        nDelta[i]=0;
    }
    //将本族群所在总族群其他位置的1对1时的节点数量加到对应部分数量差上
    pNode=this;
    do
    {
        nDelta[pNode->mPart]+=pNode->mCount;
        pNode=pNode->mpPopNext;
    } while (pNode && pNode!=this);
    
    for(SNum i=0;i<mConnCount;i++)
    {
        if(mConns[i].pop1==mPopID)
            dstPop=mConns[i].pop2;
        else
            dstPop=mConns[i].pop1;
        pDstNode=mParent->mNodePerPop[dstPop];
        
        if(mConns[i].bOneToOne)//将目标族群对应部分的数量差减去目标族群节点数，以求得最终的数量差
        {
            mGains[pNode->mPart].mOneToOne=true;
            pNode=pDstNode;
            do
            {
                nDelta[pNode->mPart]-=pNode->mCount;
                pNode=pNode->mpPopNext;
            } while (pNode!=pDstNode);
            //统计割边数
            pNode=pDstNode;
            do
            {
                if(pNode->mPart!=mPart)
                    mCarveCount+=(SFNum)pNode->mCount;
                pNode=pNode->mpPopNext;
            } while (pNode!=pDstNode);
            
            
        }
        else
        {
            pNode=pDstNode;
            do
            {
                if(pNode->mPart==mPart)
                {
                    if(pNode->mPopID==mPopID)
                    {
                        selfPropa+=mConns[i].fPropa;
                    }
                    else
                    {
                        internal+=(SFNum)pNode->mCount*mConns[i].fPropa;
                    }
                    
                }
                else
                {
                    mGains[pNode->mPart].mExternal+=(SFNum)pNode->mCount*mConns[i].fPropa;
                }
                
                if(pNode->mPart!=mPart)
                    mCarveCount+=(SFNum)((SFNum)pNode->mCount*(SFNum)mCount)*mConns[i].fPropa;
                pNode=pNode->mpPopNext;
            } while (pNode!=pDstNode);
        }
        
    }
    for(SNum i=0;i<mPartCount;i++)
    {
        mGains[i].mInternal=internal;
        mGains[i].mSelfPropa=selfPropa;
    }
}

void PopGraph::PopNode::DeleteWholeLink(PopNode *pNode)
{
    PopNode *pPrev=NULL,*pFirst=pNode;
    if(!pNode)
        return;
    pPrev=pNode;
    pNode=pNode->mpNext;
    while(pNode && pNode!=pFirst)
    {
        delete pPrev;
        pPrev=pNode;
        pNode=pNode->mpNext;
    }
}

PopGraph::PopNode *PopGraph::PopNode::Next()
{
    return mpNext;
}

SNum PopGraph::PopNode::GetCount()
{
    return mCount;
}

SNum PopGraph::PopNode::GetPart()
{
    return mPart;
}

SNum PopGraph::PopNode::GetPopID()
{
    return mPopID;
}

SFNum PopGraph::PopNode::GetCarveCount()
{
    return mCarveCount;
}

SNum PopGraph::PopNode::GetLastMove()
{
    return mLastMove;
}

PopGraph::PopGain *PopGraph::PopNode::GetPartGain(SNum part)
{
    if(part<0 || part>=mPartCount)
        return NULL;
    return &mGains[part];
}

PopGraph::PopGain *PopGraph::PopNode::GetMaxGain(SNum ncount,bool bPositive)
{
    if(ncount<0)
        ncount=mCount;
    PopGraph::PopGain *pMax=NULL;
    SFNum fMaxGain=0.0,fGain;
    for(SNum i=0;i<mPartCount;i++)
    {
        if(i==mPart)
            continue;
        fGain=mGains[i].GetGain(ncount);
        if((!pMax && !bPositive) || fGain>fMaxGain)
        {
            fMaxGain=fGain;
            pMax=&mGains[i];
        }
    }
    return pMax;
}

void PopGraph::PopNode::PrintToScreen()
{
    printf("Pop %d,node %d,part %d\n",mPopID,mCount,mPart);
}

PopGraph::PopNode *PopGraph::PopNode::CloneList()
{
    PopNode *pNode=this,*pNew,*pFirst=NULL,*pOld=NULL;
    //从所属总族群的第一个节点开始克隆
    while(pNode->mpPrev->mPopID==mPopID && pNode->mpPrev!=this)
    {
        pNode=pNode->mpPrev;
    }
    if(pNode!=this)
    {
        return pNode->CloneList();
    }
    do
    {
        pNew=new PopNode(mParent,pNode->mPopID,pNode->mCount,pNode->mPart);
        if(pOld)
        {
            pOld->Insert(pNew);
            pOld=pNew;
        }
        else
        {
            pOld=pNew;
            pFirst=pNew;
        }
        
        pNode=pNode->mpNext;
    } while (pNode && pNode!=this);
    
    return pFirst;
}

SFNum PopGraph::PopGain::Get121Gain(SFNum srcDelta,SFNum dstDelta)
{
    if(srcDelta*dstDelta>=0.0)//如果两边的差值同号，代表两边不需要有边连接
        return 0.0;
    srcDelta=abs(srcDelta);
    dstDelta=abs(dstDelta);
    return srcDelta<dstDelta?srcDelta:dstDelta;
}

void PopGraph::PopGain::Reset()
{
    mExternal=0.0;//族群内每个节点到目标部分节点的边数
    mInternal=0.0;//族群内每个节点到所在部分其他族群节点的边数
    mSelfPropa=0.0;//族群内每个节点和其他节点的边数比率
    mSelfCount=0.0;//族群节点数
    mOneToOne=false;//是否有一对一的边
    mSrcDelta=0.0;//族群所在部分与其他有一对一关系的族群的节点数差
    mDstDelta=0.0;//目标部分中的族群和其他有一对一关系的族群的节点数差
}

SFNum PopGraph::PopGain::GetGain(SNum ncount)
{
    SFNum fCount=(SFNum)ncount;
    if(ncount<0)
        fCount=mSelfCount;
    if(fCount>mSelfCount)
        fCount=mSelfCount;
    SFNum fAllToAll= fCount*mExternal-fCount*mInternal-(mSelfCount-fCount)*fCount*mSelfPropa;
    if(!mOneToOne)
        return fAllToAll;
    SFNum dst=mDstDelta+fCount;
    SFNum src=mSrcDelta-fCount;
    return fAllToAll+Get121Gain(mSrcDelta,mDstDelta)-Get121Gain(src,dst);
}

SNum PopGraph::PopGain::GetCountWithMaxGain()
{
    SFNum fATAInf;
    SFNum fATAMax;//全对全收益最大时的count取值
    SNum nATAMax;
    if(mSelfPropa==0.0)//不存在自连接的边
    {
        if(mExternal>mInternal)
            fATAMax=mSelfCount;
        else
            fATAMax=0.0;
    }
    else
    {
        fATAInf=(mSelfPropa*mSelfCount+mInternal-mExternal)/(2*mSelfPropa);
    }
    //ATA是个有最小值的二次函数，最大值在远离拐点的地方
    if(abs(fATAInf-0.0)<abs(fATAInf-mSelfCount))
    {
        nATAMax=(SNum)mSelfCount;
        fATAMax=mSelfCount;
    }
    else
    {
        fATAMax=0.0;
        nATAMax=0;
    }
    return nATAMax;
    /*if(!mOneToOne)//如果没有1对1的边
    {
        return nATAMax;
    }
    //存在1对1的边
    if(mSrcDelta<mDstDelta)
    {
        //if((mDstDelta+nATAMax)<0 || (mSrcDelta-nATAMax)>=0)
            return nATAMax;
        
    }
    else
    {
        
    }*/
    
}

bool PopGraph::PopGain::EqualRecord(const MOVE_RECORD &record)
{
    if(mParent->GetPopID()!=record.Pop)
        return false;
    if(mPart!=record.SrcPart)
        return false;
    if(mParent->GetCount()!=record.Count)
        return false;
    return true;
}