#pragma once

#include <vector>
#include <map>

#include "util.h"
#include "msim_schedule.h"

typedef struct _POP_INFO
{
    SNum nPopIndex;
    SNum nNodeCount;
}POP_INFO;

typedef struct _MOVE_RECORD//移动记录
{
    SNum Pop;
    SNum SrcPart;
    SNum DstPart;
    SNum Count;
}MOVE_RECORD;

typedef struct _CONN_INFO
{
    SNum pop1,pop2;//边所连接的两个族群
    bool bOneToOne;//是否是一对一的连接，如果是，则fPropa成员无效
    double fPropa;//连接数占全连接数的比例
    SFNum fWeight,fDelay;
}CONN_INFO;

typedef struct _SYN_BUILD//构建突触时所用的结构
{
    SNum preOffset,postOffset;
    SNum preCount,postCount;
    SFNum weight,delay;
    bool bOneToOne;//是否是一对一的连接，如果是，则fPropa成员无效
    double fPropa;//连接数占全连接数的比例
    
    SNum postWholeCount;//突触后神经元所在大族群的神经元总数
    SNum preOffsetInWhole;//该前神经元族群在整个大神经元族群中的偏移
    SNum postOffsetInWhole;//该后神经元族群在整个大神经元族群中的偏移
}SYN_BUILD;

typedef std::map<SNum,std::pair<SNum,SNum>> NodeMap;


class PopGraph
{
public:
    class PopNode;
    class PopGain//计算收益的类,表示某个族群挪到某个部分时的收益
    {
    private:
        SFNum Get121Gain(SFNum srcDelta,SFNum dstDelta);//计算一对一连接的割边数
    public:
        PopGraph::PopNode *mParent;//所在的族群
        SNum mPart;//所对应的部分
        SFNum mExternal=0.0;//族群内每个节点到目标部分节点的边数
        SFNum mInternal=0.0;//族群内每个节点到所在部分其他族群节点的边数
        SFNum mSelfPropa=0.0;//族群内每个节点和其他节点的边数比率
        SFNum mSelfCount=0.0;//族群节点数
        bool mOneToOne=false;//是否有一对一的边
        SFNum mSrcDelta=0.0;//族群所在部分与其他有一对一关系的族群的节点数差
        SFNum mDstDelta=0.0;//目标部分中的族群和其他有一对一关系的族群的节点数差
    public:
        void Reset();
        SFNum GetGain(SNum ncount);//计算收益
        SNum GetCountWithMaxGain();//计算最大收益对应的节点数
        bool EqualRecord(const MOVE_RECORD &record);//判断在完全移动的情况下与之前的移动记录是否重合
    };
    class PopNode//双向循环链表
    {
    private:
        SNum mPopID;
        SNum mCount;//节点数量
        SNum mConnCount;//连接数量
        SNum mPart;//所属的部分
        SNum mPartCount;
        double mCarveCount;//当前的割边数 
        SNum mLastMove;//上次移动的时间
        CONN_INFO * mConns;
        PopGraph *mParent;
        PopNode *mpPrev,*mpNext;//整体链表
        PopNode *mpPopPrev,*mpPopNext;//族群之间的链表
        PopGain *mGains;//各个部分的收益
    public:
        PopNode *Next();
        SNum GetCount();
        SNum GetPopID();
        SNum GetPart();
        SFNum GetCarveCount();
        SNum GetLastMove();
        PopGain *GetPartGain(SNum part);
    public:
        PopNode(PopGraph *parent,SNum popID,SNum popN,SNum part);
        ~PopNode();
        void Insert(PopNode *pNode);//在当前节点的后面插入新节点
        void Delete();//从链表中删除
        PopNode * MoveTo(SNum part,SNum ncount);//将当前簇群分裂若干个节点到目标的部分，如果当前节点被合并，则返回合并后的节点，否则返回自己
        void UpdateGains();
        PopGain *GetMaxGain(SNum ncount,bool bPositive);//返回指定数目下最多的收益,ncount为负数表示为当前族群数目，bPositive表示是否要求收益为正
        void PrintToScreen();
        PopNode *CloneList();//将整个链表克隆
    public:
        static void DeleteWholeLink(PopNode *pNode);//删除整个链表
    };
private:
    SNum mPopCount;
    SNum mNodeCount;
//构建图时使用的数据
    SNum *mPops;//每个族群的节点数
    std::vector<CONN_INFO> mConns;
//开始划分时的数据
    std::map<std::pair<SNum,SNum>,std::vector<SNum>> mNeighbor;//各个族群的邻接表
    SNum mMoveTime;//当前移动的时间
    double mCarveCount;//当前的割边数
    SNum mPartCount;//要划分的份数，大于0时代表划分开始了
    SNum *mConnCountPerPop;//每个族群的边数
    CONN_INFO **mConnPerPop;//每个族群的边
    PopNode *mFirstPop;//划分链表的首节点
    PopNode **mNodePerPop;//划分链表中各个族群的首节点
    SNum *mCountPerPart;//每个部分当前划分状态下的节点数
    std::vector<PopNode *> mInternalNode;//在非边界簇群
    double mMinCarveCount;//目前效果最好的划分结果所对应的割边数
    PopNode *mBestPop;//目前效果最好的划分结果
    MOVE_RECORD mLastBalance;//上一次平衡性移动的数据
//输出时用的数据
    std::map<SNum,std::map<std::pair<SNum,SNum>,SYN_BUILD>> m121ConnBuilds;//1对1总连接下标到连接构建数据的映射
    bool mUpdated;//划分是否发生过更新
    NodeMap *mPopIDToArray;//族群id到单个GPU内节点大数组的偏移与长度
private:
    SNum MovePop(PopNode *pNode,SNum ncount,SNum dstPart);
    void UpdatePopNode();//更新输出数据
    SNum GetSubConnOffset(SNum connIdx,SNum part1,SNum part2);//获取全连接状态下子连接在整体连接中的偏移量
    SNum GetSubPopOffset(SNum part,SNum pop);//获取子族群在总族群下的偏移量
public:
    PopGraph(SNum nPop,SNum nPopCounts[]);
    ~PopGraph();
    bool AddConn(SNum pop1,SNum pop2,bool bOneToOne,double fPropa=1.0,SFNum weight=1.0,SFNum delay=1.0);
    bool StartPartition(SNum nPart,bool bAverage=true);//开始划分成nPart个分部，bAverage表示初始划分是否用平均的方式
    bool UpdateGains();//更新所有族群的收益信息
    bool GreedyMove();//按贪心原则最大化收益下整体移动
    SNum MaximizeMove(SNum ncount,SNum tolerance,SNum srcPart,SNum dstPart);//最大化收益下从指定源部分移动指定个数的节点到指定目标部分，返回成功移动的个数
    SNum Turbulence(SNum ncount,SNum srcPart,SNum dstPart);//扰动，随机选择某个部分的非边界族群，移动指定数目的节点到目标部分
    SNum GetPartCount();
    SNum GetPartNodeCount(SNum index);
    SNum GetPopCountInPart(SNum pop,SNum part);
    void Partition(SNum turbCount,SFNum turbDecay);//划分的主体部分,count为初始扰动的节点数
    void PrintPartition(bool bPrintBest=false);
    SNum GetInnerConn(SNum nPart,SYN_BUILD *builds,SNum nBuild);//获取展开后的内部连接，返回连接数
    SNum GetOutterConn(SNum nSrcPart,SNum nDstPart,SYN_BUILD *builds,SNum nBuild);//获取展开后的不同分部之间的连接，返回连接数
    bool GetPopsInPart(SNum nPart,std::vector<POP_INFO> &Pops);//获取划分后的某个分部的所有族群信息
};