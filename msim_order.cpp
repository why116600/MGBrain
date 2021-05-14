#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "util.h"
#include "msim_order.h"

namespace MultiGPUBrain
{

OrderArrange::OrderArrange(SNum linkLen,SNum len,OrderArrange *pFirst)
:mLength(len)
,mFirst(pFirst)
,mNext(NULL)
{
    mIndex=len-linkLen-1;
    mUsed=new bool[len];
    if(pFirst==this)
    {
        mHit=new bool[len*len];
        for(SNum i=0;i<len*len;i++)
            mHit[i]=false;
    }
    else
    {
        mHit=pFirst->mHit;
    }
    
    for(SNum i=0;i<len;i++)
    {
        mUsed[i]=false;
    }
    if(linkLen>0)
    {
        mNext=new OrderArrange(linkLen-1,len,pFirst);
    }
    
}

OrderArrange::OrderArrange(SNum len)
:mUsed(NULL)
{
    new (this)OrderArrange(len-1,len,this);
}

OrderArrange::~OrderArrange()
{
    delete []mUsed;
    if(mFirst==this)
        delete []mHit;
}

bool OrderArrange::PushForward(SNum index)
{
    SNum step=(SNum)mOrders.size();
    if(index<0 || index>=mLength || mIndex==index || mUsed[index] || mHit[step*mLength+index])
        return false;
    mOrders.push_back(index);
    mUsed[index]=true;
    mHit[step*mLength+index]=true;
    return true;
}

SNum OrderArrange::PopBack()
{
    SNum index;
    SNum step=(SNum)mOrders.size()-1;
    if(mOrders.size()<=0)
        return -2;
    index=*mOrders.rbegin();
    mUsed[index]=false;
    mHit[step*mLength+index]=false;
    mOrders.pop_back();
    return index;
}

bool OrderArrange::Arrange()
{
    SNum step=0;
    SNum index=0;
    bool bForward=false;
    while(true)
    {
        if(step>=(mLength-1))
        {
            if(mNext)
            {
                if(mNext->Arrange())
                {
                    return true;
                }
                else
                {
                    do
                    {
                        index=PopBack()+1;
                        if(index==mIndex)
                            index++;
                    } while (index<mLength);
                    step=(SNum)mOrders.size();
                }
            }
            else
            {
                return true;
            }
            
        }
        else
        {
            while(!(bForward=PushForward(index++)) && index<mLength);
            if(!bForward)
            {
                do
                {
                    index=PopBack()+1;
                    if(index<0)
                        return false;
                    if(index==mIndex)
                        index++;
                } while (index<0);
            }
            else
            {
                index=0;
            }
            step=(SNum)mOrders.size();
        }
        
    }
    return false;
}

void OrderArrange::GetOrder(SNum index,std::vector<SNum> &orders)
{
    if(index==0)
    {
        orders=mOrders;
    }
    else if(mNext)
    {
        mNext->GetOrder(index-1,orders);
    }
    
}

}