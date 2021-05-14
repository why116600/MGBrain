#pragma once

#include "util.h"
#include <vector>

namespace MultiGPUBrain
{



class OrderArrange
{
private:
    std::vector<SNum> mOrders;
    bool *mUsed,*mHit;
    SNum mIndex;
    SNum mLength;

    OrderArrange *mFirst,*mNext;
private:
    OrderArrange(SNum linkLen,SNum len,OrderArrange *pFirst);
public:
    OrderArrange(SNum len);
    ~OrderArrange();
    bool PushForward(SNum index);
    SNum PopBack();
    bool Arrange();
    void GetOrder(SNum index,std::vector<SNum> &orders);
};

}