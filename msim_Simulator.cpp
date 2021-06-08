#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "util.h"
#include "PopGraph.h"
#include "msim_SynManage.h"
#include "msim_network.h"
#include "msim_Simulator.h"

namespace MultiGPUBrain
{

Simulator::Simulator(SNum nMeshSize)
:mgGens(NULL)
,mgLIF(NULL)
,mgLifArg(NULL)
,mgSynapses(NULL)
,mgSections(NULL)
,mgNetwork(NULL)
,mgRecorder(NULL)
,mgOutsToIn(NULL)
,mgActiveCountByGrid(NULL)
,mNodeChanged(true)
,mMinDelay(-1.0f)
,mMaxDelay(-1.0f)
,mMaxBlock(0)
,mBlockSize(BLOCKSIZE)
,mGPUID(-1)
,mMeshSize(nMeshSize)
,mExtraSyn(NULL)
,mMaxGrid(0)
,mInnerGridSize(0)
,mInnerBuildCount(0)
,mInnerBuild(NULL)
{
    omp_init_lock(&mLock);
    for(SNum i=0;i<TYPE_COUNT;i++)
    {
        mNodeCounts[i]=0;
        mPopCount[i]=0;
    }
}

Simulator::~Simulator()
{
    if(mExtraSyn)
        delete mExtraSyn;

	if(mInnerBuild)
		delete []mInnerBuild;

    CleanOutterBuilds();
    CleanSimulData();
	CleanSynapse();
	CleanOutputSynapse();
    CleanLIF();
    omp_destroy_lock(&mLock);
}


void Simulator::CleanOutterBuilds()
{
	for(SNum i=0;i<(SNum)mOutterBuilds.size();i++)
	{
		delete []mOutterBuilds[i].second;
	}
	mOutterBuilds.clear();
}

void Simulator::Lock()
{
    omp_set_lock(&mLock);
}

void Simulator::Unlock()
{
    omp_unset_lock(&mLock);
}

}