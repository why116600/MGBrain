
#include <stdio.h>
#include <cuda_runtime.h>

#include "util.h"
#include "sim_LIF.h"
#include "sim_recorder.h"


__constant__ SFNum EPSCInitialValue_=1.359141;
__constant__ SFNum IPSCInitialValue_=1.359141;
__constant__ SFNum RefractoryCounts_=20;
__constant__ SFNum P11_ex_=0.951229;
__constant__ SFNum P21_ex_=0.095123;
__constant__ SFNum P22_ex_=0.951229;
__constant__ SFNum P31_ex_=0.000019;
__constant__ SFNum P32_ex_=0.000388;
__constant__ SFNum P11_in_=0.951229;
__constant__ SFNum P21_in_=0.095123;
__constant__ SFNum P22_in_=0.951229;
__constant__ SFNum P31_in_=0.000019;
__constant__ SFNum P32_in_=0.000388;
__constant__ SFNum P30_=0.000398;
__constant__ SFNum P33_=0.990050;
__constant__ SFNum expm1_tau_m_=-0.009950;
__constant__ SFNum weighted_spikes_ex_=0.000000;
__constant__ SFNum weighted_spikes_in_=0.000000;

#define ARG_REF(x) args[argPos].x[argIndex]
//#define NODE(x) node->x[index]
#define NODE(x) LIF_STRUCT_REF(nodes,x,index)

#define mV_reset ARG_REF(V_reset)
#define mC_ref NODE(C_ref)
#define mI_e ARG_REF(I_e)
#define mdI_ex_ NODE(dI_ex_)
#define mI_ex_ NODE(I_ex_)
#define mdI_in_ NODE(dI_in_)
#define mI_in_ NODE(I_in_)


#ifdef SINGLE_GPU
__global__ void initLIF(GNLIF *nodes,LIF_ARG *args,SNum count)
{
	unsigned int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index>=count)
		return;
	NODE(MP)=args[NODE(argPos)].V_init[NODE(argIndex)];
	NODE(C_ref)=0.0;
	NODE(I_ex_)=0.0;
	NODE(dI_ex_)=0.0;
	NODE(I_in_)=0.0;
	NODE(dI_in_)=0.0;
}
#endif

__device__ bool simulateLIF(SNum index,SFNum input,GNLIF *nodes,LIF_ARG *args,SNum now,SFNum timestep,RECORDER *recorder)
{
	SNum argPos=NODE(argPos);
	SNum argIndex=NODE(argIndex);
	SFNum mp=NODE(MP)-mV_reset;
	//if(input!=0.0)
		//printf("neuron %d got current:%f at %.1fs when MP=%f\n",index,input,(SFNum)now*timestep,NODE(MP));
	if(mC_ref<=0.0)
	{
		//if(input!=0.0)
			//printf("spike at %f\n",now);
		mp = P30_ * mI_e  + P31_ex_ * mdI_ex_ + P32_ex_ * mI_ex_ +P31_in_ * mdI_in_ + P32_in_ * mI_in_ + expm1_tau_m_ * mp + mp;
	}
	else
		mC_ref-=timestep;

	NODE(MP)=mp+mV_reset;

	mI_ex_ = P21_ex_ * mdI_ex_ + P22_ex_ * mI_ex_;
	mdI_ex_ *= P11_ex_;
	mI_in_ = P21_in_ * NODE(dI_in_) + P22_in_ * mI_in_;
	mdI_in_ *= P11_in_;

	if(input>0.0)
		mdI_ex_ +=EPSCInitialValue_ * input;
	else
		mdI_in_ +=IPSCInitialValue_ * input;

	if(NODE(MP)>ARG_REF(V_th))
	{
#ifdef USE_RECORDER
		if(recorder)
			AddToRecorder(&recorder[index],(SFNum)now*timestep);
#endif
		//printf("neuron %d is fired at step %d for MP=%f\n",index,now,NODE(MP));
		NODE(MP)=mV_reset;
		mC_ref=ARG_REF(T_ref);
		//printf("index=%d,Next=%p\n",index,nextPoint[0]);
		return true;
	}
	return false;
}

//__device__ bool resolutionLIF(SNum index,GNLIF *node,SNum now,SNum t)
//{
	//SFNum I_ex,dI_ex;
//}

