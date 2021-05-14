#pragma once

typedef int SNum;
typedef unsigned long long SLNum;
typedef float SFNum;
typedef unsigned char SByte;

#ifndef LPVOID
	typedef void *LPVOID;
#endif

#define TYPE_COUNT 2

#define TYPE_GEN 0
#define TYPE_LIF 1

#define MAX_SPIKE_COUNT (1<<10)
#define BLOCKSIZE 512
#define GRIDSIZE(x) ((x+BLOCKSIZE-1)/BLOCKSIZE)

#define BLOCK2DSIZE 32
#define GRID2DSIZE(x) ((x+BLOCK2DSIZE-1)/BLOCK2DSIZE)

#define GETID(type,index) ((type<<24)|index)
#define GETTYPE(id) (id>>24)
#define GETNO(id) (id&0x00ffffff)

#define MAXLIF (1<<14)
#define LIF_STRUCT_INDEX(index) (index>>14)
#define LIF_NODE_INDEX(index) (index&(MAXLIF-1))
#define LIF_STRUCT_REF(arr,mem,index) (arr[index>>14].mem[index&(MAXLIF-1)])

struct  LIF_ARG
{
	SFNum V_init[MAXLIF];
	SFNum V_reset[MAXLIF];
	SFNum V_th[MAXLIF];
	SFNum Tau_m[MAXLIF];//Membrane time constant in ms
	SFNum C_m[MAXLIF];//Capacity of the membrane
	SFNum I_e[MAXLIF];//Constant external input current in pA
	SFNum T_ref[MAXLIF];//absolute refractory period
	SFNum tau_ex_[MAXLIF]; // ms
	SFNum tau_in_[MAXLIF]; // ms

	SNum PopOffset[MAXLIF];//offset in LIF nodes array of population's first node
	SNum PopNum[MAXLIF];

	LIF_ARG *pNext;//Only available in CPU
};

struct GNLIF
{
	SNum argPos[MAXLIF],argIndex[MAXLIF];

	SFNum MP[MAXLIF];
	SFNum C_ref[MAXLIF];
	SFNum I_ex_[MAXLIF];
	SFNum dI_ex_[MAXLIF];
	SFNum I_in_[MAXLIF];
	SFNum dI_in_[MAXLIF];
};

struct NLIF
{
	SFNum MP;

	SFNum V_init;
	SFNum V_reset;
	SFNum V_th;
	SFNum Tau_m;//Membrane time constant in ms
	SFNum C_m;//Capacity of the membrane
	SFNum I_e;//Constant external input current in pA
	SFNum T_ref;//absolute refractory period
	SFNum tau_ex_; // ms
	SFNum tau_in_; // ms
};

struct NSPIKE_GEN
{
	SNum length;
	SFNum spikes[MAX_SPIKE_COUNT];
};

struct NEXTRA_INFO//record extra information of neuron in larger logical neuron array
{
	SNum nTrainTag;
};

struct GNGen
{
	SNum length;
	SNum spikes[MAX_SPIKE_COUNT];
	SNum pos;
};

struct SYNAPSE
{
	SNum preIndex;//pre-synaptic node's index
	SNum postIndex;//post-synaptic node's index,filled by simulator
	SFNum weight;
	SFNum delay;
};

struct POP_CONN//族群之间的连接属性
{
	SNum preID;//pre-synaptic node's index
	SNum postID;//post-synaptic node's index,filled by simulator
	SFNum weight;
	SFNum delay;
	bool bOneToOne;
	double fProba;
};

#ifndef cudaStream_t
struct CUstream_st;
typedef CUstream_st *cudaStream_t;
#endif

struct OUTPUT_SYNAPSES//代表所有输出到一个分部的突触
{
    SYNAPSE *gSynapses;
	SLNum nGridSize;//存放突触数据的单个GPU显存块的大小
    SLNum *gSections;
	SNum nDstNode;//目标节点的个数
	SNum nDstMap;//本结构到pTarget中对应映射的下标
	SNum nMaxSynCount;//本结构的最大输出突触数
    SFNum *gCurrentBuffer;
    LPVOID pTarget;

	SNum nAccessEnable;//是否可以直接通过GPU2GPU的途径传输数据
	SFNum *gTargetCurrent;//如果可以直接点对点通信，则该成员为目标GPU的缓存
	cudaStream_t stream;
};

struct CURRENT_ELEMENT//用于跨GPU传输脉冲电流
{
	SNum nDstNode;//目标节点的个数
	SNum nDstMap;//本结构到pTarget中对应映射的下标
	SFNum *currentBuffer;
	SFNum *gTargetCurrent;//如果可以直接点对点通信，则该成员为目标GPU的缓存
};

struct NETWORK_DATA
{
	SNum offset[TYPE_COUNT+1];
	SNum maxSynapseCount;// max synapse count of one neuron within whole network
	SNum maxWholeSynCount;//所有神经元的对内、对外的最大输出突触数
	SYNAPSE *synapses;
	SLNum nGridSize;//存放突触数据的单个GPU显存块的大小
	SLNum *section;
	SNum *outSynapseCount;//denote out synapse count of each neuron in large logical array
	GNGen *gen;
	NEXTRA_INFO *genExtra;
	GNLIF *LIF;
	LIF_ARG *LIFArgs;
	NEXTRA_INFO *LIFExtra;
	SFNum timestep;
	SNum lastFired;

	SNum maxExtraCount;//单个神经元的最大额外突触数
	SNum meshSize;//额外突触的粒度大小
    SNum * node2Syn;//节点到额外突触的映射
    SNum *linkTable;//突触数组粒度对应的链表下一个节点的粒度下标，类似FAT分区格式
    SYNAPSE *extraSynapses;//额外突触数组

	SNum nOutput;//输出突触组的个数
	OUTPUT_SYNAPSES *outputs;//输出突触组
};

typedef void (*SYNAPSE_MODE)(SNum neuronIndex,SNum spikeIndex,void *param,NSPIKE_GEN *spikeEvents,NSPIKE_GEN *targetEvents,SYNAPSE *pSyn,NETWORK_DATA *pNetwork);

/*{
	const cudaError_t error = call;
	if(error!=cudaSuccess)
	{
		printf("Error:%s.Line %d,",__FILE__,__LINE__);
		printf("code:%d, reason:%s\n",error,cudaGetErrorString(error));
		exit(1);
	}
}*/

void GenerateRatedSpikes(NSPIKE_GEN *gen,SNum freq,SFNum simulTime);
