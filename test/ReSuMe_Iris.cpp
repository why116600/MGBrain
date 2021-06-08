#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <map>


#include <util.h>
#include <msim_network.h>
#include <ReSuMeLearn.h>

#define TIME_LENGTH 1000.0
#define TARGETN 3

MultiGPUBrain::Network mnetwork(0.1);
DataManager dm;


time_t delta_time(const timeval &tv1,const timeval &tv2)
{
    time_t costsec,costusec;
	if(tv2.tv_usec<tv1.tv_usec)
	{
		costsec=tv2.tv_sec-tv1.tv_sec-1;
		costusec=1000000+tv2.tv_usec-tv1.tv_usec;
	}
	else
	{
		costsec=tv2.tv_sec-tv1.tv_sec;
		costusec=tv2.tv_usec-tv1.tv_usec;
	}
    return costsec*1000000+costusec;
}

int main(int argc,char *argv[])
{
	if (argc <= 1)
	{
		fprintf(stderr, "no data arguments!\n");
		return -1;
	}
	if (!dm.LoadDat(argv[1]))
	{
		fprintf(stderr, "Cannot load data file!\n");
		return -1;
	}
	std::map<std::string, SFNum> params;
	std::vector<SNum> inns;
	std::vector<SFNum> spikes,outspikes[TARGETN];
	SYNAPSE syn;
	SNum outputn;
	SUNum len;
	SUNum* orders;
	SFNum dw;
	SNum maxi, maxs;
	SNum nCorrect = 0;
    timeval tv1,tv2,tv3,tv4;
	time_t tt1,tt2;
	srand(time(NULL));
	params["I_e"] = 3.0;
	outputn = mnetwork.CreatePopulation("LIF", 3, params);
	for (SUNum i = 0; i < TARGETN; i++)
	{
		mnetwork.WatchNeuron(outputn, i);
	}
	for (SUNum i = 0; i < dm.GetDataSize(); i++)
	{
		SNum n = mnetwork.CreateSpikeGenerator();
		inns.push_back(n);
		mnetwork.Connect(n, outputn, 0.0f, 1.0f, false, 1.0);
	}
	if (!mnetwork.Compile(1))
	{
		fprintf(stderr, "Cannot compile the network!\n");
		return -1;
	}
	//打乱顺序
	len = dm.GetCount();
	orders = new SUNum[len];
	for (SUNum i = 0; i < len; i++)
		orders[i] = i;
	for (SUNum i = 0; i < len; i++)
	{
		SUNum n = rand() % (len - i);
		SUNum t;
		t = orders[n + i];
		orders[n + i] = orders[i];
		orders[i] = t;
	}
	//mnetwork.Connect(inns[0],0,outputn,0,5000.0f,1.0f);
	/*mnetwork.Connect(inns[0],0,outputn,0,5741.91093176f,1.0f);
	mnetwork.Connect(inns[1],0,outputn,0,-222.38012743f,1.0f);
	mnetwork.Connect(inns[2],0,outputn,0,-7976.23592193f,1.0f);
	mnetwork.Connect(inns[3],0,outputn,0,-274.62229871f,1.0f);
	mnetwork.Connect(inns[0],0,outputn,1,1263.26812371f,1.0f);
	mnetwork.Connect(inns[1],0,outputn,1,-1441.49908844f,1.0f);
	mnetwork.Connect(inns[2],0,outputn,1,1348.5247809f,1.0f);
	mnetwork.Connect(inns[3],0,outputn,1,73.77322696f,1.0f);
	mnetwork.Connect(inns[0],0,outputn,2,976.78766841f,1.0f);
	mnetwork.Connect(inns[1],0,outputn,2,-2455.76308525f,1.0f);
	mnetwork.Connect(inns[2],0,outputn,2,3013.25934358f,1.0f);
	mnetwork.Connect(inns[3],0,outputn,2,-141.84535751f,1.0f);*/
	//打乱突触权值
	for(SUNum i=0;i<dm.GetDataSize();i++)
	{
		for(SUNum j=0;j<TARGETN;j++)
		{
			mnetwork.Connect(inns[i],0,outputn,j,(SFNum)(rand()%11)-5.0f,1.0f);
		}
	}
    gettimeofday(&tv1, NULL);
	//开始训练
	for (SNum t = 0; t < 10; t++)
	{
		for (SUNum i = 0; i < len; i++)
		{
			//输入数据
			for (SUNum j = 0; j < dm.GetDataSize(); j++)
			{
				dm.GetDataSpikeTrain(orders[i], j, TIME_LENGTH, spikes);
				mnetwork.SetSpikeTrain(inns[j], spikes);
				spikes.clear();
			}
			//仿真并取结果
			mnetwork.Simulate(TIME_LENGTH);
			printf("spike count:");
			for (SUNum j = 0; j < TARGETN; j++)
			{
				mnetwork.GetNeuronSpikes(outputn, j, outspikes[j]);
				printf("%ld ",outspikes[j].size());
			}
			printf("\n");
			//学习
			for (SNum j = 0; j < dm.GetDataSize(); j++)
			{
				for (SNum k = 0; k < TARGETN; k++)
				{
					dw = dm.GetDeltaWeight(orders[i], j, k, TIME_LENGTH, outspikes[k]);
					if(mnetwork.GetSynapseInfo(inns[j],0,outputn,k,&syn))
						mnetwork.Connect(inns[j],0,outputn,k,syn.weight+dw,syn.delay);
					else
						fprintf(stderr,"Fuck synapse!\n");
					
				}
			}
			for (SUNum j = 0; j < TARGETN; j++)
			{
				outspikes[j].clear();
			}
		}
	}
    gettimeofday(&tv2, NULL);
	//再次打乱顺序
	len = dm.GetCount();
	for (SUNum i = 0; i < len; i++)
		orders[i] = i;
	for (SUNum i = 0; i < len; i++)
	{
		SUNum n = rand() % (len - i);
		SUNum t;
		t = orders[n + i];
		orders[n + i] = orders[i];
		orders[i] = t;
	}
    gettimeofday(&tv3, NULL);
	//验证
	for (SUNum i = 0; i < 50; i++)
	{
		/*spikes.push_back(10.0f);
		mnetwork.SetSpikeTrain(inns[1], spikes);
		mnetwork.SetSpikeTrain(inns[2], spikes);
		mnetwork.SetSpikeTrain(inns[3], spikes);
		for(int j=2;j<=10;j++)
			spikes.push_back(10.0f*j);
		mnetwork.SetSpikeTrain(inns[0], spikes);
		double values[4];
		dm.GetDataValues(i,values,4);
		printf("input value:%lf %lf %lf %lf\n",values[0],values[1],values[2],values[3]);*/
		for (SUNum j = 0; j < dm.GetDataSize(); j++)
		{
			dm.GetDataSpikeTrain(orders[i], j, TIME_LENGTH, spikes);
			if(!mnetwork.SetSpikeTrain(inns[j], spikes))
			{
				fprintf(stderr,"setting spike train failed!");
			}
			spikes.clear();
		}
		mnetwork.Simulate(TIME_LENGTH);
		maxi = 0;
		maxs = 0;
		for (SUNum j = 0; j < TARGETN; j++)
		{
			mnetwork.GetNeuronSpikes(outputn, j, spikes);
			//if(spikes.size()>0)
			//printf("round %u,neuron %u output %ld spikes!\n",i,j,spikes.size());
			if (spikes.size() > maxs)
			{
				maxi = j;
				maxs = (SNum)spikes.size();
			}
			spikes.clear();
		}
		printf("answer:%d,target:%lf\n",maxi,dm.GetTargetValue(orders[i], 0));
		if (maxi == (SNum)dm.GetTargetValue(orders[i], 0))
			nCorrect++;
	}
    gettimeofday(&tv4, NULL);
	//输出突触权值
	printf("synapse weights:\n");
	for(SUNum i=0;i<TARGETN;i++)
	{
		for(SUNum j=0;j<dm.GetDataSize();j++)
		{
			mnetwork.GetSynapseInfo(inns[j],0,outputn,i,&syn);
			printf("%f ",syn.weight);
		}
		printf("\n");
	}
	//结果
	tt1=delta_time(tv1,tv2);
	tt2=delta_time(tv3,tv4);
	printf("Correct:%d\n", nCorrect);
	printf("trainning costs time:%ld.%06ld\n",tt1/1000000,tt1%1000000);
	printf("verifying costs time:%ld.%06ld\n",tt2/1000000,tt2%1000000);
	delete[]orders;
	return 0;
}