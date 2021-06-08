#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <string>
#include <vector>


#include <util.h>
#include <msim_network.h>


MultiGPUBrain::Network mnetwork(0.1);

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

MultiGPUBrain::PARTITION_MODEL models[]={MultiGPUBrain::LoadBalance,MultiGPUBrain::FIFP,MultiGPUBrain::Average};

int main(int argc,char *argv[])
{
    SNum exc1,exc2;
	std::map<std::string,SFNum> params;
    std::vector<SFNum> spikes;
    timeval tv1,tv2;
    long tt,ctt;
    SNum nGPU=1,nSize=100;
    SNum mode=0;
    MultiGPUBrain::PARTITION_MODEL pm=MultiGPUBrain::LoadBalance;

    if(argc>1)
        nSize=atoi(argv[1]);
    if(argc>2)
        nGPU=atoi(argv[2]);
    if(argc>3)
        mode=atoi(argv[3]);
    pm=models[mode];

    printf("Use %d GPUs to simulate a network with %d scale\n",nGPU,nSize);
    
    gettimeofday(&tv1, NULL);
    exc1=mnetwork.CreatePopulation("LIF",nSize,params);
    exc2=mnetwork.CreatePopulation("LIF",nSize,params);

    mnetwork.Connect(exc1,exc1,1.0,1.0,false,1.0);
    mnetwork.Connect(exc2,exc2,-1.0,1.0,false,1.0);
    mnetwork.Connect(exc1,exc2,10.0,1.0,true,1.0);
    mnetwork.Connect(exc2,exc1,-10.0,1.0,true,1.0);

    if(!mnetwork.Compile(nGPU,pm))
    {
        fprintf(stderr,"Fuck compiling!\n");
        return -1;
    }
    mnetwork.WatchNeuron(exc1,0);
    gettimeofday(&tv2,NULL);
    ctt=delta_time(tv1,tv2);
    printf("start simulation\n");
    gettimeofday(&tv1, NULL);
    if(nSize>20000)
    {
        mnetwork.Simulate(100.0);
    }
    else
    {
        for(int i=0;i<10;i++)
        mnetwork.Simulate(10000.0);
    }
    gettimeofday(&tv2,NULL);
    tt=delta_time(tv1,tv2);
	printf("compiling costs time:%ld.%06ld\n",ctt/1000000,ctt%1000000);
	printf("cost time:%ld.%06ld\n",tt/1000000,tt%1000000);
    mnetwork.GetNeuronSpikes(exc1,0,spikes);
    printf("Got %ld spikes\n",spikes.size());
    if(argc>4)
    {
        FILE *fp;
        fp=fopen(argv[4],"a");
        fprintf(fp,"%d %d %d %ld.%06ld %ld.%06ld\n",nSize*2,nGPU,mode,ctt/1000000,ctt%1000000,tt/1000000,tt%1000000);
        fclose(fp);
    }

    return 0;
}