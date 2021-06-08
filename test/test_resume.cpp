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

int main(int argc,char *argv[])
{
    SNum inputn,exc;
	std::map<std::string,SFNum> params;
    std::vector<SFNum> spikes;
    timeval tv1,tv2;
    long tt;
    SNum nGPU=1,nSize=2048;

    if(argc>1)
        nSize=atoi(argv[1]);
    if(argc>2)
        nGPU=atoi(argv[2]);

    printf("Use %d GPUs to simulate a network with %d scale\n",nGPU,nSize);
    
    inputn=mnetwork.CreatePopulation("LIF",nSize,params);
    exc=mnetwork.CreatePopulation("LIF",1,params);

    mnetwork.Connect(inputn,exc,10.0,1.0,false,1.0);

    if(!mnetwork.Compile(nGPU))
    {
        fprintf(stderr,"Fuck compiling!\n");
        return -1;
    }
    //mnetwork.WatchNeuron(exc,0);
    printf("start simulation\n");
    gettimeofday(&tv1, NULL);
    for(int i=0;i<10;i++)
    mnetwork.Simulate(10000.0);
    gettimeofday(&tv2,NULL);
    tt=delta_time(tv1,tv2);
	printf("cost time:%ld.%06ld\n",tt/1000000,tt%1000000);
    //mnetwork.GetNeuronSpikes(exc,0,spikes);
    //printf("Got %ld spikes\n",spikes.size());
    if(argc>3)
    {
        FILE *fp;
        fp=fopen(argv[3],"a");
        fprintf(fp,"%d %d %ld.%06ld\n",nSize*5,nGPU,tt/1000000,tt%1000000);
        fclose(fp);
    }

    return 0;
}