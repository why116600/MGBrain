
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include "util.h"




void GenerateRatedSpikes(NSPIKE_GEN *gen,SNum freq,SFNum simulTime)
{
	SFNum step=(SFNum)(simulTime/((SFNum)freq+0.0));
	SFNum pos=(SFNum)(simulTime/((SFNum)freq+1.0));
	SNum n=0;
	while(pos<simulTime)
	{
		gen->spikes[n++]=floor(pos*10.0)/10.0;
		pos+=step;
	}
	gen->length=n;

}
