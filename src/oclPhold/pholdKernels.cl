

__constant float d_lookahead = 4.0f;
__constant float d_local_rate = 0.9f;
__constant float d_delay_time = 0.9f;
__constant int   d_num_lps = 1 << 20;
__constant float d_stop_time = 60.0f;

#ifndef _PHOLD_KERNEL_H_
#define _PHOLD_KERNEL_H_

/* This reduction interleaves which threads are active by using the modulo
   operator.  This operator is very expensive on GPUs, and the interleaved 
   inactivity means that no whole warps are active, which is also very 
   inefficient */
__kernel void reduce0(__global T *g_idata, __global T *g_odata, unsigned int n, T max_val, __local T* sdata)
{
    // load shared mem
    unsigned int tid = get_local_id(0);
    unsigned int i = get_global_id(0);
    
    sdata[tid] = (i < n) ? g_idata[i] : max_val;
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    for(unsigned int s=1; s < get_local_size(0); s *= 2) {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0) {
            sdata[tid] = (sdata[tid] < sdata[tid + s] ? sdata[tid] : sdata[tid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[get_group_id(0)] = sdata[0];
}

__global void markNextEventByLP(int* event_lp, unsigned char* flags)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx == 0 || event_lp[idx] != event_lp[idx-1])
  {
    flags[idx] = 1;
  }
  else
  {
    flags[idx] = 0;
  }
}

__global void initializeSimulator(curandState* state, float* current_time, float* event_time, int* event_lp, int* events_processed)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < d_num_lps)
  {
    curandState rand = state[idx];
    curand_init((1337 << 20) + idx, 0, 0, &rand);
    event_lp[idx] = idx; //everyone starts with a events at some time between 0 and 1 time units
    event_time[idx] = curand_uniform(&rand) * 1.0f;
    current_time[idx] = 0.0f;
    state[idx] = rand;
    events_processed[idx] = 0;
  }
  else
  {
  	event_lp[idx] = idx - d_num_lps; //this gives every LP a stop event
    event_time[idx] = d_stop_time;
  }
}

__global void simulatorRun(curandState* state, float* current_time, float* event_time, int* event_lp, float* current_lbps, int* events_processed)
{
  //goal:  Minimize the number of global memory accesses / anywhere that does read/write using []/arrays
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float safe_time = *current_lbps + d_lookahead;

  //check the next event
  float next_event_time = event_time[idx];

  //ok to process?
  if(next_event_time <= safe_time && next_event_time < d_stop_time)
  {
    //generate new event
    curandState rand = state[idx];

    float cur_time = current_time[idx];
    int ev_lp = event_lp[idx];

    //sanity check
    if(cur_time > next_event_time || ev_lp != idx)
    {
      printf("EPIC FAIL! Agghh Gads!  CurrentTime: %f, EventTime: %f LP: %d, EventLP %d\n", cur_time, next_event_time, idx, ev_lp);
    }

    events_processed[idx]++;

    //create new event
    float remote_flip = curand_uniform(&rand);

    //next_event_time stores current time if we reach here
    float new_event_time = d_delay_time + next_event_time;

    int target_lp;

    if(remote_flip < d_local_rate)
    {
      target_lp = idx;
    }
    else
    {
      //target_lp could be me, however we'll assume that the probability is small.
      target_lp = curand_uniform(&rand) * (d_num_lps-1);
      new_event_time += d_lookahead;
    }

    //writes

    current_time[idx] = next_event_time;
    event_time[idx] = new_event_time;
    event_lp[idx] = target_lp;
    state[idx] = rand;
  }
}

#endif