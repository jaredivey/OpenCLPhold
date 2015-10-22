#include "mwc64x/mwc64x_rng.cl"

__constant float d_lookahead = 4.0f;
__constant float d_local_rate = 0.9f;
__constant float d_delay_time = 0.9f;
__constant int   d_num_lps = 1 << 20;
__constant float d_stop_time = 60.0f;

__kernel void initializeSimulator(__global mwc64x_state_t* state,
								__global float* current_time,
								__global float* event_time,
								__global int* event_lp,
								__global int* events_processed)
{
  int idx = get_global_id(0);

  if(idx < d_num_lps)
  {
    mwc64x_state_t rand = state[idx];
    MWC64X_SeedStreams(&rand, (1337 << 20) + idx, 0);
    event_lp[idx] = idx; //everyone starts with a events at some time between 0 and 1 time units
    event_time[idx] = (float)MWC64X_NextUint(&rand) / (float)(1e37f);
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

__global void markNextEventByLP(__global int* event_lp, __global unsigned char* flags)
{
  int idx = get_global_id(0);

  if(idx == 0 || event_lp[idx] != event_lp[idx-1])
  {
    flags[idx] = 1;
  }
  else
  {
    flags[idx] = 0;
  }
}

__global void simulatorRun(__global mwc64x_state_t* state,
						__global float* current_time,
						__global float* event_time,
						__global int* event_lp,
						__global float* current_lbps,
						__global int* events_processed)
{
  //goal:  Minimize the number of global memory accesses / anywhere that does read/write using []/arrays
  int idx = get_global_id(0);

  float safe_time = *current_lbps + d_lookahead;

  //check the next event
  float next_event_time = event_time[idx];

  //ok to process?
  if(next_event_time <= safe_time && next_event_time < d_stop_time)
  {
    //generate new event
    mwc64x_state_t rand = state[idx];

    float cur_time = current_time[idx];
    int ev_lp = event_lp[idx];

    //sanity check
    if(cur_time > next_event_time || ev_lp != idx)
    {
      printf("EPIC FAIL! Agghh Gads!  CurrentTime: %f, EventTime: %f LP: %d, EventLP %d\n", cur_time, next_event_time, idx, ev_lp);
    }

    events_processed[idx]++;

    //create new event
    float remote_flip = (float)MWC64X_NextUint(&rand) / (float)(1e37f);

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
      target_lp = ((float)MWC64X_NextUint(&rand) / (float)(1e37f)) * (d_num_lps-1);
      new_event_time += d_lookahead;
    }

    //writes
    current_time[idx] = next_event_time;
    event_time[idx] = new_event_time;
    event_lp[idx] = target_lp;
    state[idx] = rand;
  }
}