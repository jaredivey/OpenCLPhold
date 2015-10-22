/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* This sample queries and logs the properties of the CUDA devices present in the system. */

// Standard utilities and common systems includes

#include <oclUtils.h>
#include <shrQATest.h>

#include <memory>
#include <iostream>
#include <cassert>
#include <sys/time.h>
#include <limits.h>
#include <float.h>

#ifdef UNIX
#include <sstream>
#include <fstream>
#endif

#include <clpp/clpp.h>
#include <clpp/clppSort_RadixSortGPU.h>
#include <clpp/clppProgram.h>

//! Represents the state of a particular generator
typedef struct{ uint x; uint c; } mwc64x_state_t;

inline void clCheckError (cl_int err, const char *name)
{
	if (err != CL_SUCCESS)
	{
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		exit (EXIT_FAILURE);
	}
}

double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

static clppContext clpp_context;
static clppProgram pholdProgram = clppProgram();
static std::string kernelFileName = "/home/jared/repos/OpenCLPhold/src/oclPhold/phold.cl";

cl_int runInitializeSimulator (size_t *grid_size, size_t *block_size,
		cl_mem d_random_state, cl_mem d_lp_current_time, cl_mem d_event_time, cl_mem d_event_lp_number, cl_mem d_events_processed)
{
	cl_int clStatus;
	unsigned int a = 0;

	cl_kernel _kernel_initializeSimulator = clCreateKernel(pholdProgram._clProgram, "initializeSimulator", &clStatus);
	clCheckError (clStatus, "clCreateKernel: _kernel_initializeSimulator");

	clStatus = clSetKernelArg(_kernel_initializeSimulator, a++, sizeof (mwc64x_state_t), (const void*)&d_random_state);
	clCheckError (clStatus, "clSetKernelArg: d_random_state");
	clStatus |= clSetKernelArg(_kernel_initializeSimulator, a++, sizeof(cl_mem), (const void*)&d_lp_current_time);
	clCheckError (clStatus, "clSetKernelArg: d_lp_current_time");
	clStatus |= clSetKernelArg(_kernel_initializeSimulator, a++, sizeof(cl_mem), (const void*)&d_event_time);
	clCheckError (clStatus, "clSetKernelArg: d_event_time");
	clStatus |= clSetKernelArg(_kernel_initializeSimulator, a++, sizeof(cl_mem), (const void*)&d_event_lp_number);
	clCheckError (clStatus, "clSetKernelArg: d_event_lp_number");
	clStatus |= clSetKernelArg(_kernel_initializeSimulator, a++, sizeof(cl_mem), (const void*)&d_events_processed);
	clCheckError (clStatus, "clSetKernelArg: d_events_processed");
	clStatus |= clEnqueueNDRangeKernel(clpp_context.clQueue, _kernel_initializeSimulator, 1, NULL, grid_size, block_size, 0, NULL, NULL);
	clCheckError (clStatus, "clEnqueueNDRangeKernel");
	clStatus |= clFinish(clpp_context.clQueue);

	return clStatus;
}

// markNextEventByLP<<<grid_size, block_size>>>(d_event_lp_number.Current(), d_next_event_flag);
cl_int runMarkNextEventByLP (size_t *grid_size, size_t *block_size,
		cl_mem d_event_lp_number, cl_mem d_next_event_flag)
{
	cl_int clStatus;
	unsigned int a = 0;

	cl_kernel _kernel_markNextEventByLP = clCreateKernel(pholdProgram._clProgram, "markNextEventByLP", &clStatus);
	clCheckError (clStatus, "clCreateKernel: _kernel_markNextEventByLP");

	clStatus = clSetKernelArg(_kernel_markNextEventByLP, a++, sizeof (cl_mem), (const void*)&d_event_lp_number);
	clCheckError (clStatus, "clSetKernelArg: d_event_lp_number");
	clStatus |= clSetKernelArg(_kernel_markNextEventByLP, a++, sizeof(cl_mem), (const void*)&d_next_event_flag);
	clCheckError (clStatus, "clSetKernelArg: d_next_event_flag");
	clStatus |= clEnqueueNDRangeKernel(clpp_context.clQueue, _kernel_markNextEventByLP, 1, NULL, grid_size, block_size, 0, NULL, NULL);
	clCheckError (clStatus, "clEnqueueNDRangeKernel");
	clStatus |= clFinish(clpp_context.clQueue);

	return clStatus;
}
//simulatorRun<<<gird_run_size, block_size>>>(d_random_state, d_lp_current_time, d_event_time.Current(), d_event_lp_number.Current(), d_current_lbts, d_events_processed);
cl_int runSimulatorRun (size_t *grid_size, size_t *block_size,
		cl_mem d_random_state, cl_mem d_lp_current_time, cl_mem d_event_time, cl_mem d_event_lp_number, cl_mem d_current_lbts, cl_mem d_events_processed)
{
	cl_int clStatus;
	unsigned int a = 0;

	cl_kernel _kernel_simulatorRun = clCreateKernel(pholdProgram._clProgram, "simulatorRun", &clStatus);
	clCheckError (clStatus, "clCreateKernel: _kernel_simulatorRun");

	clStatus = clSetKernelArg(_kernel_simulatorRun, a++, sizeof (mwc64x_state_t), (const void*)&d_random_state);
	clCheckError (clStatus, "clSetKernelArg: d_random_state");
	clStatus |= clSetKernelArg(_kernel_simulatorRun, a++, sizeof(cl_mem), (const void*)&d_lp_current_time);
	clCheckError (clStatus, "clSetKernelArg: d_lp_current_time");
	clStatus |= clSetKernelArg(_kernel_simulatorRun, a++, sizeof(cl_mem), (const void*)&d_event_time);
	clCheckError (clStatus, "clSetKernelArg: d_event_time");
	clStatus |= clSetKernelArg(_kernel_simulatorRun, a++, sizeof(cl_mem), (const void*)&d_event_lp_number);
	clCheckError (clStatus, "clSetKernelArg: d_event_lp_number");
	clStatus |= clSetKernelArg(_kernel_simulatorRun, a++, sizeof(cl_mem), (const void*)&d_current_lbts);
	clCheckError (clStatus, "clSetKernelArg: d_current_lbts");
	clStatus |= clSetKernelArg(_kernel_simulatorRun, a++, sizeof(cl_mem), (const void*)&d_events_processed);
	clCheckError (clStatus, "clSetKernelArg: d_events_processed");
	clStatus |= clEnqueueNDRangeKernel(clpp_context.clQueue, _kernel_simulatorRun, 1, NULL, grid_size, block_size, 0, NULL, NULL);
	clCheckError (clStatus, "clEnqueueNDRangeKernel");
	clStatus |= clFinish(clpp_context.clQueue);

	return clStatus;
}

int runTest ()
{
	cl_int errNum;

    clpp_context.setup (0, 0);
    assert(pholdProgram.compile (&clpp_context, kernelFileName));

	std::cout << "Device ID: " << clpp_context.clDevice << std::endl;
	std::cout << "Platform ID: " << clpp_context.clPlatform << std::endl;

	//debug files
	std::ofstream currentTime;
	std::ofstream eventList;

	// Symbols are initialized in the .cl file for OpenCL
	int num_lps = 1 << 20;
	size_t block_size[1] = {128};
	// float delay_time = .9f;
	// float lookahead = 4.0f;
	// float local_rate = .9f;
	float stop_time = 60.0f;

	int num_events = 2 * num_lps;

	float                        current_lbts;
	double                       total_start_time;
	double                       total_duration;

	// Allocate device memory
    cl_mem d_events_processed = clCreateBuffer (clpp_context.clContext, CL_MEM_READ_WRITE, sizeof (int) * num_lps, NULL, &errNum);
    clCheckError (errNum, "clCreateBuffer: d_events_processed");

    cl_mem d_lp_current_time = clCreateBuffer (clpp_context.clContext, CL_MEM_READ_WRITE, sizeof (float) * num_lps, NULL, &errNum);
    clCheckError (errNum, "clCreateBuffer: d_lp_current_time");
    cl_mem d_random_state = clCreateBuffer (clpp_context.clContext, CL_MEM_READ_WRITE, sizeof (mwc64x_state_t) * num_lps, NULL, &errNum);
    clCheckError (errNum, "clCreateBuffer: d_random_state");

    // Need to make double buffer
    cl_mem d_event_lp_number = clCreateBuffer (clpp_context.clContext, CL_MEM_READ_WRITE, sizeof (int) * num_lps, NULL, &errNum);
    clCheckError (errNum, "clCreateBuffer: d_event_lp_number");

    // Need to make double buffer
    cl_mem d_event_time = clCreateBuffer (clpp_context.clContext, CL_MEM_READ_WRITE, sizeof (float) * num_events, NULL, &errNum);
    clCheckError (errNum, "clCreateBuffer: d_event_time");

    cl_mem d_current_lbts = clCreateBuffer (clpp_context.clContext, CL_MEM_READ_WRITE, sizeof (float), NULL, &errNum);
    clCheckError (errNum, "clCreateBuffer: d_current_lbts");

    cl_mem d_next_event_flag = clCreateBuffer (clpp_context.clContext, CL_MEM_READ_WRITE, sizeof (unsigned char) * num_events, NULL, &errNum);
    clCheckError (errNum, "clCreateBuffer: d_next_event_flag");
    //better with 32bit value?  Verify this is 8 and then check if better memory coalescing occurs with 32 int

    //INITIALIZE WORK MEMORY

    //work memory for sort
    unsigned int temp_sort_bytes = 64512;
	clppSort_RadixSortGPU RadixSort(&clpp_context, num_events, temp_sort_bytes*8, false);
	cl_mem d_temp_sort = clCreateBuffer (clpp_context.clContext, CL_MEM_READ_WRITE, temp_sort_bytes, NULL, &errNum);
    clCheckError (errNum, "clCreateBuffer: d_temp_sort");

    //work memory for reduce (4096)
    size_t temp_reduce_bytes = 64512;
	clppSort_RadixSortGPU RadixReduce(&clpp_context, num_events, temp_reduce_bytes*8, true);
	cl_mem d_temp_reduce = clCreateBuffer (clpp_context.clContext, CL_MEM_READ_WRITE, temp_reduce_bytes, NULL, &errNum);
    clCheckError (errNum, "clCreateBuffer: d_temp_reduce");

    size_t grid_size[1] = {((num_events + block_size[0] - 1) / block_size[0])};
    size_t grid_run_size[1] = {((num_lps + block_size[0] - 1) / block_size[0])};

    std::cout << "Grid Size: " << grid_size[0] << " Block Size: " << block_size[0] << std::endl;

    cl_int clStatus = runInitializeSimulator (grid_size, block_size,
    		d_random_state, d_lp_current_time, d_event_time, d_event_lp_number, d_events_processed);
	clCheckError (clStatus, "runInitializeSimulator");

	std::cout << "Running simulation..." << std::endl;

	total_start_time = cpuSecond();

    float *local_event_times = (float *)calloc (sizeof(float), num_events);
	while(true)
	{
		clEnqueueReadBuffer(clpp_context.clQueue, d_event_time, CL_TRUE, 0, 4 * num_events, local_event_times, 0, NULL, NULL);
		RadixReduce.pushDatas(local_event_times, num_events);
		RadixReduce.sort();
		RadixReduce.popDatas(local_event_times);
		current_lbts = local_event_times[0];
		std::cout << "Current LBTS: " << current_lbts << std::endl;

		if(current_lbts >= stop_time)
		{
		  break;
		}

//	CudaCheck(cub::DeviceRadixSort::SortPairs(d_temp_sort, temp_sort_bytes, d_event_time, d_event_lp_number, num_events));

//	CudaCheck(cub::DeviceRadixSort::SortPairs(d_temp_sort, temp_sort_bytes, d_event_lp_number, d_event_time, num_events));

		clStatus = runMarkNextEventByLP (grid_size, block_size,
				d_event_lp_number, d_next_event_flag);
		clCheckError (clStatus, "runMarkNextEventByLP");

		clStatus = runSimulatorRun (grid_run_size, block_size,
				d_random_state, d_lp_current_time, d_event_time, d_event_lp_number, d_current_lbts, d_events_processed);
		clCheckError (clStatus, "runSimulatorRun");
	}

	total_duration = cpuSecond() - total_start_time;

	std::cout << "Stats: " << std::endl;

	int* events_processed = (int*)malloc(sizeof(int) * num_lps);
	clEnqueueReadBuffer(clpp_context.clQueue, d_events_processed, CL_TRUE, 0, 4 * num_lps, events_processed, 0, NULL, NULL);

	int total_events_processed = 0;
	for(int i = 0; i < num_lps; ++i)
	{
		total_events_processed += events_processed[i];
	}

	std::cout << "Total Number of Events Processed: " << total_events_processed << std::endl;

	std::cout << "Simulation Run Time: " << total_duration << " seconds." << std::endl;
	std::cout << "The context: " << clpp_context.clContext << std::endl;
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
    shrQAStart(argc, argv);

    // start logs
    shrSetLogFileName ("oclDeviceQuery.txt");
    shrLog("%s Starting...\n\n", argv[0]);

    bool bPassed = true;
    std::string sProfileString = "oclDeviceQuery, Platform Name = ";
    // Get OpenCL platform ID for NVIDIA if available, otherwise default
    shrLog("OpenCL SW Info:\n\n");
    char cBuffer[1024];
    cl_platform_id clSelectedPlatformID = NULL;
    cl_int ciErrNum = oclGetPlatformID (&clSelectedPlatformID);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Get OpenCL platform name and version
    ciErrNum = clGetPlatformInfo (clSelectedPlatformID, CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer, NULL);
    if (ciErrNum == CL_SUCCESS)
    {
        shrLog(" CL_PLATFORM_NAME: \t%s\n", cBuffer);
        sProfileString += cBuffer;
    }
    else
    {
        shrLog(" Error %i in clGetPlatformInfo Call !!!\n\n", ciErrNum);
        bPassed = false;
    }
    sProfileString += ", Platform Version = ";

    ciErrNum = clGetPlatformInfo (clSelectedPlatformID, CL_PLATFORM_VERSION, sizeof(cBuffer), cBuffer, NULL);
    if (ciErrNum == CL_SUCCESS)
    {
        shrLog(" CL_PLATFORM_VERSION: \t%s\n", cBuffer);
        sProfileString += cBuffer;
    }
    else
    {
        shrLog(" Error %i in clGetPlatformInfo Call !!!\n\n", ciErrNum);
        bPassed = false;
    }
    sProfileString += ", SDK Revision = ";

    // Log OpenCL SDK Revision #
    shrLog(" OpenCL SDK Revision: \t%s\n\n\n", OCL_SDKREVISION);
    sProfileString += OCL_SDKREVISION;
    sProfileString += ", NumDevs = ";

    // Get and log OpenCL device info
    cl_uint ciDeviceCount;
    cl_device_id *devices;
    shrLog("OpenCL Device Info:\n\n");
    ciErrNum = clGetDeviceIDs (clSelectedPlatformID, CL_DEVICE_TYPE_GPU, 0, NULL, &ciDeviceCount);
    // check for 0 devices found or errors...
    if (ciDeviceCount == 0)
    {
        shrLog(" No devices found supporting OpenCL (return code %i)\n\n", ciErrNum);
        bPassed = false;
        sProfileString += "0";
    }
    else if (ciErrNum != CL_SUCCESS)
    {
        shrLog(" Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
        bPassed = false;
    }
    else
    {
        // Get and log the OpenCL device ID's
        shrLog(" %u devices found supporting OpenCL:\n\n", ciDeviceCount);
        char cTemp[2];
        #ifdef WIN32
            sprintf_s(cTemp, 2*sizeof(char), "%u", ciDeviceCount);
        #else
            sprintf(cTemp, "%u", ciDeviceCount);
        #endif
        sProfileString += cTemp;
        if ((devices = (cl_device_id*)malloc(sizeof(cl_device_id) * ciDeviceCount)) == NULL)
        {
           shrLog(" Failed to allocate memory for devices !!!\n\n");
           bPassed = false;
        }
        ciErrNum = clGetDeviceIDs (clSelectedPlatformID, CL_DEVICE_TYPE_GPU, ciDeviceCount, devices, &ciDeviceCount);
        if (ciErrNum == CL_SUCCESS)
        {
            //Create a context for the devices
            cl_context cxGPUContext = clCreateContext(0, ciDeviceCount, devices, NULL, NULL, &ciErrNum);
            if (ciErrNum != CL_SUCCESS)
            {
                shrLog("Error %i in clCreateContext call !!!\n\n", ciErrNum);
                bPassed = false;
            }
            else
            {
                // show info for each device in the context
                for(unsigned int i = 0; i < ciDeviceCount; ++i )
                {
                    shrLog(" ---------------------------------\n");
                    clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
                    shrLog(" Device %s\n", cBuffer);
                    shrLog(" ---------------------------------\n");

                    // Found the device, time to actually work
                    runTest ();
                }
                shrLog("\n");
            }
        }
        else
        {
            shrLog(" Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
            bPassed = false;
        }
    }

    // finish
    shrQAFinishExit(argc, (const char **)argv, (bPassed ? QA_PASSED : QA_FAILED) );
}
