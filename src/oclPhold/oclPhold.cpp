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

#include "clpp/clpp.h"

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

int runTest (cl_context gpuContext, cl_device_id device, clppContext clpp_context)
{
	cl_int errNum;

	std::cout << "Device ID: " << clpp_context.clDevice << std::endl;
	std::cout << "Platform ID: " << clpp_context.clPlatform << std::endl;
    cl_command_queue cqCommandQueue = clCreateCommandQueue(gpuContext, device, 0, &errNum);
    oclCheckError(errNum, CL_SUCCESS);

	//debug files
	std::ofstream currentTime;
	std::ofstream eventList;

	// Symbols are initialized in the .cl file for OpenCL
	int num_lps = 1 << 20;
	// int block_size = 128;
	// float delay_time = .9f;
	// float lookahead = 4.0f;
	// float local_rate = .9f;
	// float stop_time = 60.0f;

	int num_events = 2 * num_lps;

	float                        current_lbts;
	double                       total_start_time;
	double                       total_duration;

	// Allocate device memory
    cl_mem d_events_processed = clCreateBuffer (gpuContext, CL_MEM_READ_WRITE, sizeof (int) * num_lps, NULL, &errNum);
    clCheckError (errNum, "clCreateBuffer: d_events_processed");

    cl_mem d_lp_current_time = clCreateBuffer (gpuContext, CL_MEM_READ_WRITE, sizeof (float) * num_lps, NULL, &errNum);
    clCheckError (errNum, "clCreateBuffer: d_lp_current_time");
    cl_mem d_random_state = clCreateBuffer (gpuContext, CL_MEM_READ_WRITE, sizeof (mwc64x_state_t) * num_lps, NULL, &errNum);
    clCheckError (errNum, "clCreateBuffer: d_random_state");

    // Need to make double buffer
    cl_mem d_event_lp_number = clCreateBuffer (gpuContext, CL_MEM_READ_WRITE, sizeof (int) * num_lps, NULL, &errNum);
    clCheckError (errNum, "clCreateBuffer: d_event_lp_number");

    // Need to make double buffer
    cl_mem d_event_time = clCreateBuffer (gpuContext, CL_MEM_READ_WRITE, sizeof (float) * num_events, NULL, &errNum);
    clCheckError (errNum, "clCreateBuffer: d_event_time");

    cl_mem d_current_lbts = clCreateBuffer (gpuContext, CL_MEM_READ_WRITE, sizeof (float), NULL, &errNum);
    clCheckError (errNum, "clCreateBuffer: d_current_lbts");

    cl_mem d_next_event_flag = clCreateBuffer (gpuContext, CL_MEM_READ_WRITE, sizeof (unsigned char) * num_events, NULL, &errNum);
    clCheckError (errNum, "clCreateBuffer: d_next_event_flag");
    //better with 32bit value?  Verify this is 8 and then check if better memory coalescing occurs with 32 int

    cl_mem d_partition_int = clCreateBuffer (gpuContext, CL_MEM_READ_WRITE, sizeof (int), NULL, &errNum);
    clCheckError (errNum, "clCreateBuffer: d_partition_int");
    cl_mem d_partition_float = clCreateBuffer (gpuContext, CL_MEM_READ_WRITE, sizeof (float), NULL, &errNum);
    clCheckError (errNum, "clCreateBuffer: d_partition_float");

    //INITIALIZE WORK MEMORY

    //work memory for sort

    size_t 	temp_sort_bytes = 0;
    void*	  d_temp_sort     = NULL;

//	  clppSort_RadixSortGPU RadixSort(&clpp_context, num_events, bits, false);
//    RadixSort radixSortObject(gpuContext, cqCommandQueue, num_events, "/home/jared/CLPhold/oclPhold/", num_lps, true);
//    radixSortObject.sort(d_temp_sort, num_events, temp_sort_bytes);
//    cl_mem d_partition_float = clCreateBuffer (gpuContext, CL_MEM_READ_WRITE, sizeof (float), NULL, &errNum);
//    clCheckError (errNum, "clCreateBuffer: d_partition_float");
//    CudaCheck(cudaMalloc(&d_temp_sort, temp_sort_bytes));

    //work memory for reduce

//    size_t temp_reduce_bytes = 0;
//    void*  d_temp_reduce     = NULL;
//
//    CudaCheck(cub::DeviceReduce::Min(d_temp_reduce, temp_reduce_bytes, d_event_time.Current(), d_current_lbts, num_events));
//    CudaCheck(cudaMalloc(&d_temp_reduce, temp_reduce_bytes));
//
//    //work memory for the partition
//
//    size_t temp_partition_float_bytes = 0;
//    void*  d_temp_partition_float     = NULL;
//
//    CudaCheck(cub::DevicePartition::Flagged(d_temp_partition_float, temp_partition_float_bytes, d_event_time.Current(), d_next_event_flag, d_event_time.Current(), d_partition_float, num_events));
//
//    size_t temp_parition_int_bytes    = 0;
//    void*  d_temp_partition_int       = NULL;
//
//    CudaCheck(cub::DevicePartition::Flagged(d_temp_partition_int, temp_parition_int_bytes, d_event_lp_number.Current(), d_next_event_flag, d_event_lp_number.Current(), d_partition_int, num_events));
//
//    CudaCheck(cudaMalloc(&d_temp_partition_float, temp_partition_float_bytes));
//    CudaCheck(cudaMalloc(&d_temp_partition_int, temp_parition_int_bytes));

	std::cout << "The context: " << gpuContext << std::endl;
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
                    clppContext clpp_context;
                    clpp_context.setup (0, 0);
                    runTest (cxGPUContext, devices[i], clpp_context);
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
