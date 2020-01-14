
/**
*@copyright     All rights are reserved, this code/project is not Open Source or Free
*@author        Nathaniel Crossman (U00828694)
*@email		 crossman.4@wright.edu
*
*@Professor     Meilin Liu
*@Course_Number CS 4370/6370-90
*@Date			 11 23, 2019
*
Project Name:  CrossmanNathaniel_Project3
•	Work Efficient Parallel Reduction (Works!)
o	Every Part of this project works Completely.
o	It works for all three test cases (including bonus one)
o	I use a recursion function to get the reduction Sum
	I dynamically allocate shared memory (i.e extern)
o	Additionally, have a dynamic parallelism function (worth extra bonus points)
	I used static allocate shared memory
o	Graders Note: did bonus question and have dynamic parallelism which is also a bonus question.
•	Work Efficient Parallel Prefix Sum (Works!)
o	Elements of size 2048 works
o	Elements of size 131072 works
o	Elements of size 1048576 works
o	Elements of size 16777216 does not work
	I couldn’t get the recursive algorithm to work.


*
*
*/
// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string.h>

// CUDA runtime
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
//#include <helper_functions.h>
#include <cuda_runtime_api.h>
//#include <helper_cuda.h>

//For right now we will set Block size to a fixed value.
#define BLOCK_SIZE 1024
#define BLOCK_SIZE_ 1024
__device__ int blockID = 0; 

/*
getAnyErrors
This F() is a better way of showing if any errors happen.
I removed all inline error checking.
*/
#define getAnyErrors(msg) \
    do { \
        cudaError_t myErrorList = cudaGetLastError(); \
        if (myErrorList != cudaSuccess) { \
			printf(" Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(myErrorList),__FILE__, __LINE__ );\
        } \
    } while (0)


/*
Work Efficient Parallel Reduction kernel used for recursion
*/
__global__ void reductionSum(int *d_array_data_input, int *d_array_data_input_temp, int n) {
	extern __shared__ int partialSum[];
	unsigned int tid = threadIdx.x;
	unsigned int start = 2 * blockIdx.x*blockDim.x;
	/*
	This is only need if are data is not a 2 ^
	we are pading kind of 0.
	0 + anything = anything
	*/
	partialSum[threadIdx.x] = 0;
	partialSum[threadIdx.x + blockDim.x] = 0;
	__syncthreads(); //Needed to make sure thread all pad zero

	partialSum[tid] = (tid < n) ? d_array_data_input[start + tid] : 0;
	partialSum[blockDim.x + tid] = d_array_data_input[start + blockDim.x + tid];

	__syncthreads(); // Need to makw sure all shared memmary get a value

	for (unsigned int stride = blockDim.x; stride > 0; stride >>= 1) {
		if (tid < stride){
			partialSum[tid] += partialSum[tid + stride];
		}
		__syncthreads(); //sum do for thread.. tree
	}

	// write result for this block to global mem
	if (tid == 0) {
		d_array_data_input_temp[blockIdx.x] = partialSum[0];
	
	}
}

/*
Work Efficient Parallel Reduction Kernel with Dynamic Parallelism 
Reference:
https://stackoverflow.com/questions/30779451/understanding-dynamic-parallelism-in-cuda
https://devtalk.nvidia.com/default/topic/1028861/learning-by-coding-recursive-sum-using-dynamic-parallelism/
https://devtalk.nvidia.com/default/topic/999093/even-without-sync-a-parallel-reduction-sum-using-dynamic-parallelism-works-/
*/
__global__ void dynamic_parallelism_reduction_SumD(int*d_array_data_input, int n) {
	__shared__ int partialSum[BLOCK_SIZE];
	int index = blockDim.x * blockIdx.x +threadIdx.x;
	int tid = threadIdx.x;

	if (gridDim.x == 1) {  
		if ((tid < n))
			partialSum[tid] = d_array_data_input[tid];
		else
			partialSum[tid] = 0;
		
		if ((tid + BLOCK_SIZE) < n) 
			partialSum[tid] += d_array_data_input[tid + BLOCK_SIZE];

		//nsigned int stride = blockDim.x this is to big...
		for (unsigned int stride = blockDim.x /2 ; stride > 0; stride >>= 1) {
			__syncthreads();
			if (tid < stride) {
				partialSum[tid] += partialSum[tid + stride];
			}

		}
		// write result for this block to global mem
		if (tid == 0) {
			d_array_data_input[blockIdx.x] = partialSum[0];
		}
	} else { 
		int grid_size =  (n % 2) ? (n / 2 + 1) : (n / 2);
		// We need to check if we should add.. we do not want to re-add thought
		if ((index + grid_size) < n)
			d_array_data_input[index] += d_array_data_input[index + grid_size];
		//All threads must do the above
		__syncthreads();
		if (tid == 0) {
			//https://stackoverflow.com/questions/45674907/atomic-block-in-cuda
			//Need to know when a block is all done.. 
			int amILastBlock = atomicAdd(&blockID, 1); 
			if (amILastBlock == gridDim.x - 1) {   
				int myDynamicGrid = (grid_size & 1) + grid_size;
				dim3 dimGrid(((myDynamicGrid / 1) + BLOCK_SIZE - 1) / BLOCK_SIZE);
				blockID = 0;
				__syncthreads();
				dynamic_parallelism_reduction_SumD << <dimGrid, BLOCK_SIZE >> > (d_array_data_input, grid_size);
				getAnyErrors("Error in dynamic_parallelism_reduction_SumD");
			}
		}
	}
}


__global__ void work_efficient_scan_kernel(int *d_array_data_input, int *d_array_data_output, int n, int* auxiliaryAray) {
	__shared__ int scan_array[2* BLOCK_SIZE_];
	unsigned int tid = threadIdx.x;
	unsigned int start = 2 * blockIdx.x*blockDim.x;
	
	scan_array[threadIdx.x] = 0;
	scan_array[threadIdx.x + blockDim.x] = 0;
	//__syncthreads(); //Needed to make sure thread all pad zero
	//s_array[t] = x[start + t];
	//s_array[blockDim.x + t] = x[start + blockDim.x + t];

	//This is problem better then doing pading and syncthreads (tid < n)
	//scan_array[tid] = ((start + tid) < n) ? d_array_data_input[start + tid] : 0;

	scan_array[tid] = ((start + tid) < n) ? d_array_data_input[start + tid] : 0;
	scan_array[tid + blockDim.x] = ((start + blockDim.x + tid) < n) ? d_array_data_input[start + blockDim.x + tid] : 0;
	//scan_array[tid + blockDim.x] = d_array_data_input[start + blockDim.x + tid];
	__syncthreads(); //We need to pad and copy gl to shared


	int stride = 1;
	while (stride <= blockDim.x)
	{
		int index = (threadIdx.x + 1)*stride * 2 - 1;
		if (index < 2 * blockDim.x) {
			scan_array[index] += scan_array[index - stride];
		}
		stride = stride * 2;
		__syncthreads();
	}

	//Down
	//stride = BLOCK_SIZE >> 2;
	stride = blockDim.x / 2;
	while (stride > 0){
		int index = (threadIdx.x + 1)*stride * 2 - 1;
		if (index + stride <  2 * blockDim.x){
			scan_array[index + stride] += scan_array[index];
		}
		stride = stride / 2;//
		__syncthreads();
	}
	__syncthreads();
	
	if (tid == 0 && auxiliaryAray) {
		//printf(" auxiliaryAray scan_array[2 * BLOCK_SIZE - 1] %d \n", scan_array[2 * blockDim.x - 1]);
		auxiliaryAray[blockIdx.x] = scan_array[2 * blockDim.x - 1];
	}
	
	//d_array_data_output[start + tid] = scan_array[tid]; 
	//d_array_data_output[start + BLOCK_SIZE + tid] = scan_array[BLOCK_SIZE + tid];
		if (start + tid < n) {
		d_array_data_output[start + tid] = scan_array[tid];
	}
	if (start + blockDim.x + tid < n){
		d_array_data_output[start + blockDim.x + tid] = scan_array[tid + blockDim.x];
	}
	
}


/*
after first kernel runs d_array_data_output holds all adjusted elements.
I need the last value of each sub array.
What I need is this.
d_array_data_output = b0[2,4,6] b1[8,10,12] ....
auxiliary_array [6,12]

So I'm going to do it in a number kernel..
Do need got working in main kernel
*/
__global__ void getAuxiliaryAray(int * d_array_data_output, int * auxiliaryAray, int n) {
	int tid = threadIdx.x; //Need to help get ending block Position
	int blockId = ((tid +1 ) * (BLOCK_SIZE_ * 2)) - 1; // we need to offset the blcok..  0 to N not 1 to N
	if (n > blockId) {
		auxiliaryAray[tid] = d_array_data_output[blockId]; //d_array_data_output[blockId] gives ending value.
	}
}

__global__ void AddScannedBlocksSum(int* d_finalAnswerFromGPU, int *d_array_data_output, int n) {
	//A single value array for all blocks to add to..
	__shared__ int tempValue[1];
	//int index = threadIdx.x + blockIdx.x * blockDim.x ;
	int tid = threadIdx.x;
	int blId = blockIdx.x;
	int start = 2 * blockIdx.x * BLOCK_SIZE_;

	if (tid == 0) {
		if (blId != 0)
			tempValue[0] = d_finalAnswerFromGPU[blId-1];
		//printf("tempValue[0]; %d \n ", tempValue[0];)
	}
	__syncthreads();

	if (blId != 0) {
		//Need have a thread that sums two location as 2048..
		//d_array_data_output[tid] = 
		//d_array_data_output[blockDim.x + tid] = tempValue[0];
		//d_array_data_output[index] += tempValue[0];
		//printf("d_array_data_output[index] += tempValue[0]; %d \n ", d_array_data_output[index] += tempValue[0];)
		//Need have a thread that sums two location as 2048..
	if (blockIdx.x) {
		if (start + tid < n)
			d_array_data_output[start + tid] += tempValue[0];
		if (start + BLOCK_SIZE_ + tid < n)
			d_array_data_output[start + BLOCK_SIZE_ + tid] += tempValue[0];
	}
	}


}

//below is all prototypes
void reductionMain();
int reductionSumWith_dynamic_parallelism(int* d_array_data_input, int sizeOfN, float &secTotal);
int reductionSum_Recursion_F(int* d_array_data_input, int sizeOfN, float &secTotal);

//
void prefixSumMain();
void PrefixSum(int *y, int *x, int n);

void PrefixSum_Recursion_F(int* d_array_data_input, int *d_array_data_output, int sizeOfN, float &secTotal);
//-------------------------------
int SumReduction(int *x, int N);
int runCPU_R(int * h_array_data_input_CPU, int* h_array_data_forPrinting, int booleanValue, int sizeOfN, double &cpu_time);
//Only compares to host with de results
void verify(int resultOfHost, int resultofGpu);
void PrefixSum(int *input, int *out, int n);

//Helper f
int menuTypeOfKernel();
int menuShow();
void mainSwitch(int option);
void cls();
int debugOn();
void getBlockSize(int &blocksize);
void getSizeOfN(int &size);

void printf_matrix(int *A, int size);
void initF(int* arrayValue, int size);
void setDataValues(int* arrayValue, int size);
//Above is all prototypes

int main()
{	
	// This will pick the best possible CUDA capable device, otherwise
	// override the device ID based on input provided at the command line
	//int dev = findCudaDevice(argc, (const char **)argv);
	while (true) {
		mainSwitch(menuShow());
		printf("\n");
	}
	return 0;
}

int menuShow() {
	int hold;
	do {
		printf("1. Reduction Sum  \n");
		printf("2. Parallel Prefix Sum  \n");
		printf("3. Quit\n");
		printf("---------------------------------------\n");
		printf("Enter Choice: ");
		scanf("%d", &hold);

		if (hold < 1 || hold > 3) {
			cls();
		}
	} while (hold < 1 || hold > 3);
	return hold;
}

void cls() {
	for (int i = 0; i < 30; ++i)
			printf("\n");
	system("@cls||clear");
}

void mainSwitch(int option) {
	switch (option) {
	case 1:
		reductionMain();
		break;
	case 2:
		prefixSumMain();
		
		break;
	case 3:
		exit(0);
		break;
	}
}



void getSizeOfN(int &size) {
	printf("Please specify the size of N (N = #elements) \n");
	printf("For example, you could enter 1024 and the size would be (1024)\n");
	printf("OR, you could enter 131,072‬ and the size would be 131,072‬\n");
	printf("Enter Size:");
	scanf("%d", &size);
	cls();
}

void getBlockSize(int &blocksize) {
	printf("Please specify your Thread block\n");
	printf("For example, you could enter 512\n");
	printf("Enter Block Size:");
	scanf("%d", &blocksize);
	cls();
}

void printf_matrix(int *A, int size) {
	int i;
	for (i = 0; i < size; ++i) {
		printf("%d \t", A[i]);
	}
	printf("\n");
}

void verify(int resultOfHost, int resultofGpu) {
	if (resultOfHost == resultofGpu)
		printf("The Test Passed\n");
	else
		printf("The Test failed\n");
}

int debugOn() {
	int hold;
	do {
		printf("\nRun in debug mode?\n");
		printf("Debug mode prints out alot of helpful info,\nbut it can takes a long time with big matrixes\n");
		printf("Enter 1 for Yes and 0 for No:");
		scanf("%d", &hold);
		if (hold < 0 || hold > 1) {
			cls();
		}
	} while (hold < 0 || hold > 1);
	cls();
	return hold;
}

int menuTypeOfKernel() {
	int hold;
	do {
		printf("1. Work Efficient Parallel Reduction Sum: Recursion Function  \n");
		printf("2. Work Efficient Parallel Reduction Sum: Dynamic parallelism  \n");
		printf("---------------------------------------\n");
		printf("Enter Choice: ");
		scanf("%d", &hold);

		if (hold < 1 || hold > 2) {
			cls();
		}
	} while (hold < 1 || hold > 2);
	return hold;
}

/*
To figure out how Dynamic parallelism worked I referenced
https://devtalk.nvidia.com/default/topic/999093/even-without-sync-a-parallel-reduction-sum-using-dynamic-parallelism-works-/
His approach does not use shared memory and is substantially different compared..
However, I did look at his code so I am a Referencing it.
*/
int reductionSumWith_dynamic_parallelism(int* d_array_data_input, int sizeOfN, float &secTotal) {
	size_t size_my = sizeOfN * sizeof(int);
	int *h_final_array_data;
	h_final_array_data = (int *)malloc(size_my);
	//dim3 DimGrid((sizeOfN - 1) / (BLOCK_SIZE * 2) + 1)
	int Grid = (sizeOfN % 2) ? (sizeOfN / 2 + 1) : (sizeOfN / 2);
	int myDynamicGrid = (Grid + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(myDynamicGrid);

	printf("Kernel Reduction Sum Using Dynamic Parallelism\n");
	printf("-------------------------------------------------------------------\n");
	printf("Kernel Parameter\n");
	printf("dimBlock :%d\n", BLOCK_SIZE);
	printf("dimGrid :%d\n", myDynamicGrid);
	printf("Grid :%d\n", Grid);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//Remove
	dynamic_parallelism_reduction_SumD << <myDynamicGrid, BLOCK_SIZE >> >(d_array_data_input, sizeOfN);
	getAnyErrors("when kernel is running");
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&secTotal, start, stop);
	//Delete
	cudaEventDestroy(start);
	cudaDeviceSynchronize();
	cudaEventDestroy(stop);
	cudaDeviceSynchronize();


	cudaMemcpy(h_final_array_data, d_array_data_input, sizeof(int), cudaMemcpyDeviceToHost);
	return h_final_array_data[0];
}


/***
*This sum reduction Recursion function was Ungodly challenging to figure out.
*The Dynamic parallelism was a lot easier to write.
*
**/
int reductionSum_Recursion_F(int* d_array_data_input, int sizeOfN, float &secTotal) {
	int blocksize						= BLOCK_SIZE;		//Block size is Must be manually changed
	int myDynamicGrid					= 0;				//set based on the size of N
	int totalAllowedElementsInABlock	= (blocksize * 2);	//the total elements in a block
	int finalAnswerFromGPU				= 0;				//Hold the out come
	float temp_time						= 0;
	int *d_finalAnswerFromGPU;
	int *d_array_data_block_final_out;
	int *d_array_data_input_temp;

	//myDynamicGrid = (int)ceil((((double)sizeOfN) / double(blocksize))); wrong!!!
	myDynamicGrid = ceil((sizeOfN - 1) / (BLOCK_SIZE * 2) + 1);
	size_t size_block = myDynamicGrid * sizeof(int);
	


	// This does not work... Must use cudaMemset.. fucking bug.....
	//memset(d_array_data_block_final_out, 0, size_block); 
	//cudaMalloc((void **)(&d_array_data_block_final_out), size_block); sizeD
	cudaMalloc((void **)(&d_array_data_block_final_out), size_block);
	getAnyErrors("Allocating memory for d_array_data_block_final_out ");
	cudaMemset(d_array_data_block_final_out, 0, size_block);
	getAnyErrors("setting values in d_array_data_block_final_out to 0");

	//Crucial kernel configurations
	dim3 dimBlock(blocksize);
	dim3 dimGrid(myDynamicGrid);
	//Not sure if this is right ((1024 *2) = Potential maximum elements ) * int size 
	size_t sharedSize = totalAllowedElementsInABlock *sizeof(int);

	printf("Kernel Reduction Sum Using Recursion\n");
	printf("-------------------------------------------------------------------\n");
	printf("Kernel Parameter\n");
	printf("dimBlock :%d\n", blocksize);
	printf("dimGrid :%d\n", myDynamicGrid);
	printf("sharedSize :%zd\n", sharedSize);
	

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	reductionSum << < dimGrid, dimBlock, sharedSize >> >(d_array_data_input, d_array_data_block_final_out, sizeOfN);
	getAnyErrors("when kernel is running");
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&temp_time, start, stop);
	secTotal = secTotal + temp_time;
	//Delete
	cudaEventDestroy(start);
	cudaDeviceSynchronize();
	cudaEventDestroy(stop);
	cudaDeviceSynchronize();
	
	//we got first answer from kernal.. we need to see if we are done 
	/*
	1 call kernal
	see if number of elments is greater then 1024*2. 
	if it is call funtion again
	if not sum results. 
	myDynamicGrid = sizeOfN
	*/
	if (totalAllowedElementsInABlock >= myDynamicGrid){
		int numThreads = myDynamicGrid / 2;
		if (numThreads > 0) {
			cudaMalloc((void **)(&d_finalAnswerFromGPU), size_block);
			getAnyErrors("Copying int_sum (d_finalAnswerFromGPU) data from host to deviced ");
			cudaMemset(d_finalAnswerFromGPU, 0, sizeof(int));
			getAnyErrors("setting values in d_finalAnswerFromGPU to 0");
			//Bug not sizeOfN!! has to be the d_array_data_block_final_out size
			//Bug Double in final out.. fixed stop calling full gird.
			dimBlock.x = (numThreads < BLOCK_SIZE) ? numThreads : BLOCK_SIZE;
			dimGrid.x = (numThreads + dimBlock.x - 1) / dimBlock.x;

			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);
			reductionSum << <dimGrid, dimBlock, sharedSize >> >(d_array_data_block_final_out, d_finalAnswerFromGPU, myDynamicGrid);
			cudaEventRecord(stop, 0);
			cudaDeviceSynchronize();
			cudaEventSynchronize(stop);
			cudaDeviceSynchronize();
			cudaEventElapsedTime(&temp_time, start, stop);
			secTotal = secTotal + temp_time;
			cudaEventDestroy(start);
			cudaDeviceSynchronize();
			cudaEventDestroy(stop);
			cudaDeviceSynchronize();
			getAnyErrors("summing of total in kernel is running");

			cudaMemcpy(&finalAnswerFromGPU, d_finalAnswerFromGPU, sizeof(int), cudaMemcpyDeviceToHost);
			getAnyErrors("Copying array data from host to deviced ");
			cudaFree(d_finalAnswerFromGPU);
		}else {

			cudaMemcpy(&finalAnswerFromGPU, d_array_data_block_final_out, sizeof(int), cudaMemcpyDeviceToHost);
			getAnyErrors("Copying array data from host to deviced ");
		}
	}else {
		//Allocate memory for device d_array_data_input_temp
		 cudaMalloc((void **)(&d_array_data_input_temp), size_block);
		 getAnyErrors("Allocating memory for d_array_data_input_temp ");
		 cudaMemcpy(d_array_data_input_temp, d_array_data_block_final_out, size_block, cudaMemcpyDeviceToDevice); 		 //copy next part of data from d to Device
		 getAnyErrors("Copying array data from host to deviced ");
		 //R_time
		 finalAnswerFromGPU = reductionSum_Recursion_F(d_array_data_input_temp, myDynamicGrid, secTotal);
		 cudaFree(d_array_data_input_temp);
	}

	cudaFree(d_array_data_block_final_out);
	return finalAnswerFromGPU;
}

/*
Driver for both
*/
void reductionMain() {
	int sizeOfN = 0;			// This is the number of Elements present in the rates.. end user-specified perimeter
	//Retrieve finals reduction some value
	int finalAnswerFromCPU = 0;
	int finalAnswerFromGPU = 0;
	float secTotal = 0.0f;
	double cpu_time = 0.0f;

	int *h_array_data_input_CPU;
	int *h_array_data_input_GPU;
	int *h_array_data_forPrinting;
	int *d_array_data_input;

	int booleanValue = debugOn();

	getSizeOfN(sizeOfN);
	//getBlockSize(blocksize); not used as of right now.

	printf("ElementSize: %d \nSize of Thread block: %d", sizeOfN, BLOCK_SIZE);
	printf("\n\n");
	//set total size


	//The size of all elements
	size_t dsize = sizeOfN * sizeof(int);
	//size_t dsize_bl = blocksize * sizeof(int);

	//Allocate memory for matrices on host
	h_array_data_input_CPU		= (int*)malloc(dsize); //Run in CPU
	h_array_data_input_GPU		= (int*)malloc(dsize); // Run in GPU
	h_array_data_forPrinting	= (int*)malloc(dsize); // For displaying the data
	
	//h_temp						= (int*)malloc(dsize); // For displaying the data
	//d_array_data_input_temp		= (int*)malloc(dsize);	//date in GPU second run.
	//d_array_data_final_out		= (int*)malloc(dsize_bl); // used in last run of data

	//For testing making sure all array values have 0
	memset(h_array_data_input_CPU, 0, dsize); //For CPU
	memset(h_array_data_input_GPU, 0, dsize); // for GPU
	memset(h_array_data_forPrinting, 0, dsize);

	//memset(d_array_data_input_temp, 0, dsize);
	//memset(d_array_data_final_out, 0, dsize_bl);
	//memset(h_temp, 0, dsize);
	
	if (h_array_data_input_CPU == NULL || h_array_data_input_GPU == NULL 
		|| h_array_data_forPrinting == NULL) {
		printf("Failed to allocate host matrix C!\n");
	}

	//This is for testing.
	//initF(h_array_data_input_CPU, sizeOfN);
	//initF(h_array_data_input_GPU, sizeOfN);
	//initF(h_array_data_forPrinting, sizeOfN);

	setDataValues(h_array_data_input_CPU, sizeOfN);
	setDataValues(h_array_data_input_GPU, sizeOfN);
	setDataValues(h_array_data_forPrinting, sizeOfN);

	finalAnswerFromCPU = runCPU_R(h_array_data_input_CPU, h_array_data_forPrinting, booleanValue, sizeOfN, cpu_time);

	printf("\nThe results of CPU Multiplication:%d\n", finalAnswerFromCPU);
	printf("CPU is do working....\n\n");
	printf("***********************************************************************\n");
	//Allocate memory for device d_array_data_input
	cudaMalloc((void **)(&d_array_data_input), dsize);
	getAnyErrors("Allocating memory for d_array_data_input ");
	//copy the data from Host to Device
	cudaMemcpy(d_array_data_input, h_array_data_input_GPU, dsize, cudaMemcpyHostToDevice);
	getAnyErrors("Copying array data from host to deviced ");

	//Allocate memory for device d_array_data_input_temp
	//cudaMalloc((void **)(&d_array_data_input_temp), dsize);
	//getAnyErrors("Allocating memory for d_array_data_input_temp ");
	
	////Allocate memory for device d_array_data_final_out
	//cudaMalloc((void **)(&d_array_data_final_out), dsize_bl);
	//getAnyErrors("Allocating memory for d_array_data_final_out ");
	////will need to change
	//dim3 dimBlock(blocksize);
	//int hold = ceil(((double)sizeOfN) / dimBlock.x);
	//dim3 dimGrid(grid);

	//size_t sharedSize = (blocksize *2) * sizeof(int);

	//printf("dimBlock :%d\n", blocksize);
	//printf("dimGrid :%d\n", grid);
	//printf("sharedSize :%d\n", sharedSize);
	switch (menuTypeOfKernel()) {
	case 1:
		finalAnswerFromGPU = reductionSum_Recursion_F(d_array_data_input, sizeOfN, secTotal);
		break;
	case 2:
		finalAnswerFromGPU = reductionSumWith_dynamic_parallelism(d_array_data_input, sizeOfN, secTotal);
		break;
	}
	
	//  when every thread can finish all elements
	//cudaMemcpy(h_temp, d_array_data_input_temp, dsize, cudaMemcpyDeviceToHost);
	//getAnyErrors("get data back again from gpc");
	
	// Copy result from device to host, only when every thread can finish all elements
	//cudaMemcpy(&finalAnswerFromGPU, d_array_data_out, sizeof(int), cudaMemcpyDeviceToHost);
	

	printf("GPU done GPU\n");
	if (booleanValue) {
		printf_matrix(h_array_data_forPrinting, sizeOfN);
		
	}
	printf("\nThe results of GPU:%d\n", finalAnswerFromGPU);

	printf("\nVerifying\n");
	verify(finalAnswerFromCPU,finalAnswerFromGPU);

	cudaFree(d_array_data_input);

	//Clean up memory
	free(h_array_data_input_CPU);
	free(h_array_data_input_GPU);
	free(h_array_data_forPrinting);

	printf("Execution Time for GPU: %.5f ms\n", secTotal);
	printf("Execution Time for CPU: %.5f ms\n", cpu_time);
	printf("Speedup : %.5f ms\n", cpu_time/ secTotal);

}


//This will only return the final values of the reduction process
// This was used for testing 
void initF(int* arrayValue, int size){
	for (int i = 0; i < size; ++i)
		arrayValue[i] = i;
}

//Set array to values
void setDataValues(int* arrayValue, int size)
{
	int init = 1325;
	int i = 0;
	for (; i < size; ++i) {
		init = 3125 * init % 65521;
		arrayValue[i] = (init - 32768) / 16384;
	}
}

int SumReduction(int *x, int N) {
	for (int i = 1; i < N; i++) {
		x[0] = x[0] + x[i];
	}
	return x[0];
}

int runCPU_R(int * h_array_data_input_CPU,int* h_array_data_forPrinting, int booleanValue, int sizeOfN, double &cpu_time) {
	int finalAnswerFromCPU = 0;
	printf("CPU doing work....\n");
	//matrix mul on host 
	clock_t startTime, endTime;

	startTime = clock();

	//CPU function
	finalAnswerFromCPU = SumReduction(h_array_data_input_CPU, sizeOfN);

	endTime = clock();
	cpu_time = ((double)(endTime - startTime))* 1000.0 / CLOCKS_PER_SEC;
	//-----------------------------------------------------------

	if (booleanValue) {
		printf_matrix(h_array_data_forPrinting, sizeOfN);
	}
	return finalAnswerFromCPU;
}


/*
New Code for PrefixSum
*/
//CPU function
void PrefixSum(int *input, int *out, int n) {
	out[0] = input[0];
	for (int i = 1; i < n; ++i) {
		out[i] = out[i - 1] + input[i];
	}
}

//Check answer
void verify(int *x, int *y, int n) {
	for (int i = 0; i< n; i++) {
		if (x[i] != y[i]) {
			printf("TEST FAILED\n");
			return;
		}
	}
	printf("TEST PASSED \n");
}


void prefixSumMain() {
	int sizeOfN = 0;
	float secTotal = 0.0f;
	double cpu_time = 0.0f;
	int blocksize = BLOCK_SIZE_;
	int *h_finalAnswerFromCPU;
	int *h_finalAnswerFromGPU;
	int *h_array_data_input_CPU;
	int *h_array_data_forPrinting;
	int *d_array_data_input;
	int *d_array_data_output;
	//int totalAllowedElementsInABlock = (blocksize * 2);	//the total elements in a block
	int booleanValue = debugOn();
	getSizeOfN(sizeOfN);

	float temp_time = 0;
	printf("ElementSize: %d \nSize of Thread block: %d", sizeOfN, BLOCK_SIZE_);
	printf("\n\n");
	//set total size

	//The size of all elements
	size_t dsize = sizeOfN * sizeof(int);

	//Allocate memory for matrices on host
	h_array_data_input_CPU = (int*)malloc(dsize); //Run in CPU
	h_array_data_forPrinting = (int*)malloc(dsize); // For displaying the data
	h_finalAnswerFromCPU = (int*)malloc(dsize);
	h_finalAnswerFromGPU = (int*)malloc(dsize);

	memset(h_array_data_forPrinting, 0, dsize);
	memset(h_array_data_input_CPU, 0, dsize);
	memset(h_finalAnswerFromCPU, 0, dsize);
	memset(h_finalAnswerFromGPU, 0, dsize);
	h_finalAnswerFromGPU = (int*)malloc(dsize);

	if (h_array_data_input_CPU == NULL || h_array_data_forPrinting == NULL ||
		h_finalAnswerFromCPU == NULL || h_finalAnswerFromGPU == NULL) {
		printf("Failed to allocate host matrix C!\n");
	}

	setDataValues(h_array_data_input_CPU, sizeOfN);
	setDataValues(h_array_data_forPrinting, sizeOfN);
	//initF(h_array_data_input_CPU, sizeOfN);
	//initF(h_array_data_forPrinting, sizeOfN);
	//Copy and allocate memory

	cudaMalloc((void **)(&d_array_data_input), dsize);

	getAnyErrors("Allocating memory for d_array_data_input ");
	cudaMemcpy(d_array_data_input, h_array_data_input_CPU, dsize, cudaMemcpyHostToDevice);
	getAnyErrors("Copying array data from host to deviced ");
	cudaMalloc((void **)(&d_array_data_output), dsize);
	getAnyErrors("Allocating memory for d_array_data_output ");

	printf("CPU is do working....\n");
	printf("***********************************************************************\n");
	clock_t startTime, endTime;
	startTime = clock();
	PrefixSum(h_array_data_input_CPU, h_finalAnswerFromCPU, sizeOfN);
	endTime = clock();
	cpu_time = ((double)(endTime - startTime))* 1000.0 / CLOCKS_PER_SEC;
	printf("CPU End Value %d \n", h_finalAnswerFromCPU[sizeOfN - 1]);

	if (booleanValue) {
		printf("\nOriginal Array:\n");
		printf("-------------------------------------------------------------------\n");
		printf_matrix(h_array_data_forPrinting, sizeOfN);
		printf("\nAfter Prefix Sum Array:\n");
		printf("-------------------------------------------------------------------\n");
		printf_matrix(h_finalAnswerFromCPU, sizeOfN);
	}
	//-----------------------------------------------------------------------------------

	int myDynamicGrid = (sizeOfN - 1) / (BLOCK_SIZE_ * 2) + 1;
	size_t size_auxiliary = myDynamicGrid * sizeof(int);
	int newSize = (sizeOfN - 1) / (BLOCK_SIZE_ * 2) + 1;

	//Crucial kernel configurations
	dim3 dimBlock(blocksize);
	dim3 dimGrid(myDynamicGrid);
	printf("\nGPU is working \n");
	printf("***********************************************************************\n");
	printf("Kernel Reduction Prefix Sum\n");
	printf("-------------------------------------------------------------------\n");
	printf("Kernel Parameter\n");
	printf("dimBlock :%d\n", blocksize);
	printf("dimGrid :%d\n", myDynamicGrid);


	//dim3 GridAuxiliary(1);
	//dim3 BlockAuxiliary(myDynamicGrid);
	//printf("\nKernel Parameter\n");
	//printf("GridAuxiliary :%d\n", 1);
	//printf("BlockAuxiliary :%d\n", myDynamicGrid);
	////divide by extra factor of 2 because each subarray is BLOCK_SIZE*2 large.


	dim3 GridScanSecond(1);
	dim3 BlockScanSecond(myDynamicGrid);
	printf("\nKernel Parameter\n");
	printf("GridScanSecond :%d\n", 1);
	printf("BlockScanSecond :%d\n", myDynamicGrid);

	dim3 AddGrid(myDynamicGrid);
	dim3 AddBlock(BLOCK_SIZE_);
	//dim3 AddBlock(2 *BLOCK_SIZE_);

	printf("\nKernel Parameter\n");
	printf("AddGrid :%d\n", myDynamicGrid);
	printf("AddBlock :%d\n", BLOCK_SIZE_);

	int * auxiliaryAray;

	cudaMalloc((void **)(&auxiliaryAray), size_auxiliary);
	getAnyErrors("cudaMalloc((void **)(&auxiliaryAray), size_auxiliary);");

	cudaMemset(auxiliaryAray, 0, sizeof(int));
	getAnyErrors("cudaMemset(auxiliaryAray, 0, sizeof(int));");

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	work_efficient_scan_kernel << < dimGrid, dimBlock >> >(d_array_data_input, d_array_data_output, sizeOfN, auxiliaryAray);
	getAnyErrors("summing of total in kernel is running");
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&temp_time, start, stop);
	secTotal = secTotal + temp_time;
	cudaEventDestroy(start);
	cudaDeviceSynchronize();
	cudaEventDestroy(stop);
	cudaDeviceSynchronize();

	/*
	d_array_data_output holds all adjusted elements.
	I need the last value of each sub array.
	Couldn't figure out how to do this in the same kernel (work_efficient_scan_kernel)
	What I need is this.
	d_array_data_output = b0[2,4,6] b1[8,10,12] ....
	auxiliary_array [6,12]
	So I'm going to do it in a number kernel..
	*/

	//int * dd;
	//dd = (int*)malloc(size_auxiliary);
	//memset(dd, 0, size_auxiliary);

	//getAuxiliaryAray<< < GridAuxiliary, BlockAuxiliary >> >(d_array_data_output, auxiliaryAray, sizeOfN);
	//getAnyErrors("getAuxiliaryAray");
	//printf("s2\n");
	//cudaMemcpy(dd, auxiliaryAray, size_auxiliary, cudaMemcpyDeviceToHost);
	//printf_matrix(dd, newSize);

	int * d_finalAnswerFromGPU;
	cudaMalloc((void **)(&d_finalAnswerFromGPU), size_auxiliary);
	getAnyErrors("cudaMalloc((void **)(&d_finalAnswerFromGPU), size_auxiliary);");

	cudaMemset(d_finalAnswerFromGPU, 0, sizeof(int));
	getAnyErrors("cudaMemset(d_finalAnswerFromGPU, 0, sizeof(int))");
	/*
	int * gg;
	gg = (int*)malloc(size_auxiliary);
	memset(gg, 0, size_auxiliary);
*/
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//Re scan of auxiliaryArays
	work_efficient_scan_kernel << <1, BlockScanSecond >> >(auxiliaryAray, d_finalAnswerFromGPU, newSize, NULL);
	getAnyErrors("work_efficient_scan_kernel << <1, 512 >> >(auxiliaryAray, d_finalAnswerFromGPU, newSize, blocksize)");
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&temp_time, start, stop);
	secTotal = secTotal + temp_time;

	cudaEventDestroy(start);
	cudaDeviceSynchronize();
	cudaEventDestroy(stop);
	cudaDeviceSynchronize();
	//cudaMemcpy(gg, auxiliaryAray, size_auxiliary, cudaMemcpyDeviceToHost);
	//printf_matrix(gg, newSize);

	//int * h_finalAnswerFromGPUss;
	//h_finalAnswerFromGPUss = (int*)malloc(dsize);
	//memset(h_finalAnswerFromGPUss, 0, dsize);
	

	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//add scanned block sum i to all Values of scanned block i +1
	AddScannedBlocksSum << <AddGrid, AddBlock >> >(d_finalAnswerFromGPU, d_array_data_output, sizeOfN);
	getAnyErrors("AddScannedBlocksSum << <512, 2048 >> > (d_finalAnswerFromGPU, d_array_data_output, sizeOfN)");
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&temp_time, start, stop);
	secTotal = secTotal + temp_time;
	cudaEventDestroy(start);
	cudaDeviceSynchronize();
	cudaEventDestroy(stop);
	cudaDeviceSynchronize();

	/*cudaMemcpy(h_finalAnswerFromGPUss, d_array_data_output, dsize, cudaMemcpyDeviceToHost);
	printf_matrix(h_finalAnswerFromGPUss, sizeOfN);
*/
	cudaMemcpy(h_finalAnswerFromGPU, d_array_data_output, dsize, cudaMemcpyDeviceToHost);
	getAnyErrors("ddsfa\n");
	//printf_matrix(h_finalAnswerFromGPU, sizeOfN);
	cudaDeviceSynchronize();


	printf("\nGPU End Value %d \n", h_finalAnswerFromGPU[sizeOfN - 1]);
	if (booleanValue) {
		printf("\nOriginal Array:\n");
		printf("-------------------------------------------------------------------\n");
		printf_matrix(h_array_data_forPrinting, sizeOfN);
		printf("\nAfter Prefix Sum Array:\n");
		printf("-------------------------------------------------------------------\n");
		printf_matrix(h_finalAnswerFromGPU, sizeOfN);
	}

	printf("Verifying\n");
	verify(h_finalAnswerFromCPU, h_finalAnswerFromGPU, sizeOfN);


	cudaFree(auxiliaryAray);
	cudaFree(d_finalAnswerFromGPU);
	cudaFree(d_array_data_input);
	cudaFree(d_array_data_output);
	//Clean up memory
	free(h_finalAnswerFromGPU);
	free(h_finalAnswerFromCPU);
	free(h_array_data_input_CPU);
	free(h_array_data_forPrinting);


	printf("Execution Time for GPU: %.5f ms\n", secTotal);
	printf("Execution Time for CPU: %.5f ms\n", cpu_time);
	printf("Speedup : %.5f ms\n", cpu_time / secTotal);

}

