
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <vector>
#include <iostream>

using namespace std;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
void Gaussian(vector<vector<double>> &data, vector<double> &vector);
void ForwardElim(vector<vector<double>> &data, vector<double> &vector);
void BackSub(vector<vector<double>> &data, std::vector<double> &vector);

void GPUGaussian(vector<vector<double>> &data, int size);
__global__ void KernelForwardElim(double* upper, double* lower, int* _size, double* multiplier, int* _upperRow);

//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}

void FillMatrix(int size, vector<vector<double>> &data, vector<double> &vector);

int main()
{
    // Add vectors in parallel.
	cudaError_t cudaStatus = cudaSuccess;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	int size = 3;
	vector<vector<double>> data = { { 8,5,7 },{ 4,6,3 },{ 3,1,9 } };
	vector<double> vector = { 2,5,3 };
	std::vector<std::vector<double>> data2;
	data2 = data;
	data2[0].push_back(2);
	data2[1].push_back(5);
	data2[2].push_back(3);
	//FillMatrix(3, data, vector);
	Gaussian(data, vector);
	GPUGaussian(data2, size);
	
	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }


	cin.get();
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

void Gaussian(vector<vector<double>> &data, std::vector<double> &vector)
{
	ForwardElim(data, vector);
	BackSub(data, vector);

	for (int i = 0; i < data.size(); ++i)
	{
		for (int j = 0; j < data[i].size(); ++j)
		{
			cout << data[i][j] << '\t';
		}
		cout << vector[i] << endl;
	}
}

void ForwardElim(vector<vector<double>> &data, std::vector<double> &vector)
{
	for (unsigned int i = 0; i < (data.size() - 1); ++i)
	{
		double upper = data[i][i];
		for (unsigned int j = i; j < (data.size() - 1); ++j)
		{
			double lower = data[j + 1][i];
			double multiplier = upper / lower;
			vector[j + 1] *= multiplier;
			vector[j + 1] -= vector[i];
			for (unsigned int k = 0; k < data.size(); ++k)
			{
				data[j + 1][k] *= multiplier;
				data[j + 1][k] -= data[i][k];
			}
		}
	}
}

void BackSub(vector<vector<double>> &data, std::vector<double> &vector)
{
	for (int i = data.size() - 1; i >= 0; --i)
	{
		//data[i][data.size() - 1] /= data[i][i];
		vector[i] /= data[i][i];
		data[i][i] = 1;
		for (int j = i - 1; j >= 0; --j) //sätter j = 2 först och sen så länge 2 > 3 what
		{
			double subtrahend = data[j][i] * vector[i];
			vector[j] -= subtrahend;
			data[j][i] = 0;
		}
	}
}

void GPUGaussian(vector<vector<double>>& data, int size)
{
	double* devUpperRow		= 0;
	double* devLowerRow		= 0;
	int* devSize			= 0;
	double* devMultiplier	= 0;
	int* devUpperRowIdx		= 0;

	//malloc rows
	cudaError_t cudaStatus = cudaMalloc((void**)&devUpperRow, (size + 1) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for upperRow\n");
		return;
	}
	cudaStatus = cudaMalloc((void**)&devLowerRow, (size + 1) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for lowerRow\n");
		return;
	}
	//malloc rest
	cudaStatus = cudaMalloc((void**)&devSize, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for size\n");
		return;
	}
	cudaStatus = cudaMalloc((void**)&devMultiplier, sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for multiplier\n");
		return;
	}
	cudaStatus = cudaMalloc((void**)&devUpperRowIdx, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for upperRowIdx\n");
		return;
	}

	//memcpy size
	cudaStatus = cudaMemcpy(devSize, &size, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for size\n");
		return;
	}

	for (int i = 0; i < (size - 1); ++i)
	{
		double* tempData = data[i].data();
		cudaStatus = cudaMemcpy((void*)devUpperRow, (void*)tempData, (size + 1) * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for upperRow\n");
			return;
		}
		cudaStatus = cudaMemcpy((void*)devUpperRowIdx, (void*)&i, sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for multiplier\n");
			return;
		}

		//for a given pivot element, reduce all items below to zero
		for (int j = i + 1; j < size; ++j)
		{
			double multiplier = data[i][i] / data[j][i];
			tempData = data[j].data();
			cudaStatus = cudaMemcpy((void*)devLowerRow, (void*)tempData, (size + 1) * sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed for lowerRow HtD\n");
				return;
			}
			cudaStatus = cudaMemcpy((void*)devMultiplier, (void*)&multiplier, sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed for multiplier\n");
				return;
			}

			KernelForwardElim<<<1, 16>>>(devUpperRow, devLowerRow, devSize, devMultiplier, devUpperRowIdx);
			cudaThreadSynchronize();

			double* lowerRow = (double*)malloc((size + 1) * sizeof(double));

			cudaStatus = cudaMemcpy((void*)lowerRow, (void*)devLowerRow, (size + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed for lowerRow DtH\n");
				return;
			}

			for (int k = 0; k < size + 1; ++k)
			{
				data[j][k] = lowerRow[k];
			}
		}
	}


}
//											data[i],	data[j],	data.size(), upperMainDiag/theOneUnderIt 
__global__ void KernelForwardElim(double* upperRow, double* lowerRow, int* _size, double* multiplier, int* _upperRowIdx)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int upperRowIdx = *_upperRowIdx;
	if (col >= upperRowIdx && col <= *_size)
	{
		lowerRow[col] *= *multiplier;
		lowerRow[col] -= upperRow[col];
	}
}

void FillMatrix(int size, vector<vector<double>> &data, vector<double> &vector)
{
	data.clear();
	vector.clear();

	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size; ++j)
		{
			vector.push_back(rand() % 10 + 1); //values between 1 and 10
			data.push_back(vector);
			vector.clear();
			//pushback random values to data[i][j]
		}
	}
	for (int i = 0; i < size; ++i)
	{
		
	}
}