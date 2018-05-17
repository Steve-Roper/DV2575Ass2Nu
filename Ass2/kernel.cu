#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <iostream>
#define epsilon 0.000001

using namespace std;

void Gaussian(double*** data, int size);
void ForwardElim(double*** data, int size);
void BackSub(double*** data, int size);
void SwapRows(double*** data, int size, int upperRow, int lowerRow);
bool CompareResults(double*** data, double*** data2, int size);
void GPUGaussian(double*** data, int size, int blocks, int colsPerThread);
void FillMatrix(double*** data, double*** data2, double*** backup, int size);
void CopyMatrix(double*** src, double*** dest, int size);
__global__ void KernelForwardElim(double* upper, double* lower, int* _size, double* multiplier, int* _upperRow, int* colsPerThread);

int main()
{
	srand(time(NULL));
    // Add vectors in parallel.
	cudaError_t cudaStatus = cudaSuccess;

	//int size = 256;
	//int colsPerThread = 1;
	//vector<vector<double>> data, data2, backup;
	double** data, **data2, **backup;

	for (int size = 128; size < 1025; size *= 2)
	{
		data = (double**)malloc((size + 1) * size * sizeof(double));
		backup = (double**)malloc((size + 1) * size * sizeof(double));
		data2 = (double**)malloc((size + 1) * size * sizeof(double));
		FillMatrix(&data, &data2, &backup, size);
		Gaussian(&data, size);
		for (int colsPerThread = 1; colsPerThread < 9; colsPerThread *= 2)
		{
			CopyMatrix(&backup, &data2, size);
			int threads = (size + 1) / colsPerThread;
			int blocks = (threads - 1) / 1024 + 1; /*1024 max for current graphics card used*/
			GPUGaussian(&data2, size, blocks, colsPerThread);
			if (!CompareResults(&data, &data2, size))
			{
				break;
			}
		}
		free(data);
		free(data2);
		free(backup);
	}
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

void Gaussian(double*** data, int size)
{
	clock_t t;
	t = clock();
	ForwardElim(data, size);
	t = clock() - t;

	std::cout << "CPU Forward Substituion took: " << t << "clicks ("<< ((float)t)/CLOCKS_PER_SEC << " seconds.)" << endl;

	BackSub(data, size);
}

void ForwardElim(double*** data, int size)
{
	for (unsigned int i = 0; i < size - 1; ++i)
	{
		if (abs((*data)[i][i]) < epsilon)
		{
			int j = NULL;
			for (j = i + 1; j < size; ++j)
			{
				if (abs((*data)[j][i]) > epsilon)
				{
					SwapRows(data, size, i, j);
					break;
				}
			}

			if (j == size - 1)
				(*data)[i][i] = 1;
		}
		double upper = (*data)[i][i];
		for (unsigned int j = i + 1; j < size; ++j)
		{
			bool breaK = false;
			while (j < size && abs((*data)[j][i]) < epsilon)
			{
				++j;
				if (j == size)
				{
					breaK = true;
					break;
				}
			}
			if (breaK)
			{
				break;
			}
			double lower = (*data)[j][i];
			double multiplier = upper / lower;
			for (unsigned int k = i + 1; k < size + 1; ++k)
			{
				(*data)[j][k] *= multiplier;
				(*data)[j][k] -= (*data)[i][k];
			}
		}
	}
}

void BackSub(double*** data, int size)
{
	for (int i = size - 1; i >= 0; --i)
	{
		(*data)[i][size] /= (*data)[i][i];
		(*data)[i][i] = 1;
		for (int j = i - 1; j >= 0; --j) //sätter j = 2 först och sen så länge 2 > 3 what
		{
			double subtrahend = (*data)[j][i] * (*data)[i][size];
			(*data)[j][size] -= subtrahend;
			(*data)[j][i] = 0;
		}
	}
}

void GPUGaussian(double*** data, int size, int blocks, int colsPerThread)
{
	double* devUpperRow		= 0;
	double* devLowerRow		= 0;
	int* devSize			= 0;
	double* devMultiplier	= 0;
	int* devUpperRowIdx		= 0;
	int* devColsPerThread	= 0;

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
	cudaStatus = cudaMalloc((void**)&devColsPerThread, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for colsPerThread\n");
		return;
	}

	//memcpy size
	cudaStatus = cudaMemcpy(devSize, &size, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for size\n");
		return;
	}
	cudaStatus = cudaMemcpy(devColsPerThread, &colsPerThread, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for colsPerThread\n");
		return;
	}

	clock_t t;
	t = clock();
	for (int i = 0; i < (size - 1); ++i)
	{
		if (abs((*data)[i][i]) < epsilon)
		{
			int j = NULL;
			for (j = i + 1; j < size; ++j)
			{
				if (abs((*data)[j][i]) > epsilon)
				{
					SwapRows(data, size, i, j);
					break;
				}
			}

			if (j == size - 1)
				(*data)[i][i] = 1;
		}

		double* tempData = (double*)malloc((size + 1) * sizeof(double));
		tempData = (*data)[i];

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
			bool breaK = false;
			while (abs((*data)[j][i]) < epsilon)
			{
				++j;
				if (j == size)
				{
					breaK = true;
					break;
				}
			}
			if (breaK)
			{
				break;
			}
			double multiplier = (*data)[i][i] / (*data)[j][i];
			tempData = (*data)[j];
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

			KernelForwardElim<<<blocks, 1024>>>(devUpperRow, devLowerRow, devSize, devMultiplier, devUpperRowIdx, devColsPerThread);
			

			double* lowerRow = (double*)malloc((size + 1) * sizeof(double));

			cudaStatus = cudaMemcpy((void*)lowerRow, (void*)devLowerRow, (size + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed for lowerRow DtH\n");
				return;
			}

			(*data)[j] = lowerRow;
		}	
	}
	t = clock() - t;
	std::cout << "GPU Forward Substituion took: " << t << "clicks (" << ((float)t) / CLOCKS_PER_SEC << " seconds.)" << endl;
	BackSub(data, size);

	cudaFree(devUpperRow);
	cudaFree(devLowerRow);
	cudaFree(devSize);
	cudaFree(devMultiplier);
	cudaFree(devUpperRowIdx);
	cudaFree(devColsPerThread);
}

void SwapRows(double*** data, int size, int upperRow, int lowerRow)
{
	double* temp = (double*)malloc((size + 1) * sizeof(double));
	temp = (*data)[upperRow];
	(*data)[upperRow] = (*data)[lowerRow];
	(*data)[lowerRow] = temp;
}

bool CompareResults(double*** data, double*** data2, int size)
{
	bool test = true;
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size + 1; ++j)
		{
			if (abs((*data)[i][j] - (*data2)[i][j]) > epsilon && abs((*data)[i][j]) > epsilon && abs((*data2)[i][j]) > epsilon)
			{
				cout << "Something went wrong" << endl;
				cout << "CPU: " << (*data)[i][j] << "|\tGPU:" << (*data2)[i][j] << endl;
				test = false;
			}
		}
	}
	if (test)
	{
		cout << "CPU and GPU results match!" << endl;
	}
	return test;
}

//											data[i],	data[j],	data.size(), upperMainDiag/theOneUnderIt 
__global__ void KernelForwardElim(double* upperRow, double* lowerRow, int* _size, double* multiplier, int* _upperRowIdx, int* colsPerThread)
{
	int _colsPerThread = *colsPerThread;
	int startCol = (threadIdx.x + blockIdx.x * blockDim.x) * _colsPerThread;
	int upperRowIdx = *_upperRowIdx;
	for (int col = startCol; col < _colsPerThread + startCol; ++col)
	{
		if (col > upperRowIdx && col <= *_size)
		{
			lowerRow[col] *= *multiplier;
			lowerRow[col] -= upperRow[col];
		}
	}
}

void FillMatrix(double*** data, double*** data2, double*** backup, int size)
{
	for (int i = 0; i < size; ++i)
	{
		double* temp = (double*)malloc((size + 1) * sizeof(double));
		double* temp2 = (double*)malloc((size + 1) * sizeof(double));
		double* temp3 = (double*)malloc((size + 1) * sizeof(double));
		for (int j = 0; j < size + 1; ++j)
		{
			double val = rand() % 10 + 1;
			temp[j] = temp2[j] = temp3[j] = val;
		}
		(*data)[i] = temp;
		(*data2)[i] = temp2;
		(*backup)[i] = temp3;
	}
}

void CopyMatrix(double*** src, double*** dest, int size)
{
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size + 1; ++j)
		{
			(*dest)[i][j] = (*src)[i][j];
		}
	}
}