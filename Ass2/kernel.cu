#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <iostream>
#define epsilon 0.000001

using namespace std;

void Gaussian(double* data, int size);
void ForwardElim(double* data, int size);
void BackSub(double* data, int size);
void SwapRows(double* data, int size, int upperRow, int lowerRow);
bool CompareResults(double* data, double* data2, int size);
void GPUGaussian(double* &data, int size, int blocks, int rowsPerThread);
void FillMatrix(double* data, double* data2, double* backup, int size);
void CopyMatrix(double* src, double* dest, int size);
__global__ void KernelForwardElim(double* _matrix, int* _size, int* _upperRowIdx, int* _rowsPerThread);

int main()
{
	srand(time(NULL));
    // Add vectors in parallel.
	cudaError_t cudaStatus = cudaSuccess;

	//int size = 256;
	//int colsPerThread = 1;
	//vector<vector<double>> data, data2, backup;
	double* data, *data2, *backup;

	for (int size = 128; size < 1025; size *= 2)
	{
		cout << "Working on size: " << size << endl;
		data = (double*)malloc((size + 1) * size * sizeof(double));
		backup = (double*)malloc((size + 1) * size * sizeof(double));
		data2 = (double*)malloc((size + 1) * size * sizeof(double));
		FillMatrix(data, data2, backup, size);
		Gaussian(data, size);
		for (int rowsPerThread = 1; rowsPerThread < 9; rowsPerThread *= 2)
		{
			cout << "Working on rowsPerThread: " << rowsPerThread << endl;
			CopyMatrix(backup, data2, size);
			int threads = (size + 1) / rowsPerThread;
			int blocks = (threads - 1) / 1024 + 1; /*1024 max for current graphics card used*/
			GPUGaussian(data2, size, blocks, rowsPerThread);
			if (!CompareResults(data, data2, size))
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

void Gaussian(double* data, int size)
{
	clock_t t;
	t = clock();
	ForwardElim(data, size);
	t = clock() - t;

	std::cout << "CPU Forward Substituion took: " << t << "clicks ("<< ((float)t)/CLOCKS_PER_SEC << " seconds.)" << endl;

	BackSub(data, size);
}

void ForwardElim(double* data, int size)
{
	for (unsigned int i = 0; i < size - 1; ++i)
	{
		/*if (abs(data[i * (size + 1) + i]) < epsilon)
		{
			int j = NULL;
			for (j = i + 1; j < size; ++j)
			{
				if (abs(data[j * (size + 1) + i]) > epsilon)
				{
					SwapRows(data, size, i, j);
					break;
				}
			}

			if (j == size - 1)
				data[i * (size + 1) + i] = 1;
		}*/
		double upper = data[i * (size + 1) + i];
		for (unsigned int j = i + 1; j < size; ++j)
		{
			/*bool breaK = false;
			while (j < size && abs(data[j * (size + 1) + i]) < epsilon)
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
			}*/
			double lower = data[j * (size + 1) + i];
			double multiplier = upper / lower;
			for (unsigned int k = i + 1; k < size + 1; ++k)
			{
				data[j * (size + 1) + k] *= multiplier;
				data[j * (size + 1) + k] -= data[i * (size + 1) + k];
			}
		}
	}
}

void BackSub(double* data, int size)
{
	for (int i = size - 1; i >= 0; --i)
	{
		data[i * (size + 1) + size] /= data[i * (size + 1) + i];
		//data[i * (size + 1) + i] = 1;
		for (int j = i - 1; j >= 0; --j)
		{
			double subtrahend = data[j * (size + 1) + i] * data[i * (size + 1) + size];
			data[j * (size + 1) + size] -= subtrahend;
			//data[j * (size + 1) * i] = 0;
		}
	}
}

void GPUGaussian(double* &data, int size, int blocks, int rowsPerThread)
{
	double* devMatrix		= 0;
	int* devSize			= 0;
	int* devUpperRowIdx		= 0;
	int* devRowsPerThread	= 0;

	//malloc matrix
	cudaError_t cudaStatus = cudaMalloc((void**)&devMatrix, (size + 1) * size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for matrix\n");
		return;
	}
	/*cudaError_t cudaStatus = cudaMalloc((void**)&devUpperRow, (size + 1) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for upperRow\n");
		return;
	}
	cudaStatus = cudaMalloc((void**)&devLowerRow, (size + 1) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for lowerRow\n");
		return;
	}*/
	//malloc rest
	cudaStatus = cudaMalloc((void**)&devSize, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for size\n");
		return;
	}
	cudaStatus = cudaMalloc((void**)&devUpperRowIdx, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for upperRowIdx\n");
		return;
	}
	cudaStatus = cudaMalloc((void**)&devRowsPerThread, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for colsPerThread\n");
		return;
	}

	//memcpy matrix
	cudaStatus = cudaMemcpy(devMatrix, data, (size + 1) * size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for multiplier\n");
		return;
	}
	//memcpy size
	cudaStatus = cudaMemcpy(devSize, &size, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for size\n");
		return;
	}
	//memcpy rowsPerThread
	cudaStatus = cudaMemcpy(devRowsPerThread, &rowsPerThread, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for colsPerThread\n");
		return;
	}

	clock_t t;
	t = clock();
	for (int i = 0; i < (size - 1); ++i)
	{
		/*if (abs((*data)[i][i]) < epsilon)
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
		}*/

		/*double* tempData = (double*)malloc((size + 1) * sizeof(double));
		tempData = (*data)[i];*/

		/*cudaStatus = cudaMemcpy((void*)devUpperRow, (void*)tempData, (size + 1) * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for upperRow\n");
			return;
		}*/
		cudaStatus = cudaMemcpy((void*)devUpperRowIdx, (void*)&i, sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for multiplier\n");
			return;
		}

		//for a given pivot element, reduce all items below to zero
		/*for (int j = i + 1; j < size; ++j)
		{*/
			/*bool breaK = false;
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
			}*/
			/*double multiplier = (*data)[i][i] / (*data)[j][i];
			tempData = (*data)[j];*/
			/*cudaStatus = cudaMemcpy((void*)devLowerRow, (void*)tempData, (size + 1) * sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed for lowerRow HtD\n");
				return;
			}*/
			/*cudaStatus = cudaMemcpy((void*)devMultiplier, (void*)&multiplier, sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed for multiplier\n");
				return;
			}*/

			KernelForwardElim<<<blocks, 1024>>>(devMatrix, devSize, devUpperRowIdx, devRowsPerThread);
			

			double* tempMatrix = (double*)malloc((size + 1) * size * sizeof(double));

			cudaStatus = cudaMemcpy((void*)tempMatrix, (void*)devMatrix, (size + 1) * size * sizeof(double), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed for lowerRow DtH\n");
				return;
			}
			
			data = tempMatrix;

		//}	
	}
	t = clock() - t;
	std::cout << "GPU Forward Substituion took: " << t << "clicks (" << ((float)t) / CLOCKS_PER_SEC << " seconds.)" << endl;
	BackSub(data, size);

	cudaFree(devMatrix);
	cudaFree(devSize);
	cudaFree(devUpperRowIdx);
	cudaFree(devRowsPerThread);
}

void SwapRows(double* data, int size, int upperRow, int lowerRow)
{
	for (int i = 0; i < size + 1; ++i)
	{
		double temp = data[upperRow * (size + 1) + i];
		data[upperRow * (size + 1) + i] = data[lowerRow * (size + 1) * i];
		data[lowerRow * (size + 1) + i] = temp;
	}
}

bool CompareResults(double* data, double* data2, int size)
{
	bool test = true;
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size + 1; ++j)
		{
			if (abs(data[i * (size + 1) + j] - data2[i * (size + 1) + j]) > epsilon && abs(data[i * (size + 1) + j]) > epsilon && abs(data2[i * (size + 1) + j]) > epsilon)
			{
				cout << "Something went wrong" << endl;
				cout << "CPU: " << data[i * (size + 1) + j] << "|\tGPU:" << data2[i * (size + 1) + j] << endl;
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
__global__ void KernelForwardElim(double* _matrix, int* _size, int* _upperRowIdx, int* _rowsPerThread)
{
	int rowsPerThread = *_rowsPerThread;
	int upperRowIdx = *_upperRowIdx;
	int startRow = (threadIdx.x + blockIdx.x * blockDim.x) * rowsPerThread + 1 + upperRowIdx;
	int size = *_size;
	if (startRow > upperRowIdx)
	{
		for (int row = startRow; row < rowsPerThread + startRow; ++row)
		{
			if (row < size)
			{
				double multiplier = _matrix[upperRowIdx * (size + 1) + upperRowIdx] / _matrix[row * (size + 1) + upperRowIdx];
				double test = _matrix[upperRowIdx * (size + 1) + upperRowIdx];
				double test2 = _matrix[row * (size + 1) + upperRowIdx];
				for (int i = upperRowIdx + 1; i < size + 1; ++i)
				{
					_matrix[row * (size + 1) + i] *= multiplier;
					_matrix[row * (size + 1) + i] -= _matrix[upperRowIdx * (size + 1) + i];
					double test3 = _matrix[row * (size + 1) + i];
				}
			}
		}
	}
}

void FillMatrix(double* data, double* data2, double* backup, int size)
{
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size + 1; ++j)
		{
			data[i * (size + 1)  + j] = data2[i * (size + 1) + j] = backup[i * (size + 1) + j] = rand() % 10 + 1;
		}
	}
}

void CopyMatrix(double* src, double* dest, int size)
{
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size + 1; ++j)
		{
			dest[i * (size + 1) + j] = src[i * (size + 1) + j];
		}
	}
}