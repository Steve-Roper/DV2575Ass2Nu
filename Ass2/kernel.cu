#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <iostream>
#define epsilon 0.000001

using namespace std;

void Gaussian(float* data, int size, FILE* file);
void ForwardElim(float* data, int size);
void BackSub(float* data, int size);
void SwapRows(float* data, int size, int upperRow, int lowerRow);
bool CompareResults(float* data, float* data2, int size);
void GPUGaussian(float* &data, int size, int blocks, int rowsPerThread, FILE* file);
void FillMatrix(float* data, float* data2, float* backup, int size);
void CopyMatrix(float* src, float* dest, int size);
__global__ void KernelForwardElim(float* _matrix, int _size, int _upperRowIdx, int _rowsPerThread);

int main()
{
	srand(time(NULL));
    // Add vectors in parallel.
	cudaError_t cudaStatus = cudaSuccess;

	//int size = 256;
	//int colsPerThread = 1;
	//vector<vector<float>> data, data2, backup;
	float* data, *data2, *backup;
	FILE* file = fopen("data.csv", "w+");
	for (int size = 128; size < 2049; size *= 2)
	{
		std::cout << "---------------------------------------------------------------------" << endl;
		std::cout << "Working on size: " << size << endl;
		data = (float*)malloc((size + 1) * size * sizeof(float));
		backup = (float*)malloc((size + 1) * size * sizeof(float));
		data2 = (float*)malloc((size + 1) * size * sizeof(float));
		FillMatrix(data, data2, backup, size);
		Gaussian(data, size, file);
		for (int rowsPerThread = 1; rowsPerThread < 9; rowsPerThread *= 2)
		{
			std::cout << "Working on rowsPerThread: " << rowsPerThread << endl;
			CopyMatrix(backup, data2, size);
			int threads = (size + 1) / rowsPerThread;
			int blocks = (threads - 1) / 1024 + 1; /*1024 max for current graphics card used*/
			GPUGaussian(data2, size, blocks, rowsPerThread, file);
			if (!CompareResults(data, data2, size))
			{
				break;
			}
		}
		fprintf(file, "\n");
		free(data);
		free(data2);
		free(backup);
	}
	fclose(file);
	std::cout << "---------------------------------------------------------------------" << endl;
	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	std::cout << "Press any key to continue . . .";
	std::cin.get();
    return 0;
}

void Gaussian(float* data, int size, FILE* file)
{
	clock_t t;
	t = clock();
	ForwardElim(data, size);
	t = clock() - t;

	std::cout << "CPU Forward Substituion took: " << t << "clicks ("<< ((float)t)/CLOCKS_PER_SEC << " seconds.)" << endl;
	fprintf(file, "%f.2,", ((float)t) / CLOCKS_PER_SEC);

	BackSub(data, size);
}

void ForwardElim(float* data, int size)
{
	for (unsigned int i = 0; i < size - 1; ++i)
	{
		float upper = data[i * (size + 1) + i];
		for (unsigned int j = i + 1; j < size; ++j)
		{
			float lower = data[j * (size + 1) + i];
			float multiplier = upper / lower;
			for (unsigned int k = i + 1; k < size + 1; ++k)
			{
				data[j * (size + 1) + k] *= multiplier;
				data[j * (size + 1) + k] -= data[i * (size + 1) + k];
			}
		}
	}
}

void BackSub(float* data, int size)
{
	for (int i = size - 1; i >= 0; --i)
	{
		data[i * (size + 1) + size] /= data[i * (size + 1) + i];
		for (int j = i - 1; j >= 0; --j)
		{
			float subtrahend = data[j * (size + 1) + i] * data[i * (size + 1) + size];
			data[j * (size + 1) + size] -= subtrahend;
		}
	}
}

void GPUGaussian(float* &data, int size, int blocks, int rowsPerThread, FILE* file)
{
	float* devMatrix		= 0;

	clock_t t;
	t = clock();

	cudaError_t cudaStatus = cudaMalloc((void**)&devMatrix, (size + 1) * size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for matrix\n");
		return;
	}
	cudaStatus = cudaMemcpy(devMatrix, data, (size + 1) * size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for multiplier\n");
		return;
	}

	float* tempMatrix = (float*)malloc((size + 1) * size * sizeof(float));
	for (int i = 0; i < (size - 1); ++i)
	{

			KernelForwardElim<<<blocks, 1024>>>(devMatrix, size, i, rowsPerThread);
	}
	cudaStatus = cudaMemcpy((void*)tempMatrix, (void*)devMatrix, (size + 1) * size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for lowerRow DtH\n");
		return;
	}
	data = tempMatrix;
	t = clock() - t;
	std::cout << "GPU Forward Substituion took: " << t << "clicks (" << ((float)t) / CLOCKS_PER_SEC << " seconds.)" << endl;
	fprintf(file, "%f.2,", ((float)t) / CLOCKS_PER_SEC);
	BackSub(data, size);

	cudaFree(devMatrix);
	cudaFree(tempMatrix);
}

void SwapRows(float* data, int size, int upperRow, int lowerRow)
{
	for (int i = 0; i < size + 1; ++i)
	{
		float temp = data[upperRow * (size + 1) + i];
		data[upperRow * (size + 1) + i] = data[lowerRow * (size + 1) * i];
		data[lowerRow * (size + 1) + i] = temp;
	}
}

bool CompareResults(float* data, float* data2, int size)
{
	bool test = true;
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size + 1; ++j)
		{
			if (abs(data[i * (size + 1) + j] - data2[i * (size + 1) + j]) > epsilon && abs(data[i * (size + 1) + j]) > epsilon && abs(data2[i * (size + 1) + j]) > epsilon)
			{
				std::cout << "Something went wrong" << endl;
				std::cout << "CPU: " << data[i * (size + 1) + j] << "|\tGPU:" << data2[i * (size + 1) + j] << endl;
				test = false;
			}
		}
	}
	if (test)
	{
		std::cout << "CPU and GPU results match!" << endl;
	}
	return test;
}

__global__ void KernelForwardElim(float* _matrix, int _size, int _upperRowIdx, int _rowsPerThread)
{
	int startRow = (threadIdx.x + blockIdx.x * blockDim.x) * _rowsPerThread + 1 + _upperRowIdx;
	if (startRow > _upperRowIdx)
	{
		for (int row = startRow; row < _rowsPerThread + startRow; ++row)
		{
			if (row < _size)
			{
				float multiplier = _matrix[_upperRowIdx * (_size + 1) + _upperRowIdx] / _matrix[row * (_size + 1) + _upperRowIdx];
				for (int i = _upperRowIdx + 1; i < _size + 1; ++i)
				{
					_matrix[row * (_size + 1) + i] *= multiplier;
					_matrix[row * (_size + 1) + i] -= _matrix[_upperRowIdx * (_size + 1) + i];
				}
			}
		}
	}
}

void FillMatrix(float* data, float* data2, float* backup, int size)
{
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size + 1; ++j)
		{
			data[i * (size + 1)  + j] = data2[i * (size + 1) + j] = backup[i * (size + 1) + j] = rand() % 10 + 1;
		}
	}
}

void CopyMatrix(float* src, float* dest, int size)
{
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size + 1; ++j)
		{
			dest[i * (size + 1) + j] = src[i * (size + 1) + j];
		}
	}
}