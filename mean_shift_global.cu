#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>

#ifdef A
    #define DATASET 1
    #define SIZE 600
    #define DIMENSIONS 2
#else
    #define DATASET 2
    #define SIZE 210
    #define DIMENSIONS 7
#endif

__global__ void mean_shift(double *y, double *x, double *h, double *kernel, double *i_vectors, double *norm)
{
  //Thread scoped variables.
  int basic_index = blockIdx.x * (DIMENSIONS);
  int index = threadIdx.x * (DIMENSIONS);
  double e = 0.0001;
  double factor;

  //Initializing the norm.
  norm[blockIdx.x] = 1.0;

  //Main loop.
  while (norm[blockIdx.x] > e) {

    factor = 0.0;
    //Calculating factor.
    for (int i = 0; i < (DIMENSIONS); i++)
    {
        factor += (y[basic_index + i] - x[index + i])*(y[basic_index + i] - x[index + i]);
    }

    if (sqrt(factor) > *h*(*h))
    {
      kernel[blockIdx.x * SIZE + threadIdx.x] = 0.0;
      for (int i = 0; i < (DIMENSIONS); i++)
      {
        i_vectors[blockIdx.x * SIZE * DIMENSIONS + index+i] = 0.0;
      }
    }
    else
    {
      factor = exp(-(factor)/(2* (*h) * (*h)));

      kernel[blockIdx.x * SIZE + threadIdx.x] = factor;

      //Storing the 'y' vectors before normalization.
      for (int i = 0; i < (DIMENSIONS); i++)
      {
        i_vectors[blockIdx.x * SIZE * DIMENSIONS + index+i] = factor * x[index + i];
      }
    }

    __syncthreads();

    //Only the first thread of each block will calculate
    //the final vectors.
    if (threadIdx.x == 0)
    {
      double y_new[DIMENSIONS];
      for (int i = 0; i < (DIMENSIONS); i++)
      {
	       y_new[i] = 0.0;
      }

      double sum = 0.0;

      for (int i = 0; i < (SIZE) * (DIMENSIONS); i += (DIMENSIONS))
      {
      	for (int j = 0; j < (DIMENSIONS); j++)
      	{
      	  y_new[j] += i_vectors[blockIdx.x * SIZE * DIMENSIONS + i + j];
      	}
      }

      //Calculating the sum of each row of the kernel_matrix.
      for (int i = 0; i < (SIZE); i++)
      {
        sum += kernel[blockIdx.x * SIZE + i];
      }

      //Normalizing and updating the 'y' vector.
      for (int i = 0; i < (DIMENSIONS); i++)
      {
        y_new[i] = y_new[i] / sum;
      }


      //Re-initializing norm.
      norm[blockIdx.x] = 0.0;

      //Calculating the norm of the vector
      //and updating the 'y' vector
      for (int i = 0; i < (DIMENSIONS); i++)
      {

        norm[blockIdx.x] += (y_new[i]-y[basic_index + i])*(y_new[i]-y[basic_index + i]);
        y[basic_index + i] = y_new[i];

      }

      norm[blockIdx.x] = sqrt(norm[blockIdx.x]);

    }

    __syncthreads();

  }

}

int main(int argc, char **argv)
{
    FILE *file;
    char data_file_name[64], results_file_name[64];
    double *y, *x, *z, h, *kernel, *i_vectors, *norm;
    double *d_y, *d_x, *d_h, *d_kernel, *d_i_vectors, *d_norm;
    int size_double = SIZE * DIMENSIONS * sizeof(double);

	if (argc == 2)
	{
		h = atof(argv[1]);
	}
	else
	{
		if (DATASET == 1)
		{
			h = 1.0;
		}
		else
		{
			h = 1.4565;
		}
	}

    if (DATASET == 1)
    {
        sprintf(data_file_name, "data_cuda.bin");
        sprintf(results_file_name, "results_global.bin");
    }
    else
    {
        sprintf(data_file_name, "data_cuda_seeds.bin");
        sprintf(results_file_name, "results_global_seeds.bin");
    }

    y = (double *)malloc(size_double);
    x = (double *)malloc(size_double);
    z = (double *)malloc(size_double);

    kernel = (double *)malloc(SIZE * SIZE * sizeof(double));
    i_vectors = (double *)malloc(SIZE * DIMENSIONS * SIZE * sizeof(double));
    norm = (double *)malloc(SIZE * sizeof(double));

    file = fopen(data_file_name, "rb");
    if (file == NULL)
    {
        printf("Could not open file\n");
        exit(1);
    }

    if(fread(y, sizeof(double), SIZE * DIMENSIONS, file) != SIZE*DIMENSIONS){
        printf("Error at reading file\n");
        exit(1);
    }

    fclose(file);

    for(int i = 0; i < SIZE * DIMENSIONS; i++)
    {
        x[i] = y[i];
    }

    cudaMalloc((void **)&d_y, size_double);
    cudaMalloc((void **)&d_x, size_double);
    cudaMalloc((void **)&d_h, sizeof(double));
    cudaMalloc((void **)&d_kernel, SIZE * SIZE * sizeof(double));
    cudaMalloc((void **)&d_i_vectors, SIZE * DIMENSIONS * SIZE * sizeof(double));
    cudaMalloc((void **)&d_norm, SIZE * sizeof(double));

    cudaMemcpy(d_y, y, size_double, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, size_double, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, &h, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, SIZE * SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_i_vectors, i_vectors, SIZE * DIMENSIONS * SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_norm, norm, SIZE * sizeof(double), cudaMemcpyHostToDevice);

    //Time variables.
    struct timeval startwtime, endwtime;

    gettimeofday (&startwtime, NULL);

    mean_shift<<<SIZE,SIZE>>>(d_y, d_x, d_h, d_kernel, d_i_vectors, d_norm);

    cudaDeviceSynchronize();

    gettimeofday (&endwtime, NULL);

    cudaMemcpy(z, d_y, size_double, cudaMemcpyDeviceToHost);

    double exec_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
    + endwtime.tv_sec - startwtime.tv_sec);
    printf("Runtime: %.14f \n", exec_time);

    file = fopen(results_file_name, "wb");
    if (file == NULL)
    {
        printf("Could not open file\n");
        exit(1);
    }

    fwrite(z, sizeof(double), SIZE * DIMENSIONS, file);

    fclose(file);

    cudaFree(d_y);
    cudaFree(d_x);
    cudaFree(d_h);
    cudaFree(d_kernel);
    cudaFree(d_i_vectors);
    cudaFree(d_norm);

    free(y);
    free(x);
    free(z);
    free(kernel);
    free(i_vectors);
    free(norm);

    return 0;
}
