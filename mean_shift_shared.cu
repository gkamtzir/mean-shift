#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>

#define SIZE 600
#define DIMENSIONS 2

__global__ void mean_shift(double *y, double *x, double *h)
{
  //Shared variables.
  __shared__ double kernel[SIZE];
  __shared__ double i_vectors[SIZE*DIMENSIONS];
  __shared__ double norm;

  //Thread scoped variables.
  int basic_index = blockIdx.x * (DIMENSIONS);
  int index = threadIdx.x * (DIMENSIONS);
  double e = 0.0001;
  double factor;

  //Initializing the norm.
  norm = 1.0;

  //Main loop.
  while (norm > e) {

    factor = 0.0;
    //Calculating factor.
    for (int i = 0; i < (DIMENSIONS); i++)
    {
        factor += (y[basic_index + i] - x[index + i])*(y[basic_index + i] - x[index + i]);
    }

    if (sqrt(factor) > *h*(*h))
    {
      kernel[threadIdx.x] = 0.0;
      for (int i = 0; i < (DIMENSIONS); i++)
      {
        i_vectors[index+i] = 0.0;
      }
    }
    else
    {
      factor = exp(-(factor)/(2* (*h) * (*h)));

      kernel[threadIdx.x] = factor;

      //Storing the 'y' vectors before normalization.
      for (int i = 0; i < (DIMENSIONS); i++)
      {
        i_vectors[index+i] = factor * x[index + i];
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
      	  y_new[j] += i_vectors[i + j];
      	}
      }

      //Calculating the sum of each row of the kernel_matrix.
      for (int i = 0; i < (SIZE); i++)
      {
        sum += kernel[i];
      }

      //Normalizing and updating the 'y' vector.
      for (int i = 0; i < (DIMENSIONS); i++)
      {
        y_new[i] = y_new[i] / sum;
      }

      //Re-initializing norm.
      norm = 0.0;

      //Calculating the norm of the vector
      //and updating the 'y' vector
      for (int i = 0; i < (DIMENSIONS); i++)
      {

        norm += (y_new[i]-y[basic_index + i])*(y_new[i]-y[basic_index + i]);
        y[basic_index + i] = y_new[i];

      }

      norm = sqrt(norm);

    }

    __syncthreads();

  }

}

int main(int argc, char **argv)
{
    FILE *file;
    double *y, *x, *z, h;
    double *d_y, *d_x, *d_h;
    int size_double = SIZE * DIMENSIONS * sizeof(double);
	
	if (argc == 2)
	{
		h = atof(argv[1]);
	}
	else
	{
		 h = 1.0;
	}

    y = (double *)malloc(size_double);
    x = (double *)malloc(size_double);
    z = (double *)malloc(size_double);

    file = fopen("data_cuda.bin", "rb");
    if (file == NULL)
    {
        printf("Could not open file\n");
        exit(1);
    }

    if (fread(y, sizeof(double), SIZE * DIMENSIONS, file) != SIZE * DIMENSIONS)
    {
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

    cudaMemcpy(d_y, y, size_double, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, size_double, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, &h, sizeof(double), cudaMemcpyHostToDevice);

    //Time variables.
    struct timeval startwtime, endwtime;

    gettimeofday (&startwtime, NULL);

    mean_shift<<<SIZE,SIZE>>>(d_y, d_x, d_h);

    cudaDeviceSynchronize();

    gettimeofday (&endwtime, NULL);

    cudaMemcpy(z, d_y, size_double, cudaMemcpyDeviceToHost);
	
    double exec_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
    + endwtime.tv_sec - startwtime.tv_sec);
    printf("Runtime: %.14f \n", exec_time);

    file = fopen("results_shared.bin", "wb");
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

    free(y);
    free(x);
    free(z);

    return 0;
}
