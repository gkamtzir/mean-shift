CC=nvcc

all: mean_shift_shared mean_shift_global mean_shift_shared_seeds mean_shift_global_seeds

mean_shift_shared: mean_shift_shared.cu
	$(CC) -O3 -o shared mean_shift_shared.cu

mean_shift_global: mean_shift_global.cu
	$(CC) -O3 -o global mean_shift_global.cu

mean_shift_shared_seeds: mean_shift_shared_seeds.cu
	$(CC) -O3 -o shared_seeds mean_shift_shared_seeds.cu

mean_shift_global_seeds: mean_shift_global_seeds.cu
	$(CC) -O3 -o global_seeds mean_shift_global_seeds.cu

clean:
	rm shared global shared_seeds global_seeds
