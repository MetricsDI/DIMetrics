#include "lcs.h"

#include <THC/THC.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


template <typename scalar_t>
__global__ void lcs_distance_kernel(
        const scalar_t* __restrict__ source,
        const scalar_t* __restrict__ target,
        const int* __restrict__ source_length,
        const int* __restrict__ target_length,
        const size_t source_size,
        const size_t target_size,
        int* __restrict__ operations) {

    extern __shared__ short table[];

    const int i = blockIdx.x;
    int hyp_size = source_length[i];
    int ref_size = target_length[i];

    const scalar_t* hyp_begin = source + i * source_size;
    const scalar_t* ref_begin = target + i * target_size;


    int tp;
    int fp;
    int fn;

    if (hyp_size != 0 && ref_size != 0){
        for (int i=0; i < ref_size+1; i++){
            table[i*(hyp_size + 1)] = 0;
        }
        for (int j=0; j< hyp_size+1; j++){
            table[j] = 0;
        }
        auto r_iter = (scalar_t*)ref_begin;
        for (int r=1; r < ref_size+1; r++){
            auto h_iter = (scalar_t*)hyp_begin;
            for (int h=1; h < hyp_size+1; h++){
                if (*r_iter == *h_iter){
                    table[r*(hyp_size+1) + h] = 1 + table[(r-1)*(hyp_size+1) + h-1];
                }
                else{
                    int max1 = table[(r-1)*(hyp_size+1) + h];
                    int max2 = table[r*(hyp_size+1) + h-1];
                    if (max1>max2){
                        table[r*(hyp_size+1) + h] = max1;
                    }
                    else{
                        table[r*(hyp_size+1) + h] = max2;
                    }
                    
                }
                ++h_iter;
            }
            ++r_iter;
        
        }
        tp= table[ref_size*(hyp_size+1) + hyp_size];

        fp = hyp_size - tp; 
        fn = ref_size - tp; 
    }
    else if (hyp_size == 0){
        fn = ref_size;
        tp = 0;
        fp = 0;
    }
    else{
        fp = hyp_size;
        tp = 0;
        fn = 0;
    }
   
    operations[i*3+0] = tp;
    operations[i*3+1] = fp;
    operations[i*3+2] = fn;
}

torch::Tensor LCSDistanceCuda(
        torch::Tensor source,
        torch::Tensor target,
        torch::Tensor source_length,
        torch::Tensor target_length) {

    const auto batch_size = source.size(0);
    const auto shared_size = (target.size(1) + 1) * (source.size(1) + 1)  * sizeof(short);

    at::TensorOptions options(source.device());

    options = options.dtype(at::ScalarType::Int);

    auto operations = torch::empty({batch_size, 3}, options);

    auto stream = at::cuda::getCurrentCUDAStream(source.device().index());

    AT_DISPATCH_ALL_TYPES(source.scalar_type(), "lcs_distance", ([&] {
        lcs_distance_kernel<scalar_t><<<batch_size, 1, shared_size, stream>>>(
            source.data<scalar_t>(),
            target.data<scalar_t>(),
            source_length.data<int>(),
            target_length.data<int>(),
            source.size(1),
            target.size(1),
            operations.data<int>());
    }));

    return operations;
}



