#include "lcs.h"

#include <torch/types.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



torch::Tensor LCSDistance(
        torch::Tensor source,
        torch::Tensor target,
        torch::Tensor source_length,
        torch::Tensor target_length) {

    CHECK_INPUT(source);
    CHECK_INPUT(target);
    CHECK_INPUT(source_length);
    CHECK_INPUT(target_length);

    return LCSDistanceCuda(source, target, source_length, target_length);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lcs_distance", &LCSDistance, "LCS distance");
}