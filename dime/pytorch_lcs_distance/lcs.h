#pragma once

#include <torch/extension.h>

torch::Tensor LCSDistanceCuda(
        torch::Tensor source,
        torch::Tensor target,
        torch::Tensor source_length,
        torch::Tensor target_length);