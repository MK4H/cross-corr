#pragma once

// Warp size builtin from CUDA is not a constexpr, as it
// is represented as builtin register in PTX
// This can later be changed to a constant provided during compilation
constexpr unsigned int warp_size = 32;
