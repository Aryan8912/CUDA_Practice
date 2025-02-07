#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gemm.h"

#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"
#include "cutlass/gemm/device/gemm_universal_with_absmax.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

using ElementA = cutlass::float_e4m3_t;
using ElementB = cultass::float_e4m3_t;
using ElementOutput = cutlass::float_e4m3_t;
using ElementAuxOutput = ElementOutput;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
static int const kStages = 3;
static int const kAlignmentA = 16;
static int const kAlignmentB = 16;

using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::ReLu,
    ElementOutput,
    ElementAuxOutput,
    8,
    ElementAccumulator,
    ElementAccumulator
    >;

template <typename MathOperator>
using Gemm_ = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 64, 128>, cutlass::gemm::GemmShape<64, 32, 128>, cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages,
    kAlignmentA, kAlignmentB, MathOperator
  >;

using ElementAbsmax = typename EpilogueOutputOp::ElementAbsMax;

// Command line options parsing
struct Options {
    bool help;
    bool error;
    bool reference_check;
    cutlass::gemm::GemmCoord problem_size;

    int iterations;
    int warmup_iterations;

    bool scale_A;
    bool scale_B;
    bool scale_C;

    float alpha;
    float beta;

    Options():
       help(false),
       error(false),
       reference_check(false),
       iterations(20),
       warmup_iterations(5),
       scale_A(true),
       scale_B(true),
       scale_C(true),
       alpha(1.f),
       beta(0.f)
    { }

    // Parse the command line
    void parse(int argc, char const **args){
        cutlass::CommandLine cmd(argc, args);

        if (cmd.check_cmd_line_flag("help")){
            help = true;
            return;
        }

        cmd.get_cmd_line_argument("iterations", iterations, 20);
        cmd.get_cmd_line_argument("warmup_iterations", warmup_iterations, 5);
        cmd.get_cmd_line_argument("reference-check", reference_check, false);
        cmd.get_cmd_line_argument("scale-A", scale_A, true);
        cmd.get_cmd_line_argument("scale-B", scale_B, true);
        cmd.get_cmd_line_argument("scale-C", scale_C, true);
        cmd.get_cmd_line_argument("alpha", alpha, 1.f);
        cmd.get_cmd_line_argument("beta", beta, 0.f);

        int m, n, k;
        cmd.get_cmd_line_argument("m", m, 1024);
        cmd.get_cmd_line_argument("n", n, 1024);
        cmd.get_cmd_line_argument("k", k, 1024);

        problem_size = cutlass::gemm::GemmCoord{m, n, k};
    }
}