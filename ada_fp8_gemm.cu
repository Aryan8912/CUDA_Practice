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

// Prints the usage statement.
 std::ostream & print_usage(std::ostream &out) const {

    out << "58_ada_fp8_gemm\n\n"
      << "  This example executes a GEMM using Ada FP8 Tensor Core operations. In addition to performing\n"
      << "  a normal GEMM, the kernel performs the following operations:\n"
      << "      Aux = ((alpha * scale_a * scale_b) * accumulator) + ((beta * scale_c) * source) + bias\n"
      << "        D = activation(Aux)\n\n"
      << "      if Aux is fp8:\n"
      << "         abs_max_output = max( abs(aux) | (for every aux in Aux) )\n"
      << "         Aux = scale_aux * Aux\n\n"
      << "      if D is fp8 type:\n"
      << "         abs_max_output = max( abs(d) | (for every d in D) )\n"
      << "         D = scale_d * D\n\n"
      << "Options:\n\n"
      << "  --help                           If specified, displays this usage statement\n\n"
      << "  --m=<int>                        Sets the M dimension of the GEMM\n"
      << "  --n=<int>                        Sets the N dimension of the GEMM\n"
      << "  --k=<int>                        Sets the K dimension of the GEMM\n"
      << "  --scale-A=<bool>                 Whether to apply a scaling factor to operand A (default: true)\n"
      << "  --scale-B=<bool>                 Whether to apply a scaling factor to operand B (default: true)\n"
      << "  --scale-C=<bool>                 Whether to apply a scaling factor to operand C (default: true)\n"
      << "  --iterations=<int>               Number of profiling iterations to perform\n"
      << "  --warmup-iterations=<int>        Number of warmup iterations to perform\n"
      << "  --reference-check=<bool>         If true, performs reference check\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  float gflops(float runtime_s) const {
    // Two flops per multiply-add
    return 2.0f * float(problem_size.product()) / float(1.0e9) / runtime_s;
  }

};

/// Helper class to run the kernel
template <typename Gemm>
struct TestbedRunner {

  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementCompute = typename Gemm::GemmKernel::Epilogue::Output::ElementCompute;
  using ElementScalingFactor = typename Gemm::EpilogueOutputOp::ElementScalingFactor;

  static bool const kScaleAux = Gemm::EpilogueOutputOp::kIsScalingAndAmaxAuxOutputNeeded;
  static bool const kScaleOutput = Gemm::EpilogueOutputOp::kIsScalingAndAmaxOutPutNeeded;

  // Initialization
  cutlass::Distribution::Kind init_A;
  cutlass:: Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;

  cutlass::HostTensor<typename Gemm::ElementA, typename Gemm::LayoutA> tensor_A;
  cutlass::HostTensor<typename Gemm::ElementB, typename Gemm::LayoutB> tensor_B;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_C;
  cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementAuxOutput, typename Gemm::LayoutC> tensor_Aux;
  cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementOutput, typename Gemm::LayoutC> tensor_D;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_Vector;
  cutlass::HostTensor<ElementAccumulator, typename Gemm::LayoutC> tmp_D;
  cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementOutput, typename Gemm::LayoutC> reference_D;
  cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementAuxOutput, typename Gemm::LayoutC> reference_Aux;
  cutlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_A;
  cutlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_B;
  cutlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_C;
  cutlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_D;
  cutlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_Aux;
  cutlass::HostTensor<ElementAbsmax, typename Gemm::LayoutC> abs_max_Aux;
  cutlass::HostTensor<ElementAbsmax, typename Gemm::LayoutC> abs_max_D;
  cutlass::HostTensor<ElementAbsmax, typename Gemm::LayoutC> reference_abs_max_Aux;
  cutlass::HostTensor<ElementAbsmax, typename Gemm::LayoutC> reference_abs_max_D;

  // Methods 

  TestbedRunner(
    bool scaleA = true,
    bool scaleB = true,
    bool scaleC = true,
    cutlass::Distribution::Kind init_A = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
   init_A(init_A_), init_B(init_B), init_C(init_C_), seed(seed_) { }

   // Helper to initialize scaling factors
   template <typename Element, typename Layout>
   bool initialize_scale_factor(cutlass::TensorView<Element, Layout> view, uint64_t seed, int bits=0){
    cutlass::reference::host::TensorFillRandomUniform(view, seed, double(1.), double(0.), bits);
    return true;
   }
}

// 266