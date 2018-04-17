// Generated by tfcompile, the TensorFlow graph compiler.  DO NOT EDIT!
//
// This header was generated via ahead-of-time compilation of a TensorFlow
// graph.  An object file corresponding to this header was also generated.
// This header gives access to the functionality in that object file.
//
// clang-format off

#ifndef TFCOMPILE_GENERATED___tensorflow_ClusEllipseSigma_H_  // NOLINT(build/header_guard)
#define TFCOMPILE_GENERATED___tensorflow_ClusEllipseSigma_H_  // NOLINT(build/header_guard)


#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "tensorflow/core/platform/types.h"

namespace Eigen { struct ThreadPoolDevice; }
namespace xla { class ExecutableRunOptions; }

// (Implementation detail) Entry point to the function in the object file.
extern "C" void __tensorflow_ClusEllipseSigma(
    void* result, const xla::ExecutableRunOptions* run_options,
    const void** args, void** temps, tensorflow::int64* profile_counters);




// ClusEllipseSigma represents a computation previously specified in a
// TensorFlow graph, now compiled into executable code. This extends the generic
// XlaCompiledCpuFunction class with statically type-safe arg and result
// methods. Usage example:
//
//   ClusEllipseSigma computation;
//   // ...set args using computation.argN methods
//   CHECK(computation.Run());
//   // ...inspect results using computation.resultN methods
//
// The Run method invokes the actual computation, with inputs read from arg
// buffers, and outputs written to result buffers. Each Run call may also use
// a set of temporary buffers for the computation.
//
// By default each instance of this class manages its own arg, result and temp
// buffers. The AllocMode constructor parameter may be used to modify the
// buffer allocation strategy.
//
// Under the default allocation strategy, this class is thread-compatible:
// o Calls to non-const methods require exclusive access to the object.
// o Concurrent calls to const methods are OK, if those calls are made while it
//   is guaranteed that no thread may call a non-const method.
//
// The logical function signature is:
//   (arg0: f32[1,16]) -> (f32[1,2])
//
// Memory stats:
//   arg bytes total:    64
//   arg bytes aligned:  64
//   temp bytes total:   1040
//   temp bytes aligned: 1152
class ClusEllipseSigma : public tensorflow::XlaCompiledCpuFunction {
 public:
  // Number of input arguments for the compiled computation.
  static constexpr size_t kNumArgs = 1;

  // Byte size of each argument buffer. There are kNumArgs entries.
  static const intptr_t* ArgSizes() {
    static constexpr intptr_t kArgSizes[kNumArgs] = {64};
    return kArgSizes;
  }

  // Returns static data used to create an XlaCompiledCpuFunction.
  static const tensorflow::XlaCompiledCpuFunction::StaticData& StaticData() {
    static XlaCompiledCpuFunction::StaticData* kStaticData = [](){
      XlaCompiledCpuFunction::StaticData* data =
        new XlaCompiledCpuFunction::StaticData;
      data->raw_function = __tensorflow_ClusEllipseSigma;
      data->arg_sizes = ArgSizes();
      data->num_args = kNumArgs;
      data->temp_sizes = TempSizes();
      data->num_temps = kNumTemps;
      data->result_index = kResultIndex;
      data->arg_names = StaticArgNames();
      data->result_names = StaticResultNames();
      data->program_shape = StaticProgramShape();
      return data;
    }();
    return *kStaticData;
  }

  ClusEllipseSigma(AllocMode alloc_mode = AllocMode::ARGS_RESULTS_PROFILES_AND_TEMPS)
      : XlaCompiledCpuFunction(StaticData(), alloc_mode) {}

  ClusEllipseSigma(const ClusEllipseSigma&) = delete;
  ClusEllipseSigma& operator=(const ClusEllipseSigma&) = delete;

  // Arg methods for managing input buffers. Buffers are in row-major order.
  // There is a set of methods for each positional argument, with the following
  // general form:
  //
  // void set_argN_data(void* data)
  //   Sets the buffer of type T for positional argument N. May be called in
  //   any AllocMode. Must be called before Run to have an affect. Must be
  //   called in AllocMode::RESULTS_PROFILES_AND_TEMPS_ONLY for each positional
  //   argument, to set the argument buffers.
  //
  // T* argN_data()
  //   Returns the buffer of type T for positional argument N.
  //
  // T& argN(...dim indices...)
  //   Returns a reference to the value of type T for positional argument N,
  //   with dim indices specifying which value. No bounds checking is performed
  //   on dim indices.

  void set_arg0_data(void* data) {
    set_arg_data(0, data);
  }
  float* arg0_data() {
    return static_cast<float*>(arg_data(0));
  }
  float& arg0(size_t dim0, size_t dim1) {
    return (*static_cast<float(*)[1][16]>(
        arg_data(0)))[dim0][dim1];
  }
  const float* arg0_data() const {
    return static_cast<const float*>(arg_data(0));
  }
  const float& arg0(size_t dim0, size_t dim1) const {
    return (*static_cast<const float(*)[1][16]>(
        arg_data(0)))[dim0][dim1];
  }

  // Result methods for managing output buffers. Buffers are in row-major order.
  // Must only be called after a successful Run call. There is a set of methods
  // for each positional result, with the following general form:
  //
  // T* resultN_data()
  //   Returns the buffer of type T for positional result N.
  //
  // T& resultN(...dim indices...)
  //   Returns a reference to the value of type T for positional result N,
  //   with dim indices specifying which value. No bounds checking is performed
  //   on dim indices.
  //
  // Unlike the arg methods, there is no set_resultN_data method. The result
  // buffers are managed internally, and may change after each call to Run.

  float* result0_data() {
    return static_cast<float*>(result_data(0));
  }
  float& result0(size_t dim0, size_t dim1) {
    return (*static_cast<float(*)[1][2]>(
        result_data(0)))[dim0][dim1];
  }
  const float* result0_data() const {
    return static_cast<const float*>(result_data(0));
  }
  const float& result0(size_t dim0, size_t dim1) const {
    return (*static_cast<const float(*)[1][2]>(
        result_data(0)))[dim0][dim1];
  }

  float* result_output_node0_data() {
    return static_cast<float*>(result_data(0));
  }
  float& result_output_node0(size_t dim0, size_t dim1) {
    return (*static_cast<float(*)[1][2]>(
        result_data(0)))[dim0][dim1];
  }
  const float* result_output_node0_data() const {
    return static_cast<const float*>(result_data(0));
  }
  const float& result_output_node0(size_t dim0, size_t dim1) const {
    return (*static_cast<const float(*)[1][2]>(
        result_data(0)))[dim0][dim1];
  }

 private:
  // Number of result and temporary buffers for the compiled computation.
  static constexpr size_t kNumTemps = 4;
  // The 0-based index of the result tuple in the temporary buffers.
  static constexpr size_t kResultIndex = 2;

  // Byte size of each result / temporary buffer. There are kNumTemps entries.
  static const intptr_t* TempSizes() {
    static constexpr intptr_t kTempSizes[kNumTemps] = {-1, 8, 8, 1024};
    return kTempSizes;
  }

  // Array of names of each positional argument, terminated by nullptr.
  static const char** StaticArgNames() {
    return nullptr;
  }

  // Array of names of each positional result, terminated by nullptr.
  static const char** StaticResultNames() {
    return nullptr;
  }

  // Shape of the args and results.
  static const xla::ProgramShape* StaticProgramShape() {
    static const xla::ProgramShape* kShape = nullptr;
    return kShape;
  }
};


#endif  // TFCOMPILE_GENERATED___tensorflow_ClusEllipseSigma_H_

// clang-format on
