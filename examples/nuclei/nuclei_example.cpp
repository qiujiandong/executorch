/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2023-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* This is an example ExecuTorch runner running on Arm Cortex-M and Ethos-U
 * based hardware. This example tries to illustrate a few ways to use ExecuTorch
 * and you can use it as is or remove the unneeded parts. Please use this code
 * as inspiration.
 *
 * Some defines used to configure the code:
 *
 * ET_MODEL_PTE_ADDR  - Where in memory/flash your PTE model data is, if
 *                      not set the model is supposed to have been converted to
 *                      a c-array named model_pte and put into model_pte.h
 *                      this is placed in network_model_sec linker section
 *                      that is controlled by your memory mode via the
 *                      ETHOSU_MODEL cmake parameter.
 *                      If SEMIHOSTING is define this is not used
 * ET_NUM_INFERENCES  - Numbers of times to run the inference
 * ET_LOG_DUMP_INPUT  - Control if you want input to be dumped to the log.
 * ET_LOG_DUMP_OUTPUT     - Control if you want output to be dumped to the log.
 *
 * Devtool BundleIO: Use Bundle PTE with input and reference output included to
 * check if it matches.
 *
 * ET_BUNDLE_IO       - Build in Devtools BundleIO, this makes it possible to
 *                      use bpte with bundled input and output refdata to
 *                      compare output.
 *                      See also ET_ATOL and ET_RTOL
 *   ET_ATOL              - The atol used to compare the output and ref data
 * when using ET_BUNDLE_IO ET_RTOL              - The rtol used to compare the
 * output and ref data when using ET_BUNDLE_IO
 *
 * Devtools ETDump: Speed and dumping output
 *
 * ET_EVENT_TRACER_ENABLED       - Build in Devtools ETDump event trace code
 *                                 to generate cycle data and print it base64
 *                                 coded in the log so you can get it out of
 *                                 your embedded target. This can be used to
 *                                 benchmark where time is spent. If you run
 *                                 on Ethos-U the delegate/commandstream is
 *                                 run in one go, this means that per op
 *                                 measurements is not possible.
 *  ET_DUMP_OUTPUTS              - Collect and print outputs as a base64 buffer
 *                                 in the log, see ExecuTorch Devtools for more
 *                                 info. (Requires ET_EVENT_TRACER_ENABLED)
 *  ET_DUMP_INTERMEDIATE_OUTPUTS - Collect and print intermediate outputs as a
 *                                 base64 buffer in the log, see ExecuTorch
 *                                 Devtools for more info.
 *                                 (Requires ET_EVENT_TRACER_ENABLED)
 *  ET_DEBUG_BUFFER_SIZE         - Override the size of memory area used by
 *                                 ET_DUMP_OUTPUTS or
 * ET_DUMP_INTERMEDIATE_OUTPUTS
 *
 * Warning: CPU time measurements is NOT possible in the FVP simulator and a
 * real target or FPGA must be used. NPU number are roughly OK, and can be used
 * as guidance if timeing adaptor values are set correctly.
 *
 * SEMIHOSTING - When using the FVP simulator it can be built to access your dev
 *               machines filesystem, this is used for testing models in
 *               unittest/pytest and a special version of the runner is built
 *               to read model and input as files and output is saved to the
 *               filesystem. The backends/arm/test/setup_testing.sh script will
 *               build this for you so you can use it from pytest to test with
 *               the FVP simulator.
 *
 * Memory areas used:
 *    You might want to configure this differently on your HW, like maybe all
 *    left over memory after code is linked. This needs to be big enough to fit
 *    and run your model. In our example using the FVP simulator we have much
 *    memory and set this quite high to be able to test larger models.
 *    Regarding heap/mallocs type of allocation from ExecuTorch,
 *    et_pal_allocate() is not implemented or needed.
 *
 * ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE            - Size of memory area
 *                                                          used when setting up
 *                                                          the model
 * ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE - Size of memory area
 *                                                          used when running
 *                                                          inferences
 */

#include <errno.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(ET_BUNDLE_IO)
#include <executorch/devtools/bundled_program/bundled_program.h>
#endif

/**
 * This header file is generated by the build process based on the .pte file
 * specified in the ET_PTE_FILE_PATH variable to the cmake build.
 * Control of the action of the .pte, it's use of operators and delegates, and
 * which are included in the bare metal build are also orchestrated by the
 * CMakeLists file. For example use see examples/arm/run.sh
 *
 * e.g. This includes the pte as a big chunk of data struct into this file
 */
#include "model_pte.h"

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::Tag;
using executorch::runtime::TensorInfo;
using executorch::runtime::toString;
#if defined(ET_BUNDLE_IO)
using executorch::bundled_program::compute_method_output_error_stats;
using executorch::bundled_program::ErrorStats;
using executorch::bundled_program::verify_method_outputs;
#endif

/**
 * The method_allocation_pool should be large enough to fit the setup, input
 * used and other data used like the planned memory pool (e.g. memory-planned
 * buffers to use for mutable tensor data) In this example we run on a
 * Corstone-3xx FVP so we can use a lot of memory to be able to run and test
 * large models if you run on HW this should be lowered to fit into your
 * availible memory.
 */
#if !defined(ET_RISCV_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE)
#define ET_RISCV_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE (60 * 1024 * 1024)
#endif
const size_t method_allocation_pool_size =
    ET_RISCV_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE;
unsigned char __attribute__((
    section("input_data_sec"),
    aligned(16))) method_allocation_pool[method_allocation_pool_size];

#if defined(ET_BUNDLE_IO)

const size_t testset_idx = 0; // BundleIO test indexes to test if used

#if defined(ET_ATOL)
const float et_atol = ET_ATOL;
#else
const float et_atol = 0.01;
#endif

#if defined(ET_RTOL)
const float et_rtol = ET_RTOL;
#else
const float et_rtol = 0.01;
#endif

#endif

#if defined(ET_NUM_INFERENCES)
const int num_inferences = ET_NUM_INFERENCES;
#else
const int num_inferences = 1;
#endif

/**
 * The temp_allocation_pool is used for allocating temporary data during kernel
 * or delegate execution. This will be reset after each kernel or delegate call.
 * Currently a MemoryAllocator is used but a PlatformMemoryAllocator is probably
 * a better fit.
 *
 * The Corstone-300/Corstone-320 platforms have 2MB/4MB of SRAM respectively.
 * For Shared_Sram, ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE is
 * 2MB and the linker script places the .bss.tensor_arena symbol in the SRAM.
 * For Dedicated_Sram, the .bss.tensor_arena symbol is placed in the DDR in the
 * linker script. Hence, we allocate 128MB in DDR and 384KB in the SRAM
 * (.bss.ethosu_scratch is placed in the SRAM). The examples/arm/CMakeLists.txt
 * contains the logic for the sizes of
 * ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE and
 * ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE
 */
#define ET_RISCV_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE (0x200000)
const size_t temp_allocation_pool_size =
    ET_RISCV_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE;
unsigned char __attribute__((
    section(".bss.tensor_arena"),
    aligned(16))) temp_allocation_pool[temp_allocation_pool_size];
#if defined(ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
extern "C" {
size_t ethosu_fast_scratch_size =
    ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE;
unsigned char __attribute__((section(".bss.ethosu_scratch"), aligned(16)))
dedicated_sram[ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE];
unsigned char* ethosu_fast_scratch = dedicated_sram;
}
#endif

/**
 * Implementation of the et_pal_<funcs>()
 *
 * This functions are hardware adaption type of functions for things like
 * time/logging/memory allocation that could call your RTOS or need to to
 * be implemnted in some way.
 */

ET_NORETURN void et_pal_abort(void) {
#if !defined(SEMIHOSTING)
  __builtin_trap();
#else
  _exit(-1);
#endif
}

/**
 * Emit a log message via platform output (serial port, console, etc).
 */
void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  fprintf(
      stderr,
      "%c [executorch:%s:%lu %s()] %s\n",
      level,
      filename,
      static_cast<unsigned long>(line),
      function,
      message);
}

/**
 * Dynamic memory allocators intended to be used by temp_allocator
 * to implement malloc()/free() type of allocations.
 * Currenyly not used.
 */

void* et_pal_allocate(ET_UNUSED size_t size) {
  return nullptr;
}

void et_pal_free(ET_UNUSED void* ptr) {}

namespace {

/// Lightweight heapless container that constructs and stores a T in-place.
/// Useful when you want to avoid heap allocations but need to delay
/// construction.
template <typename T>
class Box {
 public:
  Box() = default;

  ~Box() {
    if (has_value) {
      ptr()->~T();
    }
  }

  Box(const Box&) = delete;
  Box& operator=(const Box&) = delete;

  /// Destructs the already contained object if it's present and initialize a
  /// new contained object while forwarding its constructor arguments.
  template <typename... Args>
  void reset(Args&&... args) {
    if (has_value) {
      // Destroy the already contained object.
      reinterpret_cast<T*>(mem)->~T();
    }
    // Init the new object.
    new (mem) T(std::forward<Args>(args)...);
    has_value = true;
  }

  /// Returns a reference to the contained object.
  T& value() {
    return *ptr();
  }

  /// Returns a const reference to the contained object.
  const T& value() const {
    return *ptr();
  }

  T* operator->() {
    return ptr();
  }

  const T* operator->() const {
    return ptr();
  }

 private:
  alignas(T) uint8_t mem[sizeof(T)];
  bool has_value = false;

  T* ptr() {
    return reinterpret_cast<T*>(mem);
  }

  const T* ptr() const {
    return reinterpret_cast<const T*>(mem);
  }
};

template <typename ValueType>
void fill_tensor_with_default_value(Tensor& tensor) {
  ValueType fill_value{};
  if constexpr (std::is_same_v<ValueType, bool>) {
    fill_value = true;
  } else {
    fill_value = ValueType(1);
  }

  ValueType* data_ptr = tensor.mutable_data_ptr<ValueType>();
  std::fill(data_ptr, data_ptr + tensor.numel(), fill_value);
}

Error prepare_input_tensors(
    Method& method,
    MemoryAllocator& allocator,
    const std::vector<std::pair<char*, size_t>>& input_buffers) {
  MethodMeta method_meta = method.method_meta();
  size_t num_inputs = method_meta.num_inputs();
  size_t num_allocated = 0;

  EValue* input_evalues = allocator.allocateList<EValue>(num_inputs);
  ET_CHECK_OR_RETURN_ERROR(
      input_evalues != nullptr,
      MemoryAllocationFailed,
      "Could not allocate memory for input evalues.");

  Error err = method.get_inputs(input_evalues, num_inputs);
  ET_CHECK_OK_OR_RETURN_ERROR(err);

  for (size_t i = 0; i < num_inputs; i++) {
    auto tag = method_meta.input_tag(i);
    ET_CHECK_OK_OR_RETURN_ERROR(tag.error());

    if (tag.get() != Tag::Tensor) {
      ET_LOG(
          Debug,
          "Skipping non-tensor input %lu",
          static_cast<unsigned long>(i));
      continue;
    }
    Result<TensorInfo> tensor_meta = method_meta.input_tensor_meta(i);
    ET_CHECK_OK_OR_RETURN_ERROR(tensor_meta.error());

    err = Error::Ok;
    if (input_buffers.size() > 0) {
      auto [buffer, buffer_size] = input_buffers.at(i);
      if (buffer_size != tensor_meta->nbytes()) {
        ET_LOG(
            Error,
            "input size (%d) and tensor size (%d) mismatch!",
            buffer_size,
            tensor_meta->nbytes());
        err = Error::InvalidArgument;
      } else if (input_evalues[i].isTensor()) {
        // Copy the data from the input buffer to the tensor
        Tensor& tensor = input_evalues[i].toTensor();
        std::memcpy(tensor.mutable_data_ptr<int8_t>(), buffer, buffer_size);
      }
    }

    // If input_buffers.size <= 0, we don't have any input, fill it with 1's.
    if (input_buffers.size() <= 0) {
      if (input_evalues[i].isTensor()) {
        Tensor& tensor = input_evalues[i].toTensor();
        switch (tensor.scalar_type()) {
#define HANDLE_SCALAR_TYPE(cpp_type, scalar_name)     \
  case ScalarType::scalar_name:                       \
    fill_tensor_with_default_value<cpp_type>(tensor); \
    break;
          ET_FORALL_SCALAR_TYPES(HANDLE_SCALAR_TYPE)
#undef HANDLE_SCALAR_TYPE
          default:
            ET_LOG(
                Error,
                "Unhandled ScalarType %s",
                toString(tensor.scalar_type()));
            err = Error::InvalidArgument;
            break;
        }
      } else {
        printf("Input[%d]: Not Tensor\n", i);
      }
    }
  }

  return err;
}

/// Holds all state needed for setup and run phases
struct RunnerContext {
  RunnerContext() = default;
  RunnerContext(const RunnerContext& ctx) = delete;
  RunnerContext& operator=(const RunnerContext& ctx) = delete;

  const char* method_name = nullptr;
  // size_t planned_buffer_memsize = 0;
  size_t method_loaded_memsize = 0;
  size_t executor_membase = 0;
  size_t program_data_len = 0;
  size_t input_memsize = 0;
  size_t pte_size = 0;
  bool bundle_io = false;
  Box<BufferDataLoader> loader;
  Box<Program> program;
  Box<MemoryAllocator> method_allocator;
  Box<MemoryAllocator> temp_allocator;
  std::vector<Span<uint8_t>> planned_spans;
  Box<HierarchicalAllocator> planned_memory;
  Box<MemoryManager> memory_manager;
  Box<Result<Method>> method;
#if defined(ET_EVENT_TRACER_ENABLED)
  Box<ETDumpGen> etdump_gen;
#if defined(ET_DUMP_INTERMEDIATE_OUTPUTS) || defined(ET_DUMP_OUTPUTS)
  void* debug_buffer;
#endif
#endif
#if defined(SEMIHOSTING)
  Box<ArmMemoryAllocator> input_file_allocator;
  const char* output_basename = nullptr;
#endif
};

void runner_init(
    RunnerContext& ctx,
    std::vector<std::pair<char*, size_t>> input_buffers,
    size_t pte_size) {
  // Find the offset to the embedded Program.
  const void* program_data = model_pte;
  ctx.program_data_len = pte_size;
  ctx.pte_size = pte_size;

#if defined(ET_BUNDLE_IO)
  ctx.bundle_io = executorch::bundled_program::is_bundled_program(
      const_cast<void*>(program_data), ctx.pte_size);
  if (ctx.bundle_io) {
    Error status = executorch::bundled_program::get_program_data(
        const_cast<void*>(program_data),
        ctx.pte_size,
        &program_data,
        &ctx.program_data_len);
    ET_CHECK_MSG(
        status == Error::Ok,
        "get_program_data() from bundled PTE failed: 0x%x",
        (unsigned int)status);
  }
#endif

  ctx.loader.reset(program_data, ctx.program_data_len);
  auto& loader = ctx.loader.value();
  ET_LOG(
      Info,
      "PTE Model data loaded. Size: %lu bytes.",
      static_cast<unsigned long>(ctx.program_data_len));

  // Parse the program file. This is immutable, and can also be reused
  // between multiple execution invocations across multiple threads.
  Result<Program> program_result = Program::load(&loader);
  ET_CHECK_MSG(
      program_result.ok(),
      "Program loading failed @ %p: 0x%" PRIx32,
      program_data,
      program_result.error());
  ctx.program.reset(std::move(program_result.get()));
  Program& program = ctx.program.value();

  ET_LOG(
      Info,
      "Model buffer loaded, has %lu methods",
      static_cast<unsigned long>(program.num_methods()));

  {
    const auto method_name_result = program.get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    ctx.method_name = *method_name_result;
  }
  ET_LOG(Info, "Running method %s", ctx.method_name);

  Result<MethodMeta> method_meta = program.method_meta(ctx.method_name);
  if (!method_meta.ok()) {
    ET_LOG(
        Info,
        "Failed to get method_meta for %s: 0x%x",
        ctx.method_name,
        (unsigned int)method_meta.error());
  }

  ET_LOG(
      Info,
      "Setup Method allocator pool. Size: %lu bytes.",
      static_cast<unsigned long>(method_allocation_pool_size));

  ctx.method_allocator.reset(
      method_allocation_pool_size, method_allocation_pool);

  ctx.planned_spans.clear();
  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();
  ctx.planned_spans.reserve(num_memory_planned_buffers);
  // size_t planned_buffer_membase = ctx.method_allocator->used_size();

  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    size_t buffer_size =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    ET_LOG(
        Info,
        "Setting up planned buffer %lu, size %lu.",
        static_cast<unsigned long>(id),
        static_cast<unsigned long>(buffer_size));

    /* Move to it's own allocator when MemoryPlanner is in place. */
    /* Ethos-U driver requires 16 bit alignment. */
    uint8_t* buffer = reinterpret_cast<uint8_t*>(
        ctx.method_allocator->allocate(buffer_size, 16UL));
    ET_CHECK_MSG(
        buffer != nullptr,
        "Could not allocate memory for memory planned buffer size %lu",
        static_cast<unsigned long>(buffer_size));
    ctx.planned_spans.push_back({buffer, buffer_size});
  }

  // ctx.planned_buffer_memsize =
  //     ctx.method_allocator->used_size() - planned_buffer_membase;

  Span<Span<uint8_t>> planned_memory_span;
  if (!ctx.planned_spans.empty()) {
    planned_memory_span =
        Span<Span<uint8_t>>(ctx.planned_spans.data(), ctx.planned_spans.size());
  }
  ctx.planned_memory.reset(planned_memory_span);

  ctx.temp_allocator.reset(temp_allocation_pool_size, temp_allocation_pool);

  ctx.memory_manager.reset(
      &ctx.method_allocator.value(),
      &ctx.planned_memory.value(),
      &ctx.temp_allocator.value());

  // size_t method_loaded_membase = ctx.method_allocator->used_size();

  executorch::runtime::EventTracer* event_tracer_ptr = nullptr;

  ctx.method.reset(program.load_method(
      ctx.method_name, &ctx.memory_manager.value(), event_tracer_ptr));

  if (!ctx.method->ok()) {
    ET_LOG(
        Info,
        "Loading of method %s failed with status 0x%" PRIx32,
        ctx.method_name,
        ctx.method->error());
  }
  // ctx.method_loaded_memsize =
  //     ctx.method_allocator->used_size() - method_loaded_membase;
  // ET_LOG(Info, "Method '%s' loaded.", ctx.method_name);
  //
  // ET_LOG(Info, "Preparing inputs...");
  // size_t input_membase = ctx.method_allocator->used_size();

#if defined(ET_BUNDLE_IO)
  if (ctx.bundle_io) {
    ET_LOG(Info, "Input testset[%d] from bundled bpte", testset_idx);
    Error status = executorch::bundled_program::load_bundled_input(
        *ctx.method.value(),
        const_cast<void*>(reinterpret_cast<const void*>(model_pte)),
        testset_idx);
    ET_CHECK_MSG(
        status == Error::Ok,
        "load_bundled_input failed with status 0x%" PRIx32,
        status);
  } else
#endif
  {
    Error status = ::prepare_input_tensors(
        *ctx.method.value(), ctx.method_allocator.value(), input_buffers);
    ET_CHECK_MSG(
        status == Error::Ok, "Failed to prepare inputs 0x%" PRIx32, status);
  }
  // ctx.input_memsize = ctx.method_allocator->used_size() - input_membase;
  // ctx.executor_membase = ctx.method_allocator->used_size();

  ET_LOG(Info, "Input prepared.");
}

void log_mem_status(RunnerContext& ctx) {
  // size_t executor_memsize =
  //     ctx.method_allocator->used_size() - ctx.executor_membase;

  ET_LOG(
      Info,
      "model_pte_program_size:     %lu bytes.",
      static_cast<unsigned long>(ctx.program_data_len));
  ET_LOG(
      Info,
      "model_pte_loaded_size:      %lu bytes.",
      static_cast<unsigned long>(ctx.pte_size));

  if (ctx.method_allocator->size() != 0) {
    ET_LOG(
        Info,
        "method_allocator_loaded:   %lu bytes",
        static_cast<unsigned long>(ctx.method_loaded_memsize));
    ET_LOG(
        Info,
        "method_allocator_input:    %lu bytes",
        static_cast<unsigned long>(ctx.input_memsize));
  }
  if (ctx.temp_allocator->size() > 0) {
    ET_LOG(
        Info,
        "temp_allocator:            %lu",
        static_cast<unsigned long>(ctx.temp_allocator->size()));
  }
}

void print_outputs(RunnerContext& ctx) {
  std::vector<EValue> outputs(ctx.method.value()->outputs_size());
  ET_LOG(Info, "%lu outputs: ", static_cast<unsigned long>(outputs.size()));
  Error status =
      ctx.method.value()->get_outputs(outputs.data(), outputs.size());
  ET_CHECK(status == Error::Ok);

  // Print the outputs.
  for (int i = 0; i < outputs.size(); ++i) {
    if (outputs[i].isTensor()) {
      Tensor tensor = outputs[i].toTensor();
#if defined(ET_LOG_DUMP_OUTPUT)
      // The output might be collected and parsed so printf() is used instead
      // of ET_LOG() here
      for (int j = 0; j < tensor.numel(); ++j) {
        if (tensor.scalar_type() == ScalarType::Int) {
          printf(
              "Output[%d][%d]: (int) %d\n",
              i,
              j,
              tensor.const_data_ptr<int>()[j]);
        } else if (tensor.scalar_type() == ScalarType::Float) {
          printf(
              "Output[%d][%d]: (float) %f\n",
              i,
              j,
              tensor.const_data_ptr<float>()[j]);
        } else if (tensor.scalar_type() == ScalarType::Char) {
          printf(
              "Output[%d][%d]: (char) %d\n",
              i,
              j,
              tensor.const_data_ptr<int8_t>()[j]);
        } else if (tensor.scalar_type() == ScalarType::Bool) {
          printf(
              "Output[%d][%d]: (bool) %s (0x%x)\n",
              i,
              j,
              tensor.const_data_ptr<int8_t>()[j] ? "true " : "false",
              tensor.const_data_ptr<int8_t>()[j]);
        }
      }
#endif
    } else {
      printf("Output[%d]: Not Tensor\n", i);
    }
  }
}

void write_etdump(RunnerContext& ctx) {
}

bool verify_result(RunnerContext& ctx, const void* model_pte) {
  bool model_ok = false;
  (void)ctx;
  (void)model_pte;
  // No checking done, assume true
  model_ok = true;
  return model_ok;
}

bool run_model(RunnerContext& ctx, const void* model_pte) {
  Error status;
  ET_LOG(Info, "Starting running %d inferences...", num_inferences);
  int n = 0;
  for (n = 0; n < num_inferences; n++) {
    ET_LOG(Debug, "Running inference number %d", n);
    // Run the model.
    status = ctx.method.value()->execute();
    if (status != Error::Ok) {
      break;
    }
    // Reset the temporary allocator holding the scratch buffer between
    // inferences. We want to reuse the temp_allocator between inferences of the
    // same Ethos-U custom delegate, not allocate memory with every new
    // inference.
    ctx.temp_allocator.reset(temp_allocation_pool_size, temp_allocation_pool);
  }

  ET_CHECK_MSG(
      status == Error::Ok,
      "Execution of method %s failed with status 0x%" PRIx32,
      ctx.method_name,
      status);

  ET_LOG(Info, "%d inferences finished", num_inferences);
  print_outputs(ctx);
  bool model_ok = verify_result(ctx, model_pte);
  ET_LOG(Info, "Model run: %d", model_ok);

  return model_ok;
}

} // namespace

int main(int argc, const char* argv[]) {
  executorch::runtime::runtime_init();
  std::vector<std::pair<char*, size_t>> input_buffers;

  size_t pte_size = sizeof(model_pte);

  RunnerContext ctx;

  // Byte 4-7 is usually a nice magic number that could be good to print to make
  // sure it's OK ETxx for PTE and BPxx for bundled pte where xx is a number.
  ET_LOG(
      Info,
      "PTE @ %p [----%c%c%c%c]",
      model_pte,
      model_pte[4],
      model_pte[5],
      model_pte[6],
      model_pte[7]);

  runner_init(ctx, input_buffers, pte_size);
  bool model_ok = run_model(ctx, model_pte);
  ET_LOG(Info, "Model run: %d", model_ok);

  log_mem_status(ctx);
  write_etdump(ctx);

  ET_CHECK_MSG(model_ok == true, "Problem running model");

  ET_LOG(Info, "Program complete, exiting.\r\n");
  return 0;
}
