#include "GpuExactMorphology.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace {

std::atomic<unsigned long long> g_calls{0};
std::atomic<unsigned long long> g_failures{0};
std::atomic<unsigned long long> g_iterations{0};
std::once_flag g_init_once;

void reportGpuMorphology() {
  std::fprintf(stderr,
               "OPENLPT_GPU_EXACT_MORPHOLOGY calls=%llu failures=%llu "
               "iterations=%llu\n",
               g_calls.load(), g_failures.load(), g_iterations.load());
}

bool check(cudaError_t status) { return status == cudaSuccess; }

__global__ void propagateExact(const double *__restrict__ values,
                               const std::uint8_t *__restrict__ previous,
                               std::uint8_t *__restrict__ next, int rows,
                               int cols, int *changed) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = rows * cols;
  if (index >= size)
    return;

  std::uint8_t result = previous[index];
  if (result != 0U) {
    const int row = index % rows;
    const int col = index / rows;
    const double pixel = values[index];

    // Padding mode 1 ignores neighbors outside the image. The predicate is a
    // Boolean OR, so neighbor visit order cannot affect the result.
    for (int dc = -1; dc <= 1 && result != 0U; ++dc) {
      const int neighbor_col = col + dc;
      if (neighbor_col < 0 || neighbor_col >= cols)
        continue;
      for (int dr = -1; dr <= 1; ++dr) {
        const int neighbor_row = row + dr;
        if (neighbor_row < 0 || neighbor_row >= rows)
          continue;
        const int neighbor = neighbor_col * rows + neighbor_row;
        const double neighbor_value = values[neighbor];
        if (neighbor_value > pixel ||
            (neighbor_value == pixel && previous[neighbor] == 0U)) {
          result = 0U;
          break;
        }
      }
    }
  }

  next[index] = result;
  if (result != previous[index])
    atomicExch(changed, 1);
}

class ThreadContext {
public:
  ~ThreadContext() {
    if (stream_ != nullptr)
      cudaStreamDestroy(stream_);
    cudaFree(values_);
    cudaFree(state_a_);
    cudaFree(state_b_);
    cudaFree(changed_);
  }

  bool run(const double *values, int rows, int cols, bool *output) {
    if (values == nullptr || output == nullptr || rows <= 0 || cols <= 0)
      return false;

    const std::size_t count =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (!initialize() || !reserve(count))
      return false;

    if (!check(cudaMemcpyAsync(values_, values, count * sizeof(double),
                               cudaMemcpyHostToDevice, stream_)) ||
        !check(cudaMemsetAsync(state_a_, 1, count * sizeof(std::uint8_t),
                               stream_)))
      return false;

    std::uint8_t *previous = state_a_;
    std::uint8_t *next = state_b_;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    unsigned long long local_iterations = 0;

    for (;;) {
      if (!check(cudaMemsetAsync(changed_, 0, sizeof(int), stream_)))
        return false;
      propagateExact<<<blocks, threads, 0, stream_>>>(
          values_, previous, next, rows, cols, changed_);
      if (!check(cudaGetLastError()))
        return false;

      int changed = 0;
      if (!check(cudaMemcpyAsync(&changed, changed_, sizeof(int),
                                 cudaMemcpyDeviceToHost, stream_)) ||
          !check(cudaStreamSynchronize(stream_)))
        return false;
      ++local_iterations;
      if (changed == 0)
        break;
      std::uint8_t *temporary = previous;
      previous = next;
      next = temporary;
    }

    static_assert(sizeof(bool) == sizeof(std::uint8_t));
    if (!check(cudaMemcpyAsync(output, next, count * sizeof(std::uint8_t),
                               cudaMemcpyDeviceToHost, stream_)) ||
        !check(cudaStreamSynchronize(stream_)))
      return false;

    g_iterations.fetch_add(local_iterations, std::memory_order_relaxed);
    return true;
  }

private:
  bool initialize() {
    if (initialized_)
      return true;
    if (!check(cudaSetDevice(0)) || !check(cudaStreamCreate(&stream_)))
      return false;
    initialized_ = true;
    return true;
  }

  bool reserve(std::size_t count) {
    if (count <= capacity_)
      return true;

    cudaFree(values_);
    cudaFree(state_a_);
    cudaFree(state_b_);
    cudaFree(changed_);
    values_ = nullptr;
    state_a_ = nullptr;
    state_b_ = nullptr;
    changed_ = nullptr;
    capacity_ = 0;

    if (!check(cudaMalloc(&values_, count * sizeof(double))) ||
        !check(cudaMalloc(&state_a_, count * sizeof(std::uint8_t))) ||
        !check(cudaMalloc(&state_b_, count * sizeof(std::uint8_t))) ||
        !check(cudaMalloc(&changed_, sizeof(int))))
      return false;
    capacity_ = count;
    return true;
  }

  bool initialized_ = false;
  std::size_t capacity_ = 0;
  cudaStream_t stream_ = nullptr;
  double *values_ = nullptr;
  std::uint8_t *state_a_ = nullptr;
  std::uint8_t *state_b_ = nullptr;
  int *changed_ = nullptr;
};

thread_local ThreadContext g_context;

} // namespace

bool openlptGpuExactMorphology(const double *values, int rows, int cols,
                               bool *output) {
  std::call_once(g_init_once, []() {
    std::atexit(reportGpuMorphology);
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess) {
      std::fprintf(stderr,
                   "OPENLPT_GPU_EXACT_MORPHOLOGY device=%s mode=bitwise-compare\n",
                   properties.name);
    }
  });

  g_calls.fetch_add(1, std::memory_order_relaxed);
  if (g_context.run(values, rows, cols, output))
    return true;
  g_failures.fetch_add(1, std::memory_order_relaxed);
  return false;
}
