#include <benchmark/benchmark.h>
#include <core/platform/threadpool.h>
#include <core/util/thread_utils.h>
#include <core/session/onnxruntime_c_api.h>
#include <core/platform/Barrier.h>

using namespace onnxruntime;
using namespace onnxruntime::concurrency;

static void BM_CreateThreadPool(benchmark::State& state) {
  for (auto _ : state) {
    ThreadPool tp(&onnxruntime::Env::Default(), onnxruntime::ThreadOptions(), ORT_TSTR(""), 48, true);
  }
}
BENCHMARK(BM_CreateThreadPool)->UseRealTime()->Unit(benchmark::TimeUnit::kMillisecond);

//On Xeon W-2123 CPU, it takes about 2ns for each iteration
#ifdef _WIN32
#pragma optimize( "", off )
#else
#pragma GCC push_options
#pragma GCC optimize ("O0")
#endif
void SimpleForLoop(ptrdiff_t first, ptrdiff_t last){
    size_t sum = 0;
    for(;first!=last;++first){
        ++sum;
    }
}
#ifdef _WIN32
#pragma optimize( "", on )
#else
#pragma GCC pop_options
#endif

static void BM_ThreadPoolParallelFor(benchmark::State& state) {
  const size_t len = state.range(0);
  OrtThreadPoolParams tpo;
  std::unique_ptr<concurrency::ThreadPool> tp(concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo,nullptr));
  for (auto _ : state) {
    tp->ParallelFor(len,200,SimpleForLoop);
  }
}
BENCHMARK(BM_ThreadPoolParallelFor)->UseRealTime()->Unit(benchmark::TimeUnit::kNanosecond)->Arg(100)->Arg(1000)->Arg(10000)->Arg(20000)->Arg(40000)->Arg(80000);

static void BM_SimpleForLoop(benchmark::State& state) {
  const size_t len = state.range(0);
  for (auto _ : state) {
   SimpleForLoop(0,len);
  }
}
BENCHMARK(BM_SimpleForLoop)->Unit(benchmark::TimeUnit::kNanosecond)->Arg(100);


static void TestPartitionWork(std::ptrdiff_t ThreadId, std::ptrdiff_t ThreadCount, std::ptrdiff_t TotalWork,
                            std::ptrdiff_t* WorkIndex, std::ptrdiff_t* WorkRemaining) {
    const std::ptrdiff_t WorkPerThread = TotalWork / ThreadCount;
    const std::ptrdiff_t WorkPerThreadExtra = TotalWork % ThreadCount;

    if (ThreadId < WorkPerThreadExtra) {
      *WorkIndex = (WorkPerThread + 1) * ThreadId;
      *WorkRemaining = WorkPerThread + 1;
    } else {
      *WorkIndex = WorkPerThread * ThreadId + WorkPerThreadExtra;
      *WorkRemaining = WorkPerThread;
    }
  }

static void BM_SimpleScheduleWait(benchmark::State& state) {
  const size_t len = state.range(0);
  OrtThreadPoolParams tpo;
  std::unique_ptr<concurrency::ThreadPool> tp(concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo,nullptr));
  size_t threads = tp->NumThreads();

  for (auto _ : state) {
    onnxruntime::Barrier barrier(static_cast<unsigned int>(len));
     for (std::ptrdiff_t id = 0; id < threads; ++id) {
         tp->Schedule([id,threads,len](){
              std::ptrdiff_t start, work_remaining;
              TestPartitionWork(id, threads, len, &start, &work_remaining);
              SimpleForLoop(start,start+work_remaining);
         });
     }  
  }
}
BENCHMARK(BM_SimpleScheduleWait)->UseRealTime()->Unit(benchmark::TimeUnit::kNanosecond)->Arg(100)->Arg(1000)->Arg(10000)->Arg(20000)->Arg(40000)->Arg(80000);