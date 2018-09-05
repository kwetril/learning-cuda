#include "timer.hpp"

namespace learn_cuda
{

void Timer::Start()
{
  startTime = std::chrono::high_resolution_clock::now();
}

double Timer::Stop()
{
  std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = endTime - startTime;
  return duration.count() * 1000.;
}

}
