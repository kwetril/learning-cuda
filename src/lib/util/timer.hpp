#pragma once

#include <chrono>

namespace learn_cuda
{

class Timer
{
public:
  void Start();

  double Stop();

private:
  std::chrono::high_resolution_clock::time_point startTime;
};

}
