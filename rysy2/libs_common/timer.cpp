#include "timer.h"


#include <thread>
#include <chrono>
#include <ratio>


Timer::Timer()
{
  time_start = get_time();
  time_stop = time_start;
}


Timer::~Timer()
{

}

double Timer::get_time()
{

  double now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

  return now;
}

double Timer::get_duration()
{
  return time_stop - time_start;
}

void Timer::start()
{
  time_start = get_time();
}

void Timer::stop()
{
  time_stop = get_time();
}

void Timer::sleep_ms(unsigned int ms_time)
{
  std::this_thread::sleep_for(std::chrono::milliseconds(ms_time));
}
