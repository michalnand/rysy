#include "timer.h"

#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>

class Timer timer;


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
  struct timeval time;

  double res;

  gettimeofday(&time, NULL);

  res = (time.tv_sec*1000.0 + time.tv_usec/1000.0);

  return res;
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

void Timer::sleep_ms(unsigned int ms)
{
  usleep(ms*1000);
}
