#ifndef _TIMER_H_
#define _TIMER_H_


class Timer
{
  private:
    double time_start, time_stop;

  public:
    Timer();
    ~Timer();

    double get_time();
    double get_duration();
    void start();
    void stop();

    void sleep_ms(unsigned int ms);
};


extern class Timer timer;

#endif
