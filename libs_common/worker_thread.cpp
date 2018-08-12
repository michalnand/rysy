#include <worker_thread.h>
#include <unistd.h>
#include <iostream>


WorkerThread::WorkerThread()
{
  id = (unsigned long int)this;

  is_running = true;

  waiting_mutex.lock();
  done_mutex.unlock();

  WorkerThread_mutex.lock();
  idx = WorkerThread_idx;
  WorkerThread_idx++;
  WorkerThread_mutex.unlock();

  thread = new std::thread(&WorkerThread::thread_main, this);
}

WorkerThread::~WorkerThread()
{
 if (thread != nullptr)
 {
  is_running = false;
  waiting_mutex.unlock();

  thread->join();
  delete thread;
 }
}



void WorkerThread::run()
{
   done_mutex.lock();
   waiting_mutex.unlock();
}

void WorkerThread::synchornize()
{
   done_mutex.lock();
}

bool WorkerThread::is_done()
{
 if (done_mutex.try_lock())
  return true;
 return false;
}

unsigned int WorkerThread::get_id()
{
  return id;
}

void WorkerThread::thread_main()
{
 while (1)
 {
  waiting_mutex.lock();
  if (is_running == false)
  {
    done_mutex.unlock();
    break;
  }


  main();
  done_mutex.unlock();
 }
}

void WorkerThread::main()
{
  std::cout << "WorkerhThread dummy main " << idx << " " << id << "\n";
  sleep(1);
}
