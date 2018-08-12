#ifndef _WORKER_THREAD_H_
#define _WORKER_THREAD_H_

#include <thread>
#include <mutex>



class WorkerThread
{
	public:
		WorkerThread();
		virtual ~WorkerThread();

		void run();
		void synchornize();
		bool is_done();
		unsigned int get_id();

	public:

	  virtual void main();

	private:
		unsigned long int id;
		unsigned int idx;

	private:

		bool is_running;
		std::thread *thread;
		std::mutex waiting_mutex, done_mutex;

		static unsigned int WorkerThread_idx;
		static std::mutex WorkerThread_mutex;

  	private:
			void thread_main();
};



#endif
