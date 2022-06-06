//
// Created by Maksim.Zuev on 29.05.2022.
//

#ifndef PSGD_THREAD_POOL_H
#define PSGD_THREAD_POOL_H

#include "types.h"
#include "cpu_config.h"
#include <pthread.h>
#include <vector>
#include <cassert>
#include "barrier_t.h"


class thread_pool;

struct thread_data {
  thread_pool* tp;
  uint id;
};

typedef void* tp_task_return_t;
typedef void* tp_task_internal_args_t;
typedef tp_task_return_t(* tp_task_t)(tp_task_internal_args_t, unsigned) ;

class thread_pool {
private:
  const uint size;
  uint max_numa_node;
  std::vector <pthread_t> threads;
  std::vector <thread_data> thread_datas;
  std::vector <tp_task_return_t> results;
  std::atomic<bool> stop;
  barrier_t ready;
  barrier_t finished;

  std::atomic <tp_task_t> task;
  std::atomic <tp_task_internal_args_t> args;


  void thread_loop(uint thread_id) {
      config.bind_to_cpu(thread_id);
      while (true) {
          barrier_wait(&ready);
          if (stop.load()) break;
          auto a_task = task.load();
          auto the_args = args.load();
          assert(a_task != NULL);
          results[thread_id] = a_task(the_args, thread_id);
          barrier_wait(&finished);
      }
  }

  static void* thread_run(void* data) {
      thread_data* td = reinterpret_cast<thread_data*>(data);
      td->tp->thread_loop(td->id);
      return NULL;
  }

public:
  thread_pool(uint size) : size(size) {
      threads.resize(size);
      thread_datas.resize(size);
      results.resize(size);
      FOR_N(i, size) {
          thread_datas[i].id = i;
          thread_datas[i].tp = this;
      }

      stop.store(false);
      barrier_init(&ready, NULL, size + 1);
      barrier_init(&finished, NULL, size + 1);
      max_numa_node = 0;
      FOR_N(i, size) {
          max_numa_node = std::max(max_numa_node, config.get_node_for_thread(i));
          pthread_create(&threads[i], NULL, thread_pool::thread_run, reinterpret_cast<void*>(&thread_datas[i]));
      }
  }

  ~thread_pool() {
      stop.store(true);
      barrier_wait(&ready);
      FOR_N(i, size) {
          pthread_join(threads[i], NULL);
      }
      barrier_destroy(&ready);
      barrier_destroy(&finished);
  }

  uint get_size() const {
      return size;
  }

  uint get_numa_count() const {
      return max_numa_node + 1;
  }

  std::vector<tp_task_return_t> execute(tp_task_t hook, tp_task_internal_args_t hook_args) {
      task.store(hook);
      args = hook_args;
      barrier_wait(&ready);
      barrier_wait(&finished);
      task.store(NULL);
      args.store(NULL);
      return results;
  }
};

#endif //PSGD_THREAD_POOL_H
