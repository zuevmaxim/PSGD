//
// Created by Maksim.Zuev on 29.05.2022.
//

#ifndef PSGD_BARRIER_T_H
#define PSGD_BARRIER_T_H

#include <pthread.h>
#include <atomic>
#include <condition_variable>
#include <mutex>

/*
 * The purpose of this file is to allow the threading tools to be ported to
 * MacOSX which does not support pthread barriers. This is a naive barrier
 * implementation for the port which defaults to use pthreads when possible.
 */


#ifdef __APPLE__ \
    // we have to use our own barrier and timer
struct barrier_t {
    std::mutex mux;
    std::condition_variable cond;
    int total;
    std::atomic<int> current;
};
#else
typedef pthread_barrier_t barrier_t;
#endif

int barrier_init(barrier_t* b, void* attr, int count) {
#ifdef __APPLE__
    b->current.store(0);
    b->total = count;
    return 0;
#else
    return pthread_barrier_init(b, (pthread_barrierattr_t*) attr, count);
#endif
}

int barrier_wait(barrier_t* b) {
#ifdef __APPLE__
    std::unique_lock<std::mutex> lock(b->mux);
    const int cur = b->current.fetch_add(1);
    const int start_epoch = cur / b->total;
    const int threshold = (start_epoch + 1) * b->total;
    if (cur + 1 == threshold) {
      // wake up everyone, we're the last to the fence
      b->cond.notify_all();
      return 0;
    }
    // otherwise, wait the fence
    while (b->current.load() < threshold) {
        b->cond.wait(lock);
    }
    return 0;
#else
    return pthread_barrier_wait(b);
#endif
}


int barrier_destroy(barrier_t* b) {
#ifdef __APPLE__
    return 0;
#else
    return pthread_barrier_destroy(b);
#endif
}


#endif //PSGD_BARRIER_T_H
