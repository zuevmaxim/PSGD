//
// Created by Maksim.Zuev on 29.05.2022.
//

#ifndef PSGD_BARRIER_T_H
#define PSGD_BARRIER_T_H

#include <pthread.h>
#include <atomic>

/*
 * The purpose of this file is to allow the threading tools to be ported to
 * MacOSX which does not support pthread barriers. This is a naive barrier
 * implementation for the port which defaults to use pthreads when possible.
 */


#ifdef __APPLE__ \
    // we have to use our own barrier and timer
struct barrier_t {
    pthread_mutex_t mux;
    pthread_cond_t cond;
    int total;
    std::atomic<int> current;
};
#else
typedef pthread_barrier_t barrier_t;
#endif

int barrier_init(barrier_t* b, void* attr, int count) {
#ifdef __APPLE__
    pthread_mutex_init(&b->mux, NULL);
    pthread_cond_init(&b->cond, NULL);
    b->current.store(count);
    b->total = count;
    return 0;
#else
    return pthread_barrier_init(b, (pthread_barrierattr_t*) attr, count);
#endif
}

int barrier_wait(barrier_t* b) {
#ifdef __APPLE__
    pthread_mutex_lock(&b->mux);
    b->current.fetch_add(-1);
    if (b->current.load() == 0) {
      // reset the count
      b->current.store(b->total);
      // wake up everyone, we're the last to the fence
      pthread_cond_broadcast(&b->cond);
      pthread_mutex_unlock(&b->mux);
      return 0;
    }
    // otherwise, wait the fence
    while (b->current.load() != b->total) {
        pthread_cond_wait(&b->cond, &b->mux);
    }
    // release the mux
    pthread_mutex_unlock(&b->mux);
    return 0;
#else
    return pthread_barrier_wait(b);
#endif
}


int barrier_destroy(barrier_t* b) {
#ifdef __APPLE__
    return -1;
#else
    return pthread_barrier_destroy(b);
#endif
}


#endif //PSGD_BARRIER_T_H
