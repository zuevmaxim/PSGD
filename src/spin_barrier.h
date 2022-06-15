//
// Created by Maksim.Zuev on 15.06.2022.
//

#ifndef PSGD_SPIN_BARRIER_H
#define PSGD_SPIN_BARRIER_H

#include <atomic>

class spin_barrier {
  const uint total;
  std::atomic <uint> counter;

public:
  spin_barrier(uint total) : total(total), counter(0) {}

  void wait() {
      const uint value = counter.fetch_add(1);
      const uint start_epoch = value / total;
      const uint threshold = (start_epoch + 1) * total;
      if (value + 1 == threshold) return;
      while (counter.load() < threshold);
  }
};

#endif //PSGD_SPIN_BARRIER_H
