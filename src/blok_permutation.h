//
// Created by Maksim.Zuev on 29.05.2022.
//

#ifndef PSGD_BLOK_PERMUTATION_H
#define PSGD_BLOK_PERMUTATION_H

#include "types.h"
#include <vector>
#include <atomic>

class rw_lock {
private:
  std::atomic<int> counter;

public:
  void read_lock() {
      while (counter.fetch_add(1) < 0) {
          counter.fetch_sub(1);
      }
  }

  void read_unlock() {
      counter.fetch_sub(1);
  }

  void write_lock() {
      int expected = 0;
      while (!counter.compare_exchange_strong(expected, INT_MIN));
  }

  void write_unlock() {
      counter.store(0);
  }
};

class read_lock_holder {
private:
  rw_lock& lock;
public:
  read_lock_holder(rw_lock& lock) : lock(lock) {
      lock.read_lock();
  }

  ~read_lock_holder() {
      lock.read_unlock();
  }
};


class perm_node {
public:
  std::vector <uint> permutation;

  perm_node(uint size) {
      permutation.resize(size);
      FOR_N(i, size) {
          permutation[i] = i;
      }
      std::random_shuffle(permutation.begin(), permutation.end());
  }
};

class permutation {
private:
  std::atomic<perm_node*> node;
  rw_lock lock{};
  uint size;

public:
  permutation(uint size) : node(new perm_node(size)), size(size) {
  }

  ~permutation() {
      const perm_node* a_node = node.load();
      delete a_node;
  }

  uint get_permutation(uint index) {
      read_lock_holder h(lock);
      return node.load()->permutation[index];
  }

  void permute() {
      if (size == 1) return;
      perm_node* new_node = new perm_node(size);
      lock.write_lock();
      perm_node* current = node.exchange(new_node);
      lock.write_unlock();
      delete current;
  }
};


#endif //PSGD_BLOK_PERMUTATION_H
