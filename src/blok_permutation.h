//
// Created by Maksim.Zuev on 29.05.2022.
//

#ifndef PSGD_BLOK_PERMUTATION_H
#define PSGD_BLOK_PERMUTATION_H

#include "types.h"
#include <vector>
#include <atomic>
#include <random>
#include <chrono>
#include <algorithm>

class permutation;

class perm_node {
  std::atomic<perm_node*> next;
public:
  const uint size;
  uint* permutation;

  perm_node(uint size) : next(NULL), size(size) {
      permutation = new uint[size];
      FOR_N(i, size) {
          permutation[i] = i;
      }
      const uint seed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      std::mt19937 gen;
      gen.seed(seed);
      std::shuffle(permutation, permutation + size, gen);
  }

  ~perm_node() {
      perm_node* next_node = next.load();
      if (next_node != NULL) delete next_node;
      delete permutation;
  }

  perm_node* gen_next() {
      perm_node* next_node = next.load();
      if (next_node != NULL) return next_node;

      perm_node* const new_node = new perm_node(size);
      perm_node* cur_node = this;
      while (true) {
          perm_node* cur_next = NULL;
          if (cur_node->next.compare_exchange_strong(cur_next, new_node)) break;
          cur_node = cur_next;
      }
      return next.load();
  }
};

class permutation {
  const uint total_blocks;
  std::vector<perm_node*> roots;

public:
  permutation(uint nodes, uint total_blocks) : total_blocks(total_blocks) {
      roots.resize(nodes);
      FOR_N(node, nodes) {
          RUN_NUMA_START(node)
              roots[node] = new perm_node(total_blocks);
          RUN_NUMA_END
      }
  }

  ~permutation() {
      FOR_N(i, roots.size()) {
          delete roots[i];
      }
  }

  uint get_total_blocks() {
      return total_blocks;
  }

  perm_node* get_basic_permutation(uint node) {
      return roots[node];
  }
};


#endif //PSGD_BLOK_PERMUTATION_H
