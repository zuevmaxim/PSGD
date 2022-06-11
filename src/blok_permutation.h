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

class perm_node {
  std::atomic<perm_node*> next;
  
  static const uint* generate_permutation(uint size) {
      uint* permutation = new uint[size];
      FOR_N(i, size) {
          permutation[i] = i;
      }
      shuffle(permutation, size);
      return permutation;
  }
  
public:
  const uint size;
  const uint* const permutation;

  perm_node(uint size) : next(NULL), size(size), permutation(generate_permutation(size)) {}

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

  static void shuffle(uint* permutation, uint size) {
      const auto ns = std::chrono::high_resolution_clock::now().time_since_epoch();
      const uint seed = std::chrono::duration_cast<std::chrono::nanoseconds>(ns).count();
      std::mt19937 gen;
      gen.seed(seed);
      std::shuffle(permutation, permutation + size, gen);
  }
};

class permutation {
  perm_node* cluster_permutation;
  std::vector<perm_node*> roots;

public:
  permutation(uint nodes, uint perm_size, uint clusters) {
      cluster_permutation = new perm_node(clusters);
      roots.resize(nodes);
      FOR_N(node, nodes) {
          RUN_NUMA_START(node)
              roots[node] = new perm_node(perm_size);
          RUN_NUMA_END
      }
  }

  ~permutation() {
      delete cluster_permutation;
      FOR_N(i, roots.size()) {
          delete roots[i];
      }
  }

  perm_node* get_basic_permutation(uint node) {
      return roots[node];
  }

  perm_node* get_cluster_permutation() {
      return cluster_permutation;
  }
};


#endif //PSGD_BLOK_PERMUTATION_H
