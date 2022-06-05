//
// Created by Maksim.Zuev on 29.05.2022.
//

#ifndef PSGD_CPU_CONFIG_H
#define PSGD_CPU_CONFIG_H

#include "numa.h"
#include "types.h"
#include <vector>
#include <string>
#include <sstream>
#include <fstream>


class cpu_config {
private:
  unsigned cpus;
  unsigned nodes;
  unsigned phy_cpus;
  std::vector <std::vector<std::vector < int>> >
  cpu_ids;
  std::vector<int> thread_core_mapping;
  std::vector<int> thread_node_mapping;


public:


  cpu_config() {
      get_topology();
  }

  void bind_to_cpu(unsigned thread_id) {
      struct bitmask* cpu_mask;
      int cpu = thread_core_mapping[thread_id];
      cpu_mask = numa_allocate_cpumask();
      numa_bitmask_setbit(cpu_mask, cpu);
      numa_sched_setaffinity(0, cpu_mask);
      numa_free_cpumask(cpu_mask);
  }

  unsigned get_numa_count() {
      return nodes;
  }

  unsigned get_phy_cpus() {
      return phy_cpus;
  }

  unsigned get_node_for_thread(unsigned thread_id) {
      return thread_node_mapping[thread_id];
  }

private:
  void get_topology() {
      cpus = numa_num_task_cpus();
      nodes = numa_max_node() + 1;
      phy_cpus = 0;

      cpu_ids.resize(nodes);
      thread_core_mapping.resize(cpus);
      thread_node_mapping.resize(cpus);
      std::vector<int> known_siblings;
      for (unsigned cpu = 0; cpu < cpus; ++cpu) {
          // skip a core if it is a hyper-threaded logical core
          if (std::find(known_siblings.begin(), known_siblings.end(), cpu) != known_siblings.end()) continue;

          int node = numa_node_of_cpu(cpu);
          std::vector<int> phy_core;
          // find out the siblings of this core
          std::stringstream path;
          path << "/sys/devices/system/cpu/cpu" << cpu << "/topology/thread_siblings_list";
          std::ifstream f(path.str().c_str());
          if (f) {
              std::string siblings((std::istreambuf_iterator<char>(f)),
                                   std::istreambuf_iterator<char>());
              f.close();
              std::istringstream ss(siblings);
              std::string core_id;
              while (std::getline(ss, core_id, ',')) {
                  known_siblings.push_back(atoi(core_id.c_str()));
                  phy_core.push_back(atoi(core_id.c_str()));
              }
          } else {
              phy_core.push_back(cpu);
          }
          cpu_ids[node].push_back(phy_core);
          phy_cpus++;
      }

      for (int cpu = 0; cpu < cpus; ++cpu) {
          assign_thread_affinity(cpu);
      }

      std::cout << "CPUs: " << cpus << "\n";
      std::cout << "Phy cores: " << phy_cpus << "\n";
      std::cout << "Numa nodes:  " << nodes << "\n";
      FOR_N(node, cpu_ids.size()) {
          std::cout << "Node " << node << ": ";
          FOR_N(i, cpu_ids[node].size()) {
              auto& phy_cpus = cpu_ids[node][i];
              std::cout << "[";
              for (auto cpu: phy_cpus) {
                  std::cout << cpu << " ";
              }
              std::cout << "] ";
          }
          std::cout << "\n";
      }
      std::cout << std::endl;
  }

  void assign_thread_affinity(unsigned thread_id) {
      int hyper_thread_id = thread_id / phy_cpus;
      int core_id = thread_id % phy_cpus;

      int node = -1;
      int i;
      for (i = 0; i <= core_id; i += cpu_ids[++node].size());
      int index = core_id - (i - cpu_ids[node].size());
      assert(hyper_thread_id < cpu_ids[node][index].size());
      int core = cpu_ids[node][index][hyper_thread_id];

      thread_node_mapping[thread_id] = node;
      thread_core_mapping[thread_id] = core;
  }
};

static cpu_config config;

#endif //PSGD_CPU_CONFIG_H
