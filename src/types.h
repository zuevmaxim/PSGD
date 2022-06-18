//
// Created by Maksim.Zuev on 29.05.2022.
//

#ifndef PSGD_TYPES_H
#define PSGD_TYPES_H

#define FOR_N(i, n) for (uint i = 0; i < n; ++i)
#define RUN_NUMA_START(node) { numa_run_on_node(node); numa_set_preferred(node);
#define RUN_NUMA_END numa_run_on_node(-1); numa_set_preferred(-1); }

typedef double fp_type;
typedef unsigned int uint;

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

#endif //PSGD_TYPES_H
