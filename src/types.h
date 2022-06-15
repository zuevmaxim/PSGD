//
// Created by Maksim.Zuev on 29.05.2022.
//

#ifndef PSGD_TYPES_H
#define PSGD_TYPES_H

#define FOR_N(i, n) for (uint i = 0; i < n; ++i)
#define FOR_N_REV(i, n) for (uint i = n; i-- > 0;)
#define RUN_NUMA_START(node) { numa_run_on_node(node); numa_set_preferred(node);
#define RUN_NUMA_END numa_run_on_node(-1); numa_set_preferred(-1); }

typedef double fp_type;
typedef unsigned int uint;

#endif //PSGD_TYPES_H
