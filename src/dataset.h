//
// Created by Maksim.Zuev on 29.05.2022.
//

#ifndef PSGD_DATASET_H
#define PSGD_DATASET_H

#include "numa.h"
#include "dataset_local.h"

class dataset {
private:
  vector<dataset_local*> datasets;

public:
  dataset(uint nodes, const std::string& name) {
      datasets.init(nodes);
      FOR_N(i, nodes) {
          RUN_NUMA_START(i)
              if (i == 0) {
                  datasets[0] = new dataset_local(name);
              } else {
                  datasets[i] = new dataset_local(*datasets[0]);
              }
          RUN_NUMA_END
      }
  }

  dataset(const dataset& other, const std::vector<uint>& inverse_permutation) {
      datasets.init(other.datasets.size);
      FOR_N(i, datasets.size) {
          RUN_NUMA_START(i)
              if (i == 0) {
                  datasets[0] = new dataset_local(*other.datasets[0], inverse_permutation);
              } else {
                  datasets[i] = new dataset_local(*datasets[0]);
              }
          RUN_NUMA_END
      }
  }

  ~dataset() {
      FOR_N(i, datasets.size) {
          delete datasets[i];
      }
  }

  inline const dataset_local& get_data(uint node) const {
      return *datasets[node];
  }

  inline uint get_features() const {
      return datasets[0]->get_features();
  }
};


#endif //PSGD_DATASET_H
