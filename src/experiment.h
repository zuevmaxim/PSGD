//
// Created by Maksim.Zuev on 29.05.2022.
//

#ifndef PSGD_EXPERIMENT_H
#define PSGD_EXPERIMENT_H

#include "model.h"
#include "dataset.h"
#include "thread_pool.h"
#include "blok_permutation.h"
#include "cpu_config.h"
#include <atomic>


struct sgd_params {
  uint max_epochs;
  fp_type target_accuracy;
  fp_type step_decay;
  fp_type step;
  uint block_size;
};

uint calc_total_blocks(uint data_size, uint threads, uint block_size) {
    const uint a = std::max(1u, data_size / (block_size * threads));
    return a * threads;
}

template<typename T>
class Task {
public:
  sgd_params params;
  T* data_scheme;
  const dataset& train;
  const dataset& validate;
  const uint threads;
  uint* const validator;
  std::atomic<bool>* stop;
  permutation* const perm;
  const bool copy;


  Task(uint nodes,
       sgd_params* params,
       T* data_scheme,
       const dataset& train,
       const dataset& validate,
       uint threads)
      : params(*params),
        data_scheme(data_scheme->clone()),
        train(train),
        validate(validate),
        threads(threads),
        validator(new uint(0)),
        stop(new std::atomic<bool>(false)),
        perm(new permutation(nodes, calc_total_blocks(train.get_data(0).get_size(), threads, params->block_size))),
        copy(false) {}

  Task(const Task& other)
      : params(other.params),
        data_scheme(other.data_scheme->clone()),
        train(other.train),
        validate(other.validate),
        threads(other.threads),
        validator(other.validator),
        stop(other.stop),
        perm(other.perm),
        copy(true) {}

  ~Task() {
      delete data_scheme;
      if (copy) return;
      delete validator;
      delete stop;
      delete perm;
  }
};

template<typename T>
void* thread_task(void* args, const uint thread_id) {
    Task<T> task = *reinterpret_cast<Task<T>*>(args);

    const uint node = config.get_node_for_thread(thread_id);
    const uint phy_threads = std::min(task.threads, config.get_phy_cpus());
    const dataset_local& data = task.train.get_data(node);
    T* const scheme = task.data_scheme;
    vector<fp_type>* w = scheme->get_model_vector(thread_id);
    MODEL_PARAMS* const model_args = reinterpret_cast<MODEL_PARAMS*>(scheme->get_model_args(thread_id));

    perm_node* perm_n = task.perm->get_basic_permutation(node);
    const uint total_blocks = perm_n->size;
    const uint blocks_per_thread = total_blocks / task.threads;
    const uint data_size = data.get_size();
    const uint block_size = data_size / total_blocks;

    const uint start_block = blocks_per_thread * thread_id;
    const uint end_block = blocks_per_thread * (thread_id + 1);

    const uint n = task.params.max_epochs;
    uint epochs = n;
    FOR_N(e, n) {
        const fp_type step = task.params.step;
        const uint* perm_array = perm_n->permutation;

        for (uint block_index = start_block; block_index < end_block; ++block_index) {
            const uint block = perm_array[block_index];
            const uint start = block_size * block;
            const uint end = block + 1 == total_blocks ? data_size : block_size * (block + 1);

            // Update cycle must avoid any unnecessary NUMA communication
            for (uint i = start; i < end; ++i) {
                const data_point& point = data[i];
                MODEL_UPDATE(point, w, step, model_args);
                scheme->post_update(thread_id, step);
            }
        }
        task.params.step *= task.params.step_decay;


        if (task.stop->load()) {
            epochs = e + 1;
            break;
        }
        if (thread_id == *task.validator) {
            const fp_type accuracy = compute_accuracy(task.validate, w, node);
            if (accuracy > task.params.target_accuracy) {
                task.stop->store(true);
                epochs = e + 1;
                break;
            }
            uint next_id = thread_id + 1;
            if (next_id == phy_threads) next_id = 0;
            *task.validator = next_id;
        }
        perm_n = perm_n->gen_next();
    }
    return new uint(epochs);
}

template<typename T>
bool run_experiment(
    const dataset& train,
    const dataset& validate,
    thread_pool& tp,
    sgd_params* params,
    T* data_scheme,
    std::vector<void*>& results
) {
    Task<T> task(tp.get_numa_count(), params, data_scheme, train, validate, tp.get_size());

    results = tp.execute(thread_task<T>, &task);

    return *task.stop;
}


#endif //PSGD_EXPERIMENT_H
