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
#include "spin_barrier.h"


struct sgd_params {
  uint max_epochs;
  fp_type target_accuracy;
  fp_type step_decay;
  fp_type step;
  uint block_size;
};

template<typename T>
class Task {
public:
  sgd_params params;
  T* data_scheme;
  const dataset& train;
  const dataset& validate;
  const uint threads;
  spin_barrier* const barrier;
  std::atomic<uint>* const stop;
  permutation* perm;
  const bool copy;
  uint blocks_per_thread;


  Task(uint nodes,
       const sgd_params* params,
       T* data_scheme,
       const dataset& train,
       const dataset& validate,
       uint threads)
      : params(*params),
        data_scheme(data_scheme->clone()),
        train(train),
        validate(validate),
        threads(threads),
        barrier(new spin_barrier(threads)),
        stop(new std::atomic<uint>(0)),
        copy(false) {
      const uint data_size = train.get_data(0).get_size();
      blocks_per_thread = std::max(1u, data_size / (params->block_size * threads));
      const uint total_blocks = threads * blocks_per_thread;
      const uint actual_block_size = data_size / total_blocks;
      perm = new permutation(nodes,actual_block_size, data_scheme->number_of_copies());
  }

  Task(const Task& other)
      : params(other.params),
        data_scheme(other.data_scheme->clone()),
        train(other.train),
        validate(other.validate),
        threads(other.threads),
        barrier(other.barrier),
        stop(other.stop),
        perm(other.perm),
        copy(true),
        blocks_per_thread(other.blocks_per_thread) {}

  ~Task() {
      delete data_scheme;
      if (copy) return;
      delete barrier;
      delete stop;
      delete perm;
  }
};

template<typename T>
void* thread_task(void* args, const uint thread_id) {
    Task<T> task = *reinterpret_cast<Task<T>*>(args);

    const uint node = config.get_node_for_thread(thread_id);
    const dataset_local& train = task.train.get_data(node);
    const dataset_local& validate = task.validate.get_data(node);
    T* const scheme = task.data_scheme;
    vector<fp_type>* w = scheme->get_model_vector(thread_id);
    MODEL_PARAMS* const model_args = reinterpret_cast<MODEL_PARAMS*>(scheme->get_model_args(thread_id));

    perm_node* cluster_perm = task.perm->get_cluster_permutation();
    perm_node* perm_n = task.perm->get_basic_permutation(node);
    const uint block_size = perm_n->size;
    const uint threads_per_cluster = task.threads / cluster_perm->size;
    const uint blocks_per_thread = task.blocks_per_thread;
    const uint blocks_per_cluster = blocks_per_thread * threads_per_cluster;
    const uint cluster_id = thread_id / threads_per_cluster;
    const uint in_cluster_id = thread_id % threads_per_cluster;

    const uint valid_size = validate.get_size();
    const uint valid_block_size = valid_size / task.threads;
    const uint valid_start = valid_block_size * thread_id;
    const uint valid_end = thread_id + 1 == task.threads ? valid_size : valid_block_size * (thread_id + 1);
    const uint target_correct = task.params.target_accuracy * valid_size;

    vector<uint> blocks_perm;
    blocks_perm.init(blocks_per_thread);
    FOR_N(i, blocks_per_thread) {
        blocks_perm[i] = i;
    }

    const uint n = task.params.max_epochs;
    uint epochs = n;
    FOR_N(e, n) {
        const fp_type step = task.params.step;
        const uint* const perm_array = perm_n->permutation;
        const uint c = cluster_perm->permutation[cluster_id];
        const uint start_block = c * blocks_per_cluster + in_cluster_id * blocks_per_thread;

        FOR_N(block_index, blocks_per_thread) {
            const uint block = blocks_perm[block_index] + start_block;
            const uint start = block_size * block;

            // Update cycle must avoid any unnecessary NUMA communication
            FOR_N(i, block_size) {
                const uint index = perm_array[i] + start;
                const data_point point = train[index];
                MODEL_UPDATE(point, w, step, model_args);
                scheme->post_update(thread_id, step);
            }
        }
        task.params.step *= task.params.step_decay;

        perm_n = perm_n->gen_next();
        cluster_perm = cluster_perm->gen_next();

        task.stop->store(0);
        task.barrier->wait();
        const uint correct = compute_correct(validate, w, valid_start, valid_end);
        task.stop->fetch_add(correct);
        task.barrier->wait();
        if (task.stop->load() >= target_correct) {
            epochs = e + 1;
            break;
        }
        perm_node::shuffle(blocks_perm.data, blocks_per_thread);
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
