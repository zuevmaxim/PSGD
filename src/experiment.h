//
// Created by Maksim.Zuev on 29.05.2022.
//

#ifndef PSGD_EXPERIMENT_H
#define PSGD_EXPERIMENT_H

#include "model.h"
#include "dataset.h"
#include "thread_pool.h"
#include "block_permutation.h"
#include "cpu_config.h"
#include <atomic>
#include "spin_barrier.h"


struct sgd_params {
  uint max_epochs;
  fp_type target_score;
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
  metric_summary* const metric;
  permutation* const perm;
  const bool copy;
  const uint blocks_per_thread;


  Task(uint nodes,
       const sgd_params* params,
       T* data_scheme,
       const dataset& train,
       const dataset& validate,
       uint threads)
      : params(*params),
        data_scheme(data_scheme),
        train(train),
        validate(validate),
        threads(threads),
        barrier(new spin_barrier(threads)),
        metric(new metric_summary()),
        perm(new permutation(nodes)),
        copy(false),
        blocks_per_thread(std::max(1u, train.get_data(0).get_size() / (params->block_size * threads))) {}

  Task(const Task& other)
      : params(other.params),
        data_scheme(other.data_scheme->clone()),
        train(other.train),
        validate(other.validate),
        threads(other.threads),
        barrier(other.barrier),
        metric(other.metric),
        perm(other.perm),
        copy(true),
        blocks_per_thread(other.blocks_per_thread) {}

  ~Task() {
      if (copy) {
          delete data_scheme;
          return;
      }
      delete barrier;
      delete metric;
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
    vector<fp_type>* const w = scheme->get_model_vector(thread_id);
    auto* const model_args = reinterpret_cast<MODEL_PARAMS*>(scheme->get_model_args(thread_id));

    perm_node* cluster_perm = task.perm->get_cluster_permutation();
    const uint threads_per_cluster = task.threads / cluster_perm->size;
    const uint blocks_per_thread = task.blocks_per_thread;
    const uint total_blocks = blocks_per_thread * task.threads;
    const uint train_size = train.get_size();
    const uint block_size = train_size / total_blocks;
    const uint blocks_per_cluster = blocks_per_thread * threads_per_cluster;
    const uint cluster_id = thread_id / threads_per_cluster;
    const uint in_cluster_id = thread_id % threads_per_cluster;

    const uint valid_size = validate.get_size();
    const uint valid_block_size = valid_size / task.threads;
    const uint valid_start = valid_block_size * thread_id;
    const uint valid_end = thread_id + 1 == task.threads ? valid_size : valid_block_size * (thread_id + 1);
    const fp_type target_score = task.params.target_score;

    vector<uint> blocks_perm;
    blocks_perm.init(blocks_per_thread);
    FOR_N(i, blocks_per_thread) {
        blocks_perm[i] = i;
    }

    const uint n = task.params.max_epochs;
    FOR_N(e, n) {
        const fp_type step = task.params.step;
        const uint c = cluster_perm->permutation[cluster_id];
        const uint start_block = c * blocks_per_cluster + in_cluster_id * blocks_per_thread;

        FOR_N(block_index, blocks_per_thread) {
            const uint block = blocks_perm[block_index] + start_block;
            const uint start = block_size * block;
            const uint end = unlikely(block + 1 == total_blocks) ? train_size : start + block_size;

            // Update cycle must avoid any unnecessary NUMA communication
            for (uint i = start; i < end; ++i) {
                const data_point point = train[i];
                MODEL_UPDATE(point, w, step, model_args);
                scheme->post_update(thread_id, step);
            }
        }
        task.params.step *= task.params.step_decay;
        cluster_perm = cluster_perm->gen_next();

        task.metric->zero();
        task.barrier->wait();
        const auto summary = compute_metric(validate, w, valid_start, valid_end);
        task.metric->plus(summary);
        task.barrier->wait();
        const fp_type current_score = task.metric->to_score();
        if (unlikely(current_score >= target_score)) {
            return new uint(e + 1);
        }
        perm_node::shuffle(blocks_perm.data, blocks_per_thread);
    }
    task.metric->zero();
    return new uint(n);
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

    return task.metric->total() > 0;
}


#endif //PSGD_EXPERIMENT_H
