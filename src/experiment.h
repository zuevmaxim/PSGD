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
#include <chrono>
#include <sstream>
#include <atomic>


struct sgd_params {
  uint max_epochs;
  fp_type target_accuracy;
  fp_type step_decay;
  fp_type step;
  uint block_size;
};

class Task {
public:
  sgd_params params;
  Model model;
  abstract_data_scheme* data_scheme;
  const dataset& train;
  const dataset& validate;
  const uint threads;
  uint* const validator;
  std::atomic<bool>* stop;
  permutation* const perm;
  const bool copy;


  Task(sgd_params* params,
       Model model,
       abstract_data_scheme* data_scheme,
       const dataset& train,
       const dataset& validate,
       uint threads)
      : params(*params),
        model(model),
        data_scheme(data_scheme->clone()),
        train(train),
        validate(validate),
        threads(threads),
        validator(new uint(0)),
        stop(new std::atomic<bool>(false)),
        perm(new permutation(threads)),
        copy(false) {}

  Task(const Task& other)
      : params(other.params),
        model(other.model),
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

void* thread_task(void* args, const uint thread_id) {
    Task task = *reinterpret_cast<Task*>(args);

    const uint node = config.get_node_for_thread(thread_id);
    const dataset_local& data = task.train.get_data(node);
    const uint block_size = data.get_size() / task.threads;
    abstract_data_scheme* const scheme = task.data_scheme;
    vector<fp_type>& w = scheme->get_model_vector(thread_id);
    void* const model_args = scheme->get_model_args(thread_id);
    const ModelUpdate& update = task.model.update;

    const uint n = task.params.max_epochs;
    uint epochs = n;
//    std::vector<int> p(data.size);
//    FOR_N(i, data.size) {
//        p[i] = i;
//    }
    FOR_N(e, n) {
//        std::random_shuffle(p.begin(), p.end());
        const fp_type step = task.params.step;

        const uint block = task.perm->get_permutation(thread_id);
        const uint start = block_size * block;
        const uint end = std::min(data.get_size(), block_size * (block + 1));

        // Update cycle must avoid any unnecessary NUMA communication
        for (uint i = start; i < end; ++i) {
            const data_point& point = data[i];
            update(point, w, step, model_args);
            scheme->post_update(thread_id);
        }
        task.params.step *= task.params.step_decay;


        if (task.stop->load()) {
            epochs = e + 1;
            break;
        }
        if (thread_id == *task.validator) {
            const fp_type accuracy = compute_accuracy(task.model.predict, task.validate, w, node);
            if (accuracy > task.params.target_accuracy) {
                task.stop->store(true);
                epochs = e + 1;
                break;
            }
            task.perm->permute();
            uint next_id = thread_id + 1;
            if (next_id == task.threads) next_id = 0;
            *task.validator = next_id;
        }
    }
    return new uint(epochs);
}

bool run_experiment(
    const dataset& train,
    const dataset& validate,
    thread_pool& tp,
    sgd_params* params,
    Model model,
    abstract_data_scheme* data_scheme,
    std::vector<void*>& results
) {
    Task task(params, model, data_scheme, train, validate, tp.get_size());

    results = tp.execute(thread_task, &task);

    return *task.stop;
}

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration <fp_type> fp_sec;

struct experiment_configuration {
  const dataset& train_dataset;
  const dataset& test_dataset;
  const dataset& validate_dataset;

  std::string algorithm;
  unsigned test_repeats = 1;
  unsigned threads = 1, cluster_size = 1, max_epochs = 100, update_delay = 64;
  unsigned block_size = 1024;
  fp_type target_accuracy = 1, step_size = 0.5, step_decay = 0.8, mu = 1, tolerance = 0.01;

  experiment_configuration(const dataset& train_dataset,
                           const dataset& test_dataset,
                           const dataset& validate_dataset) : train_dataset(train_dataset), test_dataset(test_dataset), validate_dataset(validate_dataset) {}

  bool from_string(const std::string& command) {
      std::stringstream ss(command);

      // HogWild 10 2 1 100 1 0.97713 0.5 0.8 1.0 0.01
      ss >> algorithm >> test_repeats >> threads >> cluster_size >> max_epochs >> update_delay >> target_accuracy
         >> step_size >> step_decay >> mu >> tolerance;
      return !ss.fail();
  }

  void run_experiments() {
      thread_pool tp(threads);

      const uint features = train_dataset.get_features();
      SVMParams svm_params(mu, &train_dataset);

      Model model;
      model.predict = &svm::predict;
      model.update = &svm::update;

      sgd_params params;
      params.max_epochs = max_epochs;
      params.target_accuracy = target_accuracy;
      params.step_decay = step_decay;
      params.step = step_size;
      params.block_size = block_size;

      FOR_N(run, test_repeats) {
          std::unique_ptr <abstract_data_scheme> scheme;
          if (algorithm == "HogWild") {
              scheme.reset(new hogwild_data_scheme(features, &svm_params));
          } else if (algorithm == "HogWild++") {
              hogwild_XX_params params(threads, cluster_size, tolerance, update_delay);
              scheme.reset(new hogwild_XX_data_scheme<SVMParams>(features, &svm_params, params));
          } else {
              exit(1);
          }

          std::vector<void*> results;
          auto start = Time::now();
          bool success = run_experiment(train_dataset, validate_dataset, tp, &params, model, scheme.get(), results);
          auto end = Time::now();

          uint epochs = 0;
          FOR_N(i, threads) {
              auto e = reinterpret_cast<uint*>(results[i]);
              epochs += *e;
              delete e;
          }
          fp_type average_epochs = epochs / threads;

          fp_type train_accuracy = compute_accuracy(model.predict, train_dataset, scheme->get_model_vector(0));
          fp_type validate_accuracy = compute_accuracy(model.predict, validate_dataset, scheme->get_model_vector(0));
          fp_type test_accuracy = compute_accuracy(model.predict, test_dataset, scheme->get_model_vector(0));
          fp_sec time = end - start;

          std::cout << std::fixed << std::setprecision(5) << std::setfill(' ')
                    << "Experiment " << run + 1 << "/" << test_repeats
                    << " completed with " << (success ? "SUCCESS" : "FAIL")
                    << " with train accuracy " << train_accuracy
                    << " with validate accuracy " << validate_accuracy
                    << " with test accuracy " << test_accuracy
                    << " in " << time.count() << " seconds"
                    << " took " << average_epochs << " average epochs"
                    << std::endl;

          if (!success) {
              std::cerr << "Break experiments!" << std::endl;
              break;
          }
      }
  }
};


#endif //PSGD_EXPERIMENT_H
