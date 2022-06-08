//
// Created by Maksim.Zuev on 08.06.2022.
//

#ifndef PSGD_RUN_CONFIGURATION_H
#define PSGD_RUN_CONFIGURATION_H

#include <iomanip>
#include <memory>
#include <chrono>
#include <sstream>
#include "experiment.h"


typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration <fp_type> fp_sec;

struct experiment_configuration {
  const dataset& train_dataset;
  const dataset& test_dataset;
  const dataset& validate_dataset;

  std::string algorithm;
  unsigned test_repeats = 1;
  unsigned block_size = 512;
  unsigned threads = 1, cluster_size = 1, max_epochs = 100, update_delay = 64;
  fp_type target_accuracy = 1, step_size = 0.5, step_decay = 0.8;
  fp_type mu = 1, tolerance = 0.01;

  experiment_configuration(const dataset& train_dataset,
                           const dataset& test_dataset,
                           const dataset& validate_dataset) : train_dataset(train_dataset), test_dataset(test_dataset), validate_dataset(validate_dataset) {}

  bool from_string(const std::string& command) {
      std::stringstream ss(command);

      // HogWild 10 1 1 100 128 0.97713 0.5 0.8 512
      ss >> algorithm >> test_repeats >> threads >> cluster_size >> max_epochs >> update_delay >> target_accuracy
         >> step_size >> step_decay >> block_size;
      return !ss.fail();
  }

  template<typename T>
  T* create_scheme(uint features, void* model_args) {
  }

  template<>
  hogwild_data_scheme* create_scheme(uint features, void* model_args) {
      return new hogwild_data_scheme(features, model_args);
  }

  template<>
  hogwild_XX_data_scheme<SVMParams>* create_scheme(uint features, void* model_args) {
      auto svm_params = reinterpret_cast<SVMParams*>(model_args);
      hogwild_XX_params params(threads, cluster_size, tolerance, update_delay);
      return new hogwild_XX_data_scheme<SVMParams>(features, svm_params, params);
  }

  template<>
  mywild_data_scheme<SVMParams>* create_scheme(uint features, void* model_args) {
      auto svm_params = reinterpret_cast<SVMParams*>(model_args);
      mywild_params params(threads, cluster_size, tolerance, update_delay);
      return new mywild_data_scheme<SVMParams>(features, svm_params, params);
  }

  template<typename T>
  void run_experiments_internal(std::ostream& out) {
      std::cout << "Start experiments (" << test_repeats << ") with " << algorithm << " algorithm"
                << " threads=" << threads
                << (algorithm == "HogWild" ? "" : " cluster_size=" + std::to_string(cluster_size))
                << " target_accuracy=" << target_accuracy
                << " step_size=" << step_size
                << " step_decay=" << step_decay
                << (algorithm == "HogWild" ? "" : " update_delay=" + std::to_string(update_delay))
                << std::endl;

      thread_pool tp(threads);

      const uint features = train_dataset.get_features();
      SVMParams svm_params(mu, &train_dataset);

      sgd_params params;
      params.max_epochs = max_epochs;
      params.target_accuracy = target_accuracy;
      params.step_decay = step_decay;
      params.step = step_size;
      params.block_size = block_size;

      FOR_N(run, test_repeats) {
          std::unique_ptr <T> scheme(create_scheme<T>(features, &params));

          std::vector<void*> results;
          auto start = Time::now();
          bool success = run_experiment<T>(train_dataset, validate_dataset, tp, &params, scheme.get(), results);
          auto end = Time::now();

          uint epochs = 0;
          FOR_N(i, threads) {
              auto e = reinterpret_cast<uint*>(results[i]);
              epochs += *e;
              delete e;
          }
          fp_type average_epochs = static_cast<fp_type>(epochs) / threads;

          fp_type train_accuracy = compute_accuracy(train_dataset, scheme->get_model_vector(0));
          fp_type validate_accuracy = compute_accuracy(validate_dataset, scheme->get_model_vector(0));
          fp_type test_accuracy = compute_accuracy(test_dataset, scheme->get_model_vector(0));
          fp_sec time = end - start;

          std::cout << std::fixed << std::setprecision(5) << std::setfill(' ')
                    << "Experiment " << run + 1 << "/" << test_repeats
                    << " completed with " << (success ? "SUCCESS" : "FAIL")
                    << " train=" << train_accuracy
                    << " validate=" << validate_accuracy
                    << " test=" << test_accuracy
                    << " time=" << time.count()
                    << " epochs=" << average_epochs
                    << " per_epoch=" << time.count() / average_epochs
                    << std::endl;

          out << algorithm << ',' << threads << ',' << cluster_size << ',' << (success ? 1 : 0) << ','
              << time.count() << ',' << train_accuracy << ',' << validate_accuracy << ',' << test_accuracy << ','
              << average_epochs << ',' << time.count() / average_epochs << ','
              << step_size << ',' << step_decay << ',' << update_delay << ','
              << target_accuracy
              << std::endl;

          if (!success) {
              std::cerr << "Break experiments!" << std::endl;
              break;
          }
      }
  }

  void run_experiments(std::ostream& output_file) {
      if (algorithm == "HogWild") {
          run_experiments_internal<hogwild_data_scheme>(output_file);
      } else if (algorithm == "HogWild++") {
          run_experiments_internal<hogwild_XX_data_scheme<SVMParams>>(output_file);
      } else if (algorithm == "MyWild") {
          run_experiments_internal<mywild_data_scheme<SVMParams>>(output_file);
      } else {
          std::cerr << "Unexpected algorithm: " << algorithm << std::endl;
      }
  }
};

#endif //PSGD_RUN_CONFIGURATION_H
