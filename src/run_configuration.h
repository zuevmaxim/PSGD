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
typedef std::chrono::duration<fp_type> fp_sec;

struct experiment_configuration {
  static bool verbose;

  const dataset& train_dataset;
  const dataset& test_dataset;
  const dataset& validate_dataset;
  std::ostream& output;

  std::string algorithm;
  unsigned test_repeats = 1;
  unsigned block_size = 512;
  unsigned threads = 1, cluster_size = 1, max_epochs = 100, update_delay = 64;
  fp_type target_score = 1, step_size = 0.5, step_decay = 0.8;
  fp_type mu = 1, tolerance = 0.01;

  experiment_configuration(const dataset& train_dataset,
                           const dataset& test_dataset,
                           const dataset& validate_dataset,
                           std::ostream& output) : train_dataset(train_dataset), test_dataset(test_dataset), validate_dataset(validate_dataset), output(output) {}

  bool from_string(const std::string& command) {
      std::stringstream ss(command);
      ss >> algorithm >> test_repeats >> threads >> cluster_size >> max_epochs >> update_delay >> target_score
         >> step_size >> step_decay >> block_size;
      return !ss.fail();
  }

  template<typename T>
  T* create_scheme(uint, void*) {
      throw std::runtime_error("This function must not be called!");
  }

  template<typename T>
  void run_experiments_internal() {
      if (verbose) {
          std::cout << "Start experiments (" << test_repeats << ") with " << algorithm << " algorithm"
                    << " threads=" << threads
                    << (algorithm == "HogWild" ? "" : " cluster_size=" + std::to_string(cluster_size))
                    << " target_score=" << target_score
                    << " step_size=" << step_size
                    << " step_decay=" << step_decay
                    << (algorithm == "HogWild" ? "" : " update_delay=" + std::to_string(update_delay))
                    << " block_size=" << block_size
                    << std::endl;
      }

      thread_pool tp(threads);

      const uint features = train_dataset.get_features();
      SVMParams svm_params(mu, &train_dataset);

      sgd_params params{};
      params.max_epochs = max_epochs;
      params.target_score = target_score;
      params.step_decay = step_decay;
      params.step = step_size;
      params.block_size = block_size;

      fp_type total_time = 0;
      fp_type total_epochs = 0;
      fp_type total_epoch_time = 0;
      fp_type total_tests = 0;

      FOR_N(run, test_repeats) {
          std::unique_ptr<T> scheme(create_scheme<T>(features, &svm_params));

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

          fp_type train_score = compute_metric(train_dataset.get_data(0), scheme->get_model_vector(0)).to_score();
          fp_type validate_score = compute_metric(validate_dataset.get_data(0), scheme->get_model_vector(0)).to_score();
          fp_type test_score = compute_metric(test_dataset.get_data(0), scheme->get_model_vector(0)).to_score();
          fp_type time = static_cast<fp_sec>(end - start).count();
          fp_type epoch_time = time / average_epochs;

          if (verbose) {
              std::cout << std::fixed << std::setprecision(5) << std::setfill(' ')
                        << "Experiment " << run + 1 << "/" << test_repeats
                        << " completed with " << (success ? "SUCCESS" : "FAIL")
                        << " time=" << time
                        << " epochs=" << average_epochs
                        << " per_epoch=" << epoch_time
                        << std::endl;
          }

          output
              << algorithm << ',' << threads << ',' << cluster_size << ',' << (success ? 1 : 0) << ','
              << time << ',' << train_score << ',' << validate_score << ',' << test_score << ','
              << average_epochs << ',' << epoch_time << ','
              << step_size << ',' << step_decay << ',' << update_delay << ','
              << target_score << ',' << block_size
              << std::endl;

          if (!success) {
              if (!verbose) std::cout << '!' << std::flush;
              continue;
          }
          if (!verbose) std::cout << '.' << std::flush;

          total_time += time;
          total_epochs += average_epochs;
          total_epoch_time += epoch_time;
          total_tests++;
      }

      total_time /= total_tests;
      total_epochs /= total_tests;
      total_epoch_time /= total_tests;
      total_tests /= test_repeats;


      std::cout << (verbose ? "" : "\n")
                << "Average results:"
                << " algorithm=" << algorithm
                << " threads=" << threads
                << " block_size=" << block_size
                << " step_decay=" << step_decay
                << " update_delay=" << update_delay
                << " convergence=" << total_tests
                << " time=" << total_time
                << " epochs=" << total_epochs
                << " epoch_time=" << total_epoch_time
                << std::endl;
  }

  void run_experiments() {
      if (algorithm == "HogWild") {
          run_experiments_internal<hogwild_data_scheme>();
      } else if (algorithm == "HogWild++") {
          run_experiments_internal<hogwild_XX_data_scheme<SVMParams>>();
      } else if (algorithm == "MyWild") {
          run_experiments_internal<mywild_data_scheme<SVMParams>>();
      } else {
          std::cerr << "Unexpected algorithm: " << algorithm << std::endl;
      }
  }
};

bool experiment_configuration::verbose = false;

template<>
hogwild_data_scheme* experiment_configuration::create_scheme(uint features, void* model_args) {
    return new hogwild_data_scheme(features, model_args);
}

template<>
hogwild_XX_data_scheme<SVMParams>* experiment_configuration::create_scheme(uint features, void* model_args) {
    auto svm_params = reinterpret_cast<SVMParams*>(model_args);
    hogwild_XX_params params(threads, cluster_size, tolerance, update_delay);
    return new hogwild_XX_data_scheme<SVMParams>(features, svm_params, params);
}

template<>
mywild_data_scheme<SVMParams>* experiment_configuration::create_scheme(uint features, void* model_args) {
    auto svm_params = reinterpret_cast<SVMParams*>(model_args);
    mywild_params params(threads, cluster_size, update_delay);
    return new mywild_data_scheme<SVMParams>(features, svm_params, params);
}

#endif //PSGD_RUN_CONFIGURATION_H
