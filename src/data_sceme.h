//
// Created by Maksim.Zuev on 29.05.2022.
//

#ifndef PSGD_DATA_SCEME_H
#define PSGD_DATA_SCEME_H

#include "vectors.h"
#include "cpu_config.h"
#include <cmath>

class abstract_data_scheme {
public:

  virtual void* get_model_args(uint thread_id) = 0;

  virtual vector<fp_type>& get_model_vector(uint thread_id) = 0;

  virtual void post_update(uint thread_id, fp_type step) = 0;

  virtual abstract_data_scheme* clone() = 0;

  virtual ~abstract_data_scheme() {
  }
};

class hogwild_data_scheme final : public abstract_data_scheme {
private:
  vector<fp_type>* w;
  void* const args;
  const bool copy;

  hogwild_data_scheme(const hogwild_data_scheme& other) : w(other.w), args(other.args), copy(true) {}

public:
  hogwild_data_scheme(uint size, void* args) : w(new vector<fp_type>), args(args), copy(false) {
      w->init(size);
      FOR_N(i, size) {
          (*w)[i] = 0;
      }
  }

  ~hogwild_data_scheme() {
      if (copy) return;
      delete w;
  }


  void* get_model_args(uint thread_id) override {
      return args;
  }

  vector<fp_type>& get_model_vector(uint thread_id) override {
      return *w;
  }

  void post_update(uint thread_id, fp_type step) override {
  }

  hogwild_data_scheme* clone() override {
      return new hogwild_data_scheme(*this);
  }
};

//template<typename ModelParams>
//class abstract_copy_data_scheme : public abstract_data_scheme {
//protected:
//  const bool copy;
//  vector<vector<fp_type>*> w;
//  vector<ModelParams*> params;
//  vector<uint> thread_to_model;
//
//public:
//  abstract_copy_data_scheme(uint size, ModelParams* args, uint threads, uint cluster_size)
//      : copy(false) {
//      const uint phy_threads = std::min(threads, config.get_phy_cpus());
//      if ((phy_threads % cluster_size) != 0) throw std::runtime_error("Fractional clusters are not supported.");
//      const uint cluster_count = phy_threads / cluster_size;
//
//      w.init(cluster_count);
//      params.init(cluster_count);
//      FOR_N(cluster, cluster_count) {
//          uint basic_thread_id = cluster * cluster_size;
//          uint node = config.get_node_for_thread(basic_thread_id);
//          RUN_NUMA_START(node)
//
//              vector<fp_type>& model = *(new vector<fp_type>);
//              w[cluster] = &model;
//              model.init(size);
//              FOR_N(i, size) {
//                  model[i] = 0.0;
//              }
//
//              params[cluster] = new ModelParams(*args);
//          RUN_NUMA_END
//      }
//
//      thread_to_model.init(threads);
//      FOR_N(thread_id, threads) {
//          uint model = (thread_id % phy_threads) / cluster_size;
//          thread_to_model[thread_id] = model;
//      }
//  }
//
//  abstract_copy_data_scheme(const abstract_copy_data_scheme<ModelParams>& other)
//      : copy(true),
//        w(other.w),
//        params(other.params),
//        thread_to_model(other.thread_to_model) {}
//
//  ~abstract_copy_data_scheme() override {
//      if (copy) return;
//      FOR_N(cluster, params.size) {
//          delete params[cluster];
//          delete w[cluster];
//      }
//  }
//
//  void* get_model_args(uint thread_id) override {
//      return params[thread_to_model[thread_id]];
//  }
//
//  vector<fp_type>& get_model_vector(uint thread_id) override {
//      return *w[thread_to_model[thread_id]];
//  }
//};

struct hogwild_XX_params {
  const uint threads;
  const uint cluster_size;
  const fp_type tolerance;
  const uint phy_threads;
  const uint cluster_count;
  const uint delay;

  fp_type lambda;
  fp_type beta;

  hogwild_XX_params(uint threads, uint cluster_size, fp_type tolerance, uint delay)
      : threads(threads),
        cluster_size(cluster_size),
        tolerance(tolerance),
        phy_threads(std::min(threads, config.get_phy_cpus())),
        cluster_count(phy_threads / cluster_size),
        delay(delay * phy_threads) {
      if ((phy_threads % cluster_size) != 0) throw std::runtime_error("Fractional clusters are not supported.");
      beta = SolveBeta(cluster_count);
      lambda = 1 - pow(beta, cluster_count - 1);
  }

private:
  static fp_type SolveBeta(int n) {
      fp_type start = 0.6;
      fp_type end = 1.0;
      fp_type mid = 0.5;
      fp_type err = 0;
      if (n >= 2) {
          do {
              mid = (start + end) / 2;
              err = pow(mid, n) + mid - 1;
              if (err > 0) {
                  end = mid;
              } else {
                  start = mid;
              }
          } while (std::fabs(err) > 0.001);
      }
      if (!n)
          mid = .0;
      return mid;
  }
};

template<typename ModelParams>
class hogwild_XX_data_scheme final : public abstract_data_scheme {
private:
  const bool copy;
  vector<vector<fp_type>*> old_w;
  vector<vector<fp_type>*> w;
  vector<ModelParams*> model_params;
  vector<uint> thread_to_model;
  vector<int> next;
  uint* sync_thread;
  hogwild_XX_params params;
  int delay;

public:
  hogwild_XX_data_scheme(uint size, ModelParams* args, const hogwild_XX_params& _params)
      : copy(false), sync_thread(new uint(0)), params(_params) {
      delay = params.delay;

      const uint cluster_count = params.cluster_count;
      w.init(cluster_count);
      old_w.init(cluster_count);
      model_params.init(cluster_count);
      FOR_N(cluster, cluster_count) {
          uint basic_thread_id = cluster * params.cluster_size;
          uint node = config.get_node_for_thread(basic_thread_id);
          RUN_NUMA_START(node)

              w[cluster] = new vector<fp_type>;
              auto& model = *w[cluster];
              model.init(size);
              FOR_N(i, size) {
                  model[i] = 0.0;
              }

              old_w[cluster] = new vector<fp_type>;
              auto& old_model = *old_w[cluster];
              old_model.init(size);
              FOR_N(i, size) {
                  old_model[i] = 0.0;
              }

              model_params[cluster] = new ModelParams(*args);
          RUN_NUMA_END
      }

      thread_to_model.init(params.threads);
      FOR_N(thread_id, params.threads) {
          uint model = (thread_id % params.phy_threads) / params.cluster_size;
          thread_to_model[thread_id] = model;
      }

      next.init(size);
      FOR_N(thread_id, params.threads) {
          next[thread_id] = params.cluster_count > 1 && thread_id < params.phy_threads
                            ? (thread_id + params.cluster_size) % params.phy_threads
                            : -1;
      }
  }

  hogwild_XX_data_scheme(const hogwild_XX_data_scheme<ModelParams>& other)
      : copy(true),
        old_w(other.old_w),
        w(other.w),
        model_params(other.model_params),
        thread_to_model(other.thread_to_model),
        next(other.next),
        sync_thread(other.sync_thread),
        params(other.params),
        delay(other.params.delay) {}

  ~hogwild_XX_data_scheme() override {
      if (copy) return;
      delete sync_thread;
      FOR_N(cluster, model_params.size) {
          delete model_params[cluster];
          delete w[cluster];
          delete old_w[cluster];
      }
  }

  void* get_model_args(uint thread_id) override {
      return model_params[thread_to_model[thread_id]];
  }

  vector<fp_type>& get_model_vector(uint thread_id) override {
      return *w[thread_to_model[thread_id]];
  }

  hogwild_XX_data_scheme<ModelParams>* clone() override {
      return new hogwild_XX_data_scheme(*this);
  }

  void post_update(uint thread_id, const fp_type step) override {
      if (--delay > 0) return;

      const int next_id = next[thread_id];
      if (next_id < 0) return;
      if (thread_id != *sync_thread) return;

      const uint model = thread_to_model[thread_id];
      const uint next_model = thread_to_model[next_id];
      if (model == next_model) {
          throw std::runtime_error("Next model equals current model.");
      }

      const uint size = old_w[model]->size;
      fp_type* const old_w = this->old_w[model]->data;
      fp_type* const cur_w = w[model]->data;
      fp_type* const next_w = w[next_model]->data;

      const fp_type beta = params.beta;
      const fp_type lambda = params.lambda;
      const fp_type tolerance = params.tolerance;

      FOR_N(i, size) {
          const fp_type wi = cur_w[i];
          const fp_type delta = (wi - old_w[i]) * step;
          const fp_type next = next_w[i];
          if (std::fabs(delta) > tolerance) {
              const fp_type new_wi = next * lambda + wi * (1 - lambda) + (beta + lambda - 1) * delta;
              next_w[i] = next + beta * delta;
              cur_w[i] = new_wi;
              old_w[i] = new_wi;
          } else {
              const fp_type new_wi = next * lambda + wi * (1 - lambda) + lambda * delta;
              cur_w[i] = new_wi;
              old_w[i] = new_wi - delta;
          }
      }

      delay = params.delay;
      *sync_thread = next_id;
  }
};


#endif //PSGD_DATA_SCEME_H
