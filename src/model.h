//
// Created by Maksim.Zuev on 29.05.2022.
//

#ifndef PSGD_MODEL_H
#define PSGD_MODEL_H

#include "types.h"
#include "dataset.h"
#include "data_scheme.h"
#include <atomic>


struct SVMParams {
  const fp_type mu;
  const vector<uint> degrees;

  SVMParams(fp_type mu, const dataset* data) : mu(mu), degrees(calc_degrees(data)) {}

private:
  static vector<uint> calc_degrees(const dataset* dataset) {
      const uint features = dataset->get_features();
      vector<uint> degrees;
      degrees.init(features, 0);
      const dataset_local& points = dataset->get_data(0);
      const uint size = points.get_size();
      FOR_N(i, size) {
          const data_point point = points[i];
          FOR_N(j, point.size) {
              degrees[point.indices[j]]++;
          }
      }
      return degrees;
  }
};

namespace vectors {
  inline fp_type dot(const fp_type* const __restrict__ a_data,
                     const data_point& point) {
      fp_type result = 0;
      const uint size = point.size;
      const uint* const __restrict__ indices = point.indices;
      const fp_type* const __restrict__ b_data = point.data;
      FAST_FOR(i, size) {
          const uint index = indices[i];
          const fp_type b_val = b_data[i];
          const fp_type a_val = a_data[index];
          result += a_val * b_val;
      }
      return result;
  }

  inline void scale_and_add(fp_type* const __restrict__ a_data,
                            const data_point& point,
                            const fp_type s) {
      const uint size = point.size;
      const uint* const __restrict__ indices = point.indices;
      const fp_type* const __restrict__ b_data = point.data;
      FAST_FOR(i, size) {
          const uint index = indices[i];
          const fp_type b_val = b_data[i];
          a_data[index] += s * b_val;
      }
  }
}

namespace svm {
  static inline bool check(const vector<fp_type>* w, const data_point& point) {
      const fp_type dot = vectors::dot(w->data, point);
      return std::max(dot * point.label, 0.0) != 0;
  }

  static inline void update(const data_point& point, vector<fp_type>* w, const fp_type step, const SVMParams* args) {
      fp_type* const __restrict__ vals = w->data;
      const fp_type wxy = vectors::dot(vals, point) * point.label;

      if (wxy < 1) { // hinge is active.
          const fp_type e = step * point.label;
          vectors::scale_and_add(vals, point, e);
      }

      const uint* const __restrict__ degrees = args->degrees.data;
      const uint* const __restrict__ indices = point.indices;
      const fp_type scalar = step * args->mu;
      const uint size = point.size;
      FAST_FOR(i, size) {
          const uint j = indices[i];
          const uint deg = degrees[j];
          vals[j] *= 1 - scalar / deg;
      }
  }
}

#define MODEL_UPDATE svm::update
#define MODEL_CHECK svm::check
#define MODEL_PARAMS SVMParams

struct metric_summary {
  std::atomic<uint> true_positive;
  std::atomic<uint> true_negative;
  std::atomic<uint> false_positive;
  std::atomic<uint> false_negative;

  metric_summary() : true_positive(0), true_negative(0), false_positive(0), false_negative(0) {}

  metric_summary(uint tp, uint tn, uint fp, uint fn) : true_positive(tp), true_negative(tn), false_positive(fp), false_negative(fn) {}

  metric_summary(const metric_summary& other)
      : true_positive(other.true_positive.load()),
        true_negative(other.true_negative.load()),
        false_positive(other.false_positive.load()),
        false_negative(other.false_negative.load()) {}

  fp_type to_score() const {
      const uint tp = true_positive.load();
      const fp_type precision = static_cast<fp_type>(tp) / (tp + false_positive.load());
      const fp_type recall = static_cast<fp_type>(tp) / (tp + false_negative.load());
      return 2 * precision * recall / (precision + recall);
  }

  void plus(const metric_summary& x) {
      true_positive.fetch_add(x.true_positive.load());
      true_negative.fetch_add(x.true_negative.load());
      false_positive.fetch_add(x.false_positive.load());
      false_negative.fetch_add(x.false_negative.load());
  }

  void zero() {
      true_positive.store(0);
      true_negative.store(0);
      false_positive.store(0);
      false_negative.store(0);
  }

  uint total() const {
      return true_positive.load() + true_negative.load() + false_positive.load() + false_negative.load();
  }
};

static metric_summary compute_metric(const dataset_local& dataset, const vector<fp_type>* w, const uint start, const uint end) {
    uint tp = 0, tn = 0, fp = 0, fn = 0;
    for (uint i = start; i < end; ++i) {
        const data_point point = dataset[i];
        const bool correct = MODEL_CHECK(w, point);
        const bool positive = point.label > 0;
        if (correct) {
            if (positive) tp++; else tn++;
        } else {
            if (positive) fn++; else fp++;
        }
    }
    return {tp, tn, fp, fn};
}

static metric_summary compute_metric(const dataset_local& dataset, const vector<fp_type>* w) {
    const uint size = dataset.get_size();
    return compute_metric(dataset, w, 0, size);
}

#endif //PSGD_MODEL_H
