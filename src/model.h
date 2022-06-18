//
// Created by Maksim.Zuev on 29.05.2022.
//

#ifndef PSGD_MODEL_H
#define PSGD_MODEL_H

#include "types.h"
#include "dataset.h"
#include "data_scheme.h"


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
          const char* data = point.data;
          FOR_N(j, point.size) {
              degrees[data_point::get_next_index(data)]++;
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
      const char* data = point.data;
      FOR_N(i, size) {
          uint index;
          fp_type b_val;
          data_point::get_next(data, &index, &b_val);
          const fp_type a_val = a_data[index];
          result += a_val * b_val;
      }
      return result;
  }

  inline void scale_and_add(fp_type* const __restrict__ a_data,
                            const data_point& point,
                            const fp_type s) {
      const uint size = point.size;
      const char* data = point.data;
      FOR_N(i, size) {
          uint index;
          fp_type b_val;
          data_point::get_next(data, &index, &b_val);
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
      const fp_type scalar = step * args->mu;
      const uint size = point.size;
      const char* data = point.data;
      FOR_N(i, size) {
          const uint j = data_point::get_next_index(data);
          const uint deg = degrees[j];
          vals[j] *= 1 - scalar / deg;
      }
  }
}

#define MODEL_UPDATE svm::update
#define MODEL_CHECK svm::check
#define MODEL_PARAMS SVMParams

static uint compute_correct(const dataset_local& dataset, const vector<fp_type>* w, const uint start, const uint end) {
    uint correct = 0;
    for (uint i = start; i < end; ++i) {
        const data_point point = dataset[i];
        correct += MODEL_CHECK(w, point);
    }
    return correct;
}

static fp_type compute_accuracy(const dataset_local& dataset, const vector<fp_type>* w) {
    const uint size = dataset.get_size();
    const uint correct = compute_correct(dataset, w, 0, size);
    return static_cast<fp_type>(correct) / size;
}

#endif //PSGD_MODEL_H
