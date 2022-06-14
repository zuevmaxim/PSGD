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
          FOR_N(j, point.size) {
              degrees[point.indices[j]]++;
          }
      }
      return degrees;
  }
};

namespace svm {
  static inline bool check(const vector<fp_type>* w, const data_point& point) {
      const fp_type dot = vectors::dot(w->data, point.data, point.indices, point.size);
      return std::max(dot * point.label, 0.0) != 0;
  }

  static inline void update(const data_point& point, vector<fp_type>* w, const fp_type step, const SVMParams* args) {
      const uint size = point.size;
      const uint* const __restrict__ indices = point.indices;
      fp_type* const __restrict__ vals = w->data;
      const fp_type wxy = vectors::dot(vals, point.data, indices, size) * point.label;

      if (wxy < 1) { // hinge is active.
          const fp_type e = step * point.label;
          vectors::scale_and_add(vals, point.data, indices, size, e);
      }

      const uint* const __restrict__ degrees = args->degrees.data;
      const fp_type scalar = step * args->mu;
      FOR_N(i, size) {
          const int j = indices[i];
          const unsigned deg = degrees[j];
          vals[j] *= 1 - scalar / deg;
      }
  }
}

#define MODEL_UPDATE svm::update
#define MODEL_CHECK svm::check
#define MODEL_PARAMS SVMParams

static fp_type compute_accuracy(const dataset_local& dataset, const vector<fp_type>* w) {
    const uint size = dataset.get_size();
    uint correct = 0;
    FOR_N(i, size) {
        const data_point point = dataset[i];
        correct += MODEL_CHECK(w, point);
    }
    return static_cast<fp_type>(correct) / size;
}

#endif //PSGD_MODEL_H
