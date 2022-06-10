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
          const data_point& point = points[i];
          FOR_N(j, point.size) {
              degrees[point.indices[j]]++;
          }
      }
      return degrees;
  }
};

namespace svm {
  static inline bool predict(const vector<fp_type>& w, const data_point& point) {
      fp_type dot = vectors::dot(w, point.data, point.indices, point.size);
      return dot > 0;
  }

  static inline void update(const data_point& point, vector<fp_type>& w, const fp_type step, const SVMParams* args) {
      const SVMParams& params = *args;
      fp_type wxy = vectors::dot(w, point.data, point.indices, point.size) * point.label;

      if (wxy < 1) { // hinge is active.
          const fp_type e = step * point.label;
          vectors::scale_and_add(w, point.data, point.indices, point.size, e);
      }

      fp_type* const __restrict__ vals = w.data;
      const uint* const __restrict__ degrees = params.degrees.data;
      const uint* const __restrict__ indices = point.indices;
      const uint size = point.size;

      const fp_type scalar = step * params.mu;
      FOR_N(i, size) {
          const int j = indices[i];
          const unsigned deg = degrees[j];
          vals[j] *= 1 - scalar / deg;
      }
  }
}

#define MODEL_UPDATE svm::update
#define MODEL_PREDICT svm::predict
#define MODEL_PARAMS SVMParams

static fp_type compute_accuracy(const dataset& dataset, const vector<fp_type>& w, uint node = 0) {
    const dataset_local& points = dataset.get_data(node);
    const uint size = points.get_size();
    uint correct = 0;
    FOR_N(i, size) {
        const data_point& point = points[i];
        bool predicted = MODEL_PREDICT(w, point);
        bool expected = point.label > 0;
        correct += (predicted == expected) ? 1 : 0;
    }
    return static_cast<fp_type>(correct) / size;
}

#endif //PSGD_MODEL_H
