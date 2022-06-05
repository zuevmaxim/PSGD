//
// Created by Maksim.Zuev on 29.05.2022.
//

#ifndef PSGD_VECTOR_H
#define PSGD_VECTOR_H

#include "types.h"

template<typename T>
class vector {
public:
  uint size;
  T* data = NULL;

  vector() = default;

  vector(const vector& other) {
      init(other.size);
      FOR_N(i, size) {
          data[i] = other.data[i];
      }
  }

  void init(uint size) {
      this->size = size;
      data = new T[size];
  }

  T& operator[](unsigned index) {
      return data[index];
  }

  const T& operator[](unsigned index) const {
      return data[index];
  }

  ~vector() {
      if (data != NULL) {
          delete[] data;
      }
  }
};

struct sparse_vector {
  const uint size;
  const uint* const indices;
  const fp_type* const data;

  sparse_vector(uint size, const uint* indices, const fp_type* data) : size(size), indices(indices), data(data) {}
};


namespace vectors {
  inline fp_type dot(const vector<fp_type>& a,
                     const fp_type* const __restrict__ b_data,
                     const uint* __restrict__ indices,
                     const uint size) {
      fp_type result = 0;
      const fp_type* const __restrict__ a_data = a.data;
      FOR_N(i, size) {
          result += a_data[indices[i]] * b_data[i];
      }
      return result;
  }

  inline void scale_and_add(vector<fp_type>& a,
                            const fp_type* const __restrict__ b_data,
                            const uint* __restrict__ indices,
                            const uint size,
                            const fp_type s) {
      fp_type* const __restrict__ a_data = a.data;
      FOR_N(i, size) {
          a_data[indices[i]] += s * b_data[i];
      }
  }
}


#endif //PSGD_VECTOR_H
