//
// Created by Maksim.Zuev on 29.05.2022.
//

#ifndef PSGD_VECTOR_H
#define PSGD_VECTOR_H

#include "types.h"
#include <cassert>
#include <algorithm>

template<typename T>
class vector {
public:
  uint size;
  T* data = NULL;

  vector() = default;

  vector(const vector& other) {
      init(other.size);
      std::copy(other.data, other.data + size, data);
  }

  void init(uint size) {
      this->size = size;
      assert(data == NULL);
      data = new T[size];
  }

  void init(uint size, const T& value) {
      init(size);
      std::fill(data, data + size, value);
  }

  inline T& operator[](unsigned index) {
      return data[index];
  }

  inline const T& operator[](unsigned index) const {
      return data[index];
  }

  ~vector() {
      if (data != NULL) {
          delete[] data;
      }
  }
};

namespace vectors {
  inline fp_type dot(const fp_type* const __restrict__ a_data,
                     const fp_type* const __restrict__ b_data,
                     const uint* __restrict__ indices,
                     const uint size) {
      fp_type result = 0;
      FOR_N(i, size) {
          const uint j = indices[i];
          const fp_type b_val = b_data[i];
          const fp_type a_val = a_data[j];
          result += a_val * b_val;
      }
      return result;
  }

  inline void scale_and_add(fp_type* const __restrict__ a_data,
                            const fp_type* const __restrict__ b_data,
                            const uint* __restrict__ indices,
                            const uint size,
                            const fp_type s) {
      FOR_N(i, size) {
          const uint j = indices[i];
          const fp_type b_val = b_data[i];
          a_data[j] += s * b_val;
      }
  }
}


#endif //PSGD_VECTOR_H
