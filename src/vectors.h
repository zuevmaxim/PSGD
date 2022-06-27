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
  uint size = 0;
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


#endif //PSGD_VECTOR_H
