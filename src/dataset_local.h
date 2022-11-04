//
// Created by Maksim.Zuev on 13.08.2022.
//

#ifndef PSGD_DATASET_LOCAL_H
#define PSGD_DATASET_LOCAL_H

#include "vectors.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <random>

const uint SIZE_UINT = sizeof(uint);
const uint SIZE_FP_TYPE = sizeof(fp_type);
const uint SIZE_CHAR_PTR = sizeof(char*);

struct tmp_point {
  std::vector<fp_type> data;
  std::vector<uint> indices;
  fp_type label{};
};

struct data_point {
  uint size;
  fp_type label;
  const uint* indices;
  const fp_type* data;
};

std::vector<tmp_point> load_dataset_from_file(const std::string& name) {
    std::ifstream in;
    in.open(name);
    if (!in) {
        std::cerr << "Failed to load dataset from " << name << std::endl;
        exit(1);
    }
    std::vector<tmp_point> tmp_points;
    std::string str;
    while (std::getline(in, str)) {
        tmp_point p;
        std::stringstream ss(str);
        fp_type x{};
        int index{}, old_index = -1;
        char c{};
        ss >> x;
        p.label = (x == 1.0) ? 1.0 : -1.0;
        while (ss >> index >> c >> x) {
            if (c != ':' || index < 1) {
                std::cerr << "Warning! error while reading dataset, split symbol is " << c << " index=" << index << name << std::endl;
                continue;
            }
            index -= 1;
            assert(index >= 0);
            assert(old_index == -1 || index > old_index);
            old_index = index;
            assert(x != 0);
            p.indices.push_back(index);
            p.data.push_back(x);
        }
        tmp_points.push_back(p);
    }

    in.close();
    return tmp_points;
}

class dataset_local {
  uint _size;
  uint _features;
  uint data_buffer_size;
  char* data;
  char** points_ptr;

  explicit dataset_local(const std::vector <tmp_point>& points, bool shuffle = true)
      : dataset_local(points.size(), points.data(), shuffle) {}
public:
  dataset_local(uint size, const tmp_point* points, bool shuffle = true) : _size(size) {
      std::vector<uint> p(_size);
      FOR_N(i, _size) {
          p[i] = i;
      }
      if (shuffle) {
          std::random_device rd;
          std::mt19937 g(rd());
          std::shuffle(p.begin(), p.end(), g);
      }

      data_buffer_size = 0;
      data_buffer_size += SIZE_CHAR_PTR * _size; // pointers to points
      FOR_N(i, _size) {
          const tmp_point& point = points[p[i]];
          data_buffer_size += SIZE_UINT; // size of point
          data_buffer_size += SIZE_FP_TYPE; // label
          data_buffer_size += SIZE_UINT * point.indices.size();
          data_buffer_size += SIZE_FP_TYPE * point.indices.size();
      }
      data = new char[data_buffer_size];
      points_ptr = reinterpret_cast<char**>(data);
      _features = 0;
      char* buffer = data + SIZE_CHAR_PTR * _size;
      FOR_N(p_i, _size) {
          const uint point_i = p[p_i];
          points_ptr[p_i] = buffer;
          const tmp_point& point = points[point_i];
          uint size = point.indices.size();
          *reinterpret_cast<uint*>(buffer) = size;
          buffer += SIZE_UINT;
          *reinterpret_cast<fp_type*>(buffer) = point.label;
          buffer += SIZE_FP_TYPE;
          FOR_N(i, size) {
              uint index = point.indices[i];
              if (_features < index) _features = index;
              *reinterpret_cast<uint*>(buffer) = index;
              buffer += SIZE_UINT;
          }
          FOR_N(i, size) {
              *reinterpret_cast<fp_type*>(buffer) = point.data[i];
              buffer += SIZE_FP_TYPE;
          }
      }
      _features++;
  }

  explicit dataset_local(const std::string& name, bool shuffle = true) : dataset_local(load_dataset_from_file(name), shuffle) {}

  dataset_local(const dataset_local& other) : _size(other._size), _features(other._features), data_buffer_size(other.data_buffer_size) {
      data = new char[data_buffer_size];
      std::copy(other.data, other.data + data_buffer_size, data);
      points_ptr = reinterpret_cast<char**>(data);
  }

  dataset_local(const dataset_local& other, const std::vector<uint>& inverse_permutation) 
          : _size(other._size), _features(other._features), data_buffer_size(other.data_buffer_size) {
      assert(_size == inverse_permutation.size());
      data = new char[data_buffer_size];
      points_ptr = reinterpret_cast<char**>(data);
      char* current = data + _size * SIZE_CHAR_PTR;
      FOR_N(i, _size) {
          assert(data + _size * SIZE_CHAR_PTR <= current && current < data + data_buffer_size);
          points_ptr[i] = current;
          uint index = inverse_permutation[i];
          data_point point = other[index];
          char* buffer = other.points_ptr[index];
          uint length = (SIZE_UINT + SIZE_FP_TYPE) * (point.size + 1);
          std::copy(buffer, buffer + length, current);
          current += length;
      }
  }

  inline uint get_size() const {
      return _size;
  }

  inline uint get_features() const {
      return _features;
  }

  inline data_point operator[](const uint index) const {
      data_point point{};
      const char* buffer = points_ptr[index];
      const uint point_size = *reinterpret_cast<const uint*>(buffer);
      point.size = point_size;
      buffer += SIZE_UINT;
      point.label = *reinterpret_cast<const fp_type*>(buffer);
      buffer += SIZE_FP_TYPE;
      point.indices = reinterpret_cast<const uint*>(buffer);
      buffer += SIZE_UINT * point_size;
      point.data = reinterpret_cast<const fp_type*>(buffer);
      return point;
  }

  ~dataset_local() {
      delete[] data;
  }
};

#endif //PSGD_DATASET_LOCAL_H
