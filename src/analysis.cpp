#include <iostream>
#include "dataset_local.h"
#include <chrono>
#include <random>
#include <cassert>
#include <deque>
#include <unordered_set>

bool VERBOSE = false;
uint GROUPS = 0;
uint N = 0;
uint F = 0;
uint PER_PART = 0;
const dataset_local* my_dataset;

typedef std::pair<uint, uint> Swap;

typedef std::pair<int, uint> ScoreAndIndex;

struct Preference {
  uint index_in_permutation;
  uint from_group;
  uint to_group;
  int score_diff;
};

std::mt19937 create_random_generator() {
    const auto ns = std::chrono::high_resolution_clock::now().time_since_epoch();
    const uint seed = std::chrono::duration_cast<std::chrono::nanoseconds>(ns).count();
    std::mt19937 gen(seed);
    return gen;
}

struct Individual {
  std::vector<uint> permutation;
  std::vector<Swap> swaps;
  std::vector<std::vector<uint>> group_count;
  uint cache_score = -1;
  std::uniform_int_distribution<uint> distribution;
  std::mt19937 gen;

  Individual() : distribution(0, N - 1), gen(create_random_generator()) {
      group_count.resize(GROUPS);
      permutation.resize(N);
      FOR_N(i, N) {
          permutation[i] = i;
      }
      FOR_N(group, GROUPS) {
          group_count[group].resize(F, 0);
      }
  }

  explicit Individual(bool shuffle) : Individual() {
      if (shuffle) {
          std::shuffle(permutation.begin(), permutation.end(), gen);
      }
      calculate_group_count();
  }

  void apply_swaps() {
      for (Swap& swap: swaps) {
          uint i = swap.first, j = swap.second;
          uint part_i = get_part(i), part_j = get_part(j);
          assert(part_i != part_j);
          move_element(i, part_i, part_j);
          move_element(j, part_j, part_i);
          std::swap(permutation[i], permutation[j]);
      }
      swaps.clear();
  }

  void move_element(uint index, uint part_from, uint part_to) {
      const bool update_cache = cache_score != -1;
      const data_point& point = (*my_dataset)[permutation[index]];
      FOR_N(k, point.size) {
          uint f = point.indices[k];
          if (update_cache) {
              cache_score -= get_score(part_from, f);
          }
          group_count[part_from][f]--;
          if (update_cache) {
              cache_score += get_score(part_from, f);
              cache_score -= get_score(part_to, f);
          }
          group_count[part_to][f]++;
          if (update_cache) {
              cache_score += get_score(part_to, f);
          }
      }
  }

  static inline uint get_part(uint i) {
      return std::min(GROUPS - 1, i / PER_PART);
  }

  Swap mutate() {
      uint i = distribution(gen), j = distribution(gen);
      uint part_i = get_part(i), part_j = get_part(j);
      while (part_i == part_j) {
          j = distribution(gen);
          part_j = get_part(j);
      }
      swaps.emplace_back(i, j);
      return swaps.back();
  }

  void revert_mutation(const Swap& swap) {
      swaps.emplace_back(swap.second, swap.first);
  }

  uint get_score() {
      if (!swaps.empty()) {
          apply_swaps();
      }
      if (cache_score != -1) return cache_score;

      uint total = 0;
      FOR_N(f, F) {
          total += get_score(f);
      }
      cache_score = total;
      return total;
  }

  inline uint get_score(uint f) const {
      uint total = 0;
      FOR_N(i, GROUPS) {
          for (uint j = i + 1; j < GROUPS; ++j) {
              total += std::min(group_count[i][f], group_count[j][f]);
          }
      }
      return total;
  }

  inline uint get_score(uint g, uint f) const {
      uint total = 0;
      FOR_N(j, GROUPS) {
          if (j == g) continue;
          total += std::min(group_count[g][f], group_count[j][f]);
      }
      return total;
  }

  std::vector<Preference> score_possible_groups(uint index_in_permutation) {
      std::vector<Preference> result;
      result.reserve(GROUPS - 1);
      get_score();
      uint original_index = permutation[index_in_permutation];
      uint my_part = get_part(index_in_permutation);
      const data_point& point = (*my_dataset)[original_index];
      FOR_N(part, GROUPS) {
          if (part == my_part) continue;
          int current_features_score = 0;
          int new_features_score = 0;
          FOR_N(k, point.size) {
              uint f = point.indices[k];
              current_features_score += (int) get_score(f);
              group_count[my_part][f]--;
              group_count[part][f]++;
              new_features_score += (int) get_score(f);
              group_count[my_part][f]++;
              group_count[part][f]--;
          }
          result.push_back({index_in_permutation, my_part, part, new_features_score - current_features_score});
      }
      return result;
  }

  bool operator<(const Individual& other) const {
      auto* my = const_cast<Individual*>(this);
      auto* o = const_cast<Individual*>(&other);
      return my->get_score() < o->get_score();
  }

  bool operator>(const Individual& other) const {
      auto* my = const_cast<Individual*>(this);
      auto* o = const_cast<Individual*>(&other);
      return my->get_score() > o->get_score();
  }

  bool operator<=(const Individual& other) const {
      auto* my = const_cast<Individual*>(this);
      auto* o = const_cast<Individual*>(&other);
      return my->get_score() <= o->get_score();
  }

  bool operator>=(const Individual& other) const {
      auto* my = const_cast<Individual*>(this);
      auto* o = const_cast<Individual*>(&other);
      return my->get_score() >= o->get_score();
  }

  bool dump(const std::string& file_name) {
      apply_swaps();
      std::ofstream file;
      file.open(file_name);
      if (!file.good()) {
          std::cerr << "Failed to open file " << file_name << "!" << std::endl;
          return false;
      }

      for (uint i: permutation) {
          file << i << '\n';
      }

      file.close();
      return true;
  }

  void sort_in_groups() {
      std::vector<int> point_score_buffer(N, -1);
      FOR_N(part, GROUPS) {
          uint begin = PER_PART * part;
          uint end = part == GROUPS - 1 ? N : PER_PART * (part + 1);
          std::sort(permutation.data() + begin, permutation.data() + end, [&](int i, int j) {
            return score_point_in_group(part, i, point_score_buffer) < score_point_in_group(part, j, point_score_buffer);
          });
      }
  }

private:
  int score_point_in_group(uint part, int i, std::vector<int>& score_buffer) {
      int& point_score = score_buffer[i];
      if (point_score != -1) return point_score;
      int score_i = 0;
      const data_point& point_i = (*my_dataset)[i];
      FOR_N(s, point_i.size) {
          uint f = point_i.indices[s];
          FOR_N(g, GROUPS) {
              if (g == part) {
                  score_i += (GROUPS - 1) * group_count[g][f];
              } else {
                  score_i -= group_count[g][f];
              }
          }
      }
      point_score = score_i;
      return score_i;
  }

  void calculate_group_count() {
      FOR_N(i, GROUPS) {
          std::fill(group_count[i].begin(), group_count[i].end(), 0);
      }
      FOR_N(i, my_dataset->get_size()) {
          uint part = get_part(i);
          const data_point& point = (*my_dataset)[permutation[i]];
          FOR_N(j, point.size) {
              group_count[part][point.indices[j]]++;
          }
      }
      get_score();
  }

};

double get_improvement(const uint initial_score, uint score) {
    return int((1 - double(score) / initial_score) * 1000) / 10.0;
}

void genetic_algorithm(Individual& best, uint fail_tries_threshold, uint max_failed_epochs) {
    const uint initial_best_score = best.get_score();
    uint current_score = initial_best_score;

    uint failed_epochs = 0;
    FOR_N(epoch, INT_MAX) {
        uint failed_tries = 0;
        while (failed_tries <= fail_tries_threshold) {
            Swap swap = best.mutate();
            uint score = best.get_score();
            if (score < current_score) {
                current_score = score;
                break;
            }
            best.revert_mutation(swap);
            failed_tries++;
        }
        if (failed_tries > fail_tries_threshold) {
            failed_epochs++;
            if (VERBOSE) {
                std::cout << failed_tries << " was not enough to generate new mutation " << failed_epochs << "/" << max_failed_epochs << std::endl;
            }
            if (failed_epochs >= max_failed_epochs) break;
        }

        if (VERBOSE) {
            if ((epoch & 0xfff) == 0xfff) {
                std::cout << "epoch " << epoch + 1
                          << " best score is " << current_score
                          << " it is " << get_improvement(initial_best_score, current_score) << "% less than initial score"
                          << std::endl;
            }
        }
    }

    best.sort_in_groups();
}

std::vector<std::vector<std::vector<ScoreAndIndex>>> get_preferences(Individual& best) {
    std::vector<std::vector<std::vector<ScoreAndIndex>>> s;
    s.resize(GROUPS);
    FOR_N(i, GROUPS) s[i].resize(GROUPS);
    FOR_N(i, N) {
        auto preferences = best.score_possible_groups(i);
        for (auto& preference: preferences) {
            if (preference.from_group == preference.to_group) continue;
            s[preference.from_group][preference.to_group].emplace_back(preference.score_diff, preference.index_in_permutation);
        }
    }
    FOR_N(i, GROUPS) {
        FOR_N(j, GROUPS) {
            std::sort(s[i][j].begin(), s[i][j].end(), std::greater<ScoreAndIndex>());
        }
    }
    return s;
}


bool find_best_swap(std::vector<std::vector<std::vector<ScoreAndIndex>>>& s,
                    std::unordered_set<uint>& used_indices,
                    std::vector<uint>& chain) {
    FOR_N(i, GROUPS) {
        FOR_N(j, GROUPS) {
            while (!s[i][j].empty() && used_indices.find(s[i][j].back().second) != used_indices.end()) {
                s[i][j].pop_back();
            }
        }
    }
    std::vector<std::vector<std::vector<std::pair<int, std::vector<uint>>>>> d(GROUPS - 1);
    int INF = 1e6;
    FOR_N(r, GROUPS - 1) {
        d[r].resize(GROUPS);
        FOR_N(i, GROUPS) {
            d[r][i].resize(GROUPS, {INF, {}});
            if (r > 0) continue;
            FOR_N(j, GROUPS) {
                if (i == j) continue;
                if (s[i][j].empty()) continue;
                d[r][i][j] = {s[i][j].back().first, {}};
            }
        }
    }
    for (uint r = 1; r < GROUPS - 1; ++r) {
        FOR_N(i, GROUPS) {
            FOR_N(j, GROUPS) {
                if (i == j) continue;
                d[r][i][j] = d[r - 1][i][j];
                FOR_N(k, GROUPS) {
                    if (k == i || k == j) continue;
                    if (s[i][k].empty()) continue;
                    if (d[r - 1][k][j].first == INF) continue;
                    if (std::find(d[r - 1][k][j].second.begin(), d[r - 1][k][j].second.end(), i) != d[r - 1][k][j].second.end()) {
                        continue;
                    }
                    int candidate_score = s[i][k].back().first + d[r - 1][k][j].first;
                    if (d[r][i][j].first == INF || candidate_score < d[r][i][j].first) {
                        d[r][i][j].first = candidate_score;
                        d[r][i][j].second = d[r - 1][k][j].second;
                        d[r][i][j].second.push_back(k);
                    }
                }
            }
        }
    }
    int min_score = INF;
    uint gi, gj;
    uint R = GROUPS - 2;
    FOR_N(i, GROUPS) {
        FOR_N(j, GROUPS) {
            if (i == j) continue;
            if (d[R][i][j].first == INF) continue;
            if (s[j][i].empty()) continue;
            if (s[j][i].back().first + d[R][i][j].first < min_score) {
                gi = i;
                gj = j;
                min_score = s[j][i].back().first + d[R][i][j].first;
            }
        }
    }

    if (min_score == INF || min_score >= 0) return false;
    chain.clear();
    std::vector<uint> group_seq;
    group_seq.reserve(2 + d[R][gi][gj].second.size());
    group_seq.push_back(gi);
    for (uint i = d[R][gi][gj].second.size(); i-- > 0;) {
        group_seq.push_back(d[R][gi][gj].second[i]);
    }
    group_seq.push_back(gj);
    FOR_N(k, group_seq.size()) {
        uint i = k > 0 ? group_seq[k - 1] : group_seq[group_seq.size() - 1];
        uint j = group_seq[k];
        chain.push_back(s[i][j].back().second);
    }

    return true;
}

void greedy_algorithm(Individual& best, uint max_epochs) {
    const uint initial_best_score = best.get_score();

    uint current_score = initial_best_score;
    FOR_N(ii, max_epochs) {
        auto s = get_preferences(best);
        std::unordered_set<uint> used_indices;
        while (true) {
            std::vector<uint> chain;
            if (!find_best_swap(s, used_indices, chain)) break;
            FOR_N(i, chain.size()) {
                used_indices.insert(chain[i]);
                if (i >= 1) {
                    best.swaps.emplace_back(chain[0], chain[i]);
                }
            }
        }
        if (best.swaps.empty()) break;
        uint current_best_score = best.get_score();
        if (current_best_score >= current_score) break;
        current_score = current_best_score;
        if (VERBOSE) {
            std::cout << "best score is " << current_best_score
                      << " it is " << get_improvement(initial_best_score, current_score) << "% less than initial score"
                      << std::endl;
        }
    }

    best.sort_in_groups();
}


int main(int argc, char** argv) {
    assert(argc >= 5);

    int splits = std::atoi(argv[1]);
    GROUPS = std::atoi(argv[2]);
    std::string dataset_path = argv[3];
    std::string output_path = argv[4];

    uint fail_tries_threshold = 300;
    uint max_failed_epochs = 100;

    VERBOSE = "-v" == std::string(argv[argc - 1]);

    Individual result;
    auto points = load_dataset_from_file(dataset_path);
    uint per_split = points.size() / splits;
    for (int s = 0; s < splits; ++s) {
        N = s == splits - 1 ? points.size() - s * per_split : per_split;
        uint offset = s * per_split;
        my_dataset = new dataset_local(N, points.data() + offset, false);
        F = my_dataset->get_features();
        PER_PART = N / GROUPS;

        Individual best = Individual(false);
        const uint initial_score = best.get_score();

        genetic_algorithm(best, fail_tries_threshold, max_failed_epochs);
        double genetic_improvement = get_improvement(initial_score, best.get_score());

        greedy_algorithm(best, 10);
        double greedy_improvement = get_improvement(initial_score, best.get_score());

        best.apply_swaps();
        std::copy(best.permutation.begin(), best.permutation.end(), std::back_inserter(result.permutation));
        for (int i = 0; i < N; ++i) {
            result.permutation[i + offset] += offset;
        }

        std::cout << "Optimization completed. Initial score was " << initial_score
                  << " genetic optimized " << genetic_improvement << "%"
                  << " greedy optimized " << greedy_improvement - genetic_improvement << "%"
                  << std::endl;

        delete my_dataset;
    }
    result.dump(output_path);

    return 0;
}
