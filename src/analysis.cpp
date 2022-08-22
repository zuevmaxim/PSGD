#include <iostream>
#include "dataset_local.h"
#include <chrono>
#include <random>
#include <cassert>
#include <deque>
#include <unordered_set>

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
  std::vector<std::vector<uint>> score;
  uint cache_score = -1;

  Individual() {
      score.resize(GROUPS);
      permutation.resize(N);
      FOR_N(i, N) {
          permutation[i] = i;
      }
      FOR_N(group, GROUPS) {
          score[group].resize(F, 0);
      }
  }

  explicit Individual(bool shuffle) : Individual() {
      if (shuffle) {
          std::shuffle(permutation.begin(), permutation.end(), create_random_generator());
      }
      calculate_score();
  }

  void apply_swaps(bool recalculate) {
      for (Swap& swap: swaps) {
          uint i = swap.first, j = swap.second;
          uint part_i = get_part(i), part_j = get_part(j);
          if (recalculate) {
              const data_point& point_i = (*my_dataset)[permutation[i]];
              FOR_N(k, point_i.size) {
                  score[part_i][point_i.indices[k]]--;
                  score[part_j][point_i.indices[k]]++;
              }
              const data_point& point_j = (*my_dataset)[permutation[j]];
              FOR_N(k, point_j.size) {
                  score[part_i][point_j.indices[k]]++;
                  score[part_j][point_j.indices[k]]--;
              }
          }
          std::swap(permutation[i], permutation[j]);
      }
      swaps.clear();
  }

  static inline uint get_part(uint i) {
      return std::min(GROUPS - 1, i / PER_PART);
  }

  Swap mutate() {
      std::uniform_int_distribution<uint> distribution(0, N - 1);
      auto gen = create_random_generator();
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
      if (swaps.empty() && cache_score != -1) return cache_score;
      apply_swaps(true);
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
              total += std::min(score[i][f], score[j][f]);
          }
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
              score[my_part][f]--;
              score[part][f]++;
              new_features_score += (int) get_score(f);
              score[my_part][f]++;
              score[part][f]--;
          }
          result.push_back({index_in_permutation, my_part, part, new_features_score - current_features_score});
      }
      return result;
  }

  void calculate_score() {
      apply_swaps(false);
      FOR_N(i, my_dataset->get_size()) {
          uint part = get_part(i);
          const data_point& point = (*my_dataset)[permutation[i]];
          FOR_N(j, point.size) {
              score[part][point.indices[j]]++;
          }
      }
      cache_score = get_score();
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
      apply_swaps(true);
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

  static Individual load(const std::string& file_name) {
      Individual result;
      std::ifstream file;
      file.open(file_name);
      if (!file.good()) {
          std::cerr << "Failed to open file " << file_name << "!" << std::endl;
          return result;
      }
      uint i = 0, index;
      while (file >> index) {
          result.permutation[i++] = index;
      }
      result.calculate_score();
      return result;
  }

private:
  int score_point(uint part, int i) {
      int score_i = 0;
      const data_point& point_i = (*my_dataset)[i];
      FOR_N(s, point_i.size) {
          uint f = point_i.indices[s];
          FOR_N(g, GROUPS) {
              if (g == part) {
                  score_i += (GROUPS - 1) * score[g][f];
              } else {
                  score_i -= score[g][f];
              }
          }
      }
      return score_i;
  }

public:

  void sort_in_groups() {
      FOR_N(part, GROUPS) {
          uint begin = PER_PART * part;
          uint end = part == GROUPS - 1 ? N : PER_PART * (part + 1);
          std::sort(permutation.data() + begin, permutation.data() + end, [&](int i, int j) {
            return score_point(part, i) < score_point(part, j);
          });
      }
  }
};


Individual genetic_algorithm(const std::string& output_dir, uint fail_tries_threshold, uint max_failed_epochs) {
    Individual best(false);

    uint initial_best_score = best.get_score();
    uint current_best_score = initial_best_score;
    std::cout << "Initial score is " << initial_best_score << std::endl;

    int best_improvement = 0;
    uint failed_epochs = 0;
    FOR_N(epoch, INT_MAX) {
        uint failed_tries = 0;
        while (failed_tries <= fail_tries_threshold) {
            Swap swap = best.mutate();
            uint score = best.get_score();
            if (score < current_best_score) {
                current_best_score = score;
                break;
            }
            best.revert_mutation(swap);
            failed_tries++;
        }
        if (failed_tries > fail_tries_threshold) {
            failed_epochs++;
            std::cout << failed_tries << " was not enough to generate new mutation " << failed_epochs << "/" << max_failed_epochs << std::endl;
            if (failed_epochs >= max_failed_epochs) break;
        }
        if ((epoch & 0xfff) == 0xfff) {
            int improvement = int((1 - double(current_best_score) / initial_best_score) * 100);
            if (improvement > best_improvement) {
                best_improvement = improvement;
                bool success = best.dump(output_dir + "/" + std::to_string(improvement) + ".txt");
                if (success) {
                    std::remove((output_dir + "/" + std::to_string(improvement - 1) + ".txt").c_str());
                }
            }
            std::cout << "epoch " << epoch + 1
                      << " best score is " << current_best_score
                      << " it is " << improvement << "% less than initial score"
                      << std::endl;
        }
    }
    best.dump(output_dir + "/best.txt");
    return best;
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

Individual greedy_algorithm(const std::string& output_dir, const std::string& initial_individual) {
    Individual best = initial_individual == "basic" ? Individual(false) : Individual::load(initial_individual);
    uint initial_best_score = best.get_score();

    std::cout << "Initial score is " << initial_best_score << std::endl;

    uint current_score = initial_best_score;
    while (true) {
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
        int improvement = int((1 - double(current_best_score) / initial_best_score) * 100);
        std::cout << "best score is " << current_best_score
                  << " it is " << improvement << "% less than initial score"
                  << std::endl;
    }

    best.sort_in_groups();
    best.dump(output_dir + "/best-final.txt");

    return best;
}


int main(int argc, char** argv) {
    assert(argc >= 4);

    GROUPS = std::atoi(argv[1]);
    std::string dataset_path = argv[2];
    std::string output_dir = argv[3];

    uint fail_tries_threshold = 200;
    uint max_failed_epochs = 100;


    my_dataset = new dataset_local(dataset_path, false);
    N = my_dataset->get_size();
    F = my_dataset->get_features();
    PER_PART = N / GROUPS;
    std::cout << "Dataset loaded! " << my_dataset->get_size() << "x" << my_dataset->get_features() << std::endl;

    if (argc == 4) {
        genetic_algorithm(output_dir, fail_tries_threshold, max_failed_epochs);
    } else {
        std::string initial = argv[4];
        greedy_algorithm(output_dir, initial);
    }
}
