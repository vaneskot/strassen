// Author: Ivan Kotenkov <koteniv1@fit.cvut.cz>

#include <algorithm>
#include <random>

#include "utils.h"

void FillVector(RealType* v, IndexType n, unsigned seed) {
  std::uniform_real_distribution<RealType> unif(-1., 1.);
  std::default_random_engine re(seed);
  for (size_t i = 0; i < n; ++i) {
    // v[i] = unif(re);
    v[i] = 1.;
  }
}

void PrintMatrix(const RealType* m, int n, int max_elements) {
  const int upper_bound = std::min(n, max_elements);
  for (int i = 0; i < upper_bound; ++i) {
    for (int j = 0; j < upper_bound; ++j) {
      std::cout << m[i * n + j] << " ";
    }
    std::cout << std::endl;
  }
}
