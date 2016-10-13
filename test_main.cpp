// Author: Ivan Kotenkov <koteniv1@fit.cvut.cz>

#include <assert.h>
#include <cstring>

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>

#include "strassen.h"

void MultiplySimple(double* a, double* b, int n, double* res);

void FillVector(double* v, size_t n, unsigned seed) {
  std::uniform_real_distribution<double> unif(-10., 10);
  std::default_random_engine re(seed);
  for (size_t i = 0; i < n; ++i) {
    v[i] = unif(re);
  }
}

void PrintMatrix(const double* m, int n, int max_elements) {
  const int upper_bound = std::min(n, max_elements);
  for (int i = 0; i < upper_bound; ++i) {
    for (int j = 0; j < upper_bound; ++j) {
      std::cout << m[i * n + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, char* argv[]) {
  const IndexType kSizes[] = {1, 2, 3, 8, 16, 17, 31, 32, 64, 65, 128};
  const double kEps = 1e-10;

  for (IndexType matrix_size : kSizes) {
    std::cout << "Size: " << matrix_size;
    const IndexType full_array_size = matrix_size * matrix_size;
    std::unique_ptr<double[]> a(new double[full_array_size]);
    std::unique_ptr<double[]> b(new double[full_array_size]);

    FillVector(a.get(), full_array_size, 1);
    FillVector(b.get(), full_array_size, 2);

    std::unique_ptr<double[]> res(new double[full_array_size]);
    memset(res.get(), 0, full_array_size * sizeof(double));
    MultiplySimple(a.get(), b.get(), matrix_size, res.get());

    std::unique_ptr<double[]> res_strassen(new double[full_array_size]);
    memset(res_strassen.get(), 0, full_array_size * sizeof(double));
    MultiplyStrassen(a.get(), b.get(), matrix_size, res_strassen.get());

    bool correct = true;
    for (IndexType i = 0; i < full_array_size; ++i) {
      if (fabs(res[i] - res_strassen[i]) > kEps) {
        correct = false;
        std::cout << std::endl;
        std::cout << "First difference: index: (" << i / matrix_size << ", "
                  << i % matrix_size << "); simple: " << res[i]
                  << "; strassen: " << res_strassen[i] << std::endl;
        break;
      }
    }

    if (!correct) {
      std::cout << "A" << std::endl;
      PrintMatrix(a.get(), matrix_size, 5);
      std::cout << "B" << std::endl;
      PrintMatrix(b.get(), matrix_size, 5);

      std::cout << "Res simple" << std::endl;
      PrintMatrix(res.get(), matrix_size, 5);

      std::cout << "Res strassen" << std::endl;
      PrintMatrix(res_strassen.get(), matrix_size, 5);
      assert(false);
    }
    std::cout << " -- OK" << std::endl;
  }
  return 0;
}
