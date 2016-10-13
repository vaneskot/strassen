// Author: Ivan Kotenkov <koteniv1.fit.cvut.cz>

#include <assert.h>
#include <cmath>
#include <cstring>
#include <ctime>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "strassen.h"

template <typename ClosureType>
void TimeClosure(ClosureType closure, const std::string& label) {
  auto start_time = std::chrono::high_resolution_clock::now();
  closure();
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(
      end_time - start_time);
  std::cout << label << ": " << duration.count() << "s\n";
}

template <typename MultFuncType>
void TimeMultiply(MultFuncType mult_func, double *a, double *b, size_t n,
                  double *res, const std::string &label) {
  TimeClosure([mult_func, a, b, n, res](){mult_func(a, b, n, res);}, label);
}

void FillVector(double* v, size_t n) {
  std::uniform_real_distribution<double> (-10., 10);
  std::default_random_engine re;
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
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " matrix_sizes" << std::endl;
    return 0;
  }
  for (int i = 1; i < argc; ++i) {
    const IndexType matrix_size = atoi(argv[i]);
    const IndexType full_array_size = matrix_size * matrix_size;
    std::unique_ptr<double[]> a(new double[full_array_size]);
    std::unique_ptr<double[]> b(new double[full_array_size]);

    FillVector(a.get(), full_array_size);
    FillVector(b.get(), full_array_size);

    std::unique_ptr<double[]> res_strassen(new double[full_array_size]);
    memset(res_strassen.get(), 0, full_array_size * sizeof(double));
    TimeMultiply(MultiplyStrassen, a.get(), b.get(), matrix_size,
                 res_strassen.get(), "MultiplyStrassen size = " +
                                         std::to_string(matrix_size) + " ");
  }
  return 0;
}
