// Author: Ivan Kotenkov <koteniv1.fit.cvut.cz>

#include <assert.h>
#include <cmath>
#include <ctime>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>

void MultiplyStrassen(double* a, double* b, int n, double* c);
void MultiplySimple(double* a, double* b, int n, double* res);

template <typename ClosureType>
void TimeClosure(ClosureType closure, const std::string& label) {
  double start_time = clock();
  closure();
  std::cout << label << ": " << (clock() - start_time) / CLOCKS_PER_SEC << "\n";
}

template <typename MultFuncType>
void TimeMultiply(MultFuncType mult_func, double *a, double *b, size_t n,
                  double *res, const std::string &label) {
  TimeClosure([mult_func, a, b, n, res](){mult_func(a, b, n, res);}, label);
}

void FillVector(double* v, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    v[i] = i;
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
    std::cout << "Usage: " << argv[0] << " matrix_size" << std::endl;
    return 0;
  }
  const int matrix_size = atoi(argv[1]);
  const int full_array_size = matrix_size * matrix_size;
  std::unique_ptr<double[]> a(new double[full_array_size]);
  std::unique_ptr<double[]> b(new double[full_array_size]);

  FillVector(a.get(), full_array_size);
  FillVector(b.get(), full_array_size);

  std::unique_ptr<double[]> res(new double[full_array_size]);
  memset(res.get(), 0, full_array_size * sizeof(double));
  TimeMultiply(MultiplySimple, a.get(), b.get(), matrix_size, res.get(),
               "MultiplySimple");
  PrintMatrix(res.get(), matrix_size, 5);

  std::unique_ptr<double[]> res_strassen(new double[full_array_size]);
  memset(res_strassen.get(), 0, full_array_size * sizeof(double));
  TimeMultiply(MultiplyStrassen, a.get(), b.get(), matrix_size,
               res_strassen.get(), "MultiplyStrassen");

  PrintMatrix(res_strassen.get(), matrix_size, 5);

  const double kEps = 1e-10;

  for (size_t i = 0; i < full_array_size; ++i) {
    assert(fabs(res[i] - res_strassen[i]) < kEps);
  }
  return 0;
}
