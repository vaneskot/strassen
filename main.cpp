// Author: Ivan Kotenkov <koteniv1.fit.cvut.cz>

#include <assert.h>
#include <cmath>
#include <cstring>

#include <iostream>
#include <memory>
#include <string>

#include "strassen.h"

template <typename MultFuncType>
void TimeMultiply(MultFuncType mult_func, RealType *a, RealType *b, size_t n,
                  RealType *res, const std::string &label) {
  TimeClosure([mult_func, a, b, n, res](){mult_func(a, b, n, res);}, label);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " matrix_sizes" << std::endl;
    return 0;
  }
  for (int i = 1; i < argc; ++i) {
    const IndexType matrix_size = atoi(argv[i]);
    const IndexType full_array_size = matrix_size * matrix_size;
    std::unique_ptr<RealType[]> a(new RealType[full_array_size]);
    std::unique_ptr<RealType[]> b(new RealType[full_array_size]);

    FillVector(a.get(), full_array_size, 0);
    FillVector(b.get(), full_array_size, 0);

    std::unique_ptr<RealType[]> res_strassen(new RealType[full_array_size]);
    memset(res_strassen.get(), 0, full_array_size * sizeof(RealType));
    TimeMultiply(MultiplyStrassen, a.get(), b.get(), matrix_size,
                 res_strassen.get(), "MultiplyStrassen size = " +
                                         std::to_string(matrix_size) + " ");
  }
  return 0;
}
