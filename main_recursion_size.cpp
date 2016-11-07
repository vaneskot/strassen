// Author: Ivan Kotenkov <koteniv1@fit.cvut.cz>

#include <memory>
#include <string>

#include "strassen.h"

template <typename MultFuncType>
void TimeMultiply(MultFuncType mult_func, RealType *a, RealType *b, IndexType n,
                  IndexType max_recursion_size, RealType *res,
                  const std::string &label) {
  TimeClosure([mult_func, a, b, n, max_recursion_size,
               res]() { mult_func(a, b, n, max_recursion_size, res); },
              label);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " matrix_size" << std::endl;
    return 0;
  }

  const IndexType matrix_size = atoi(argv[1]);
  const IndexType full_array_size = matrix_size * matrix_size;
  std::unique_ptr<RealType[]> a(new RealType[full_array_size]);
  std::unique_ptr<RealType[]> b(new RealType[full_array_size]);

  FillVector(a.get(), full_array_size, 0);
  FillVector(b.get(), full_array_size, 0);

  for (int max_recursion_size = 1; max_recursion_size <= matrix_size / 2;
       max_recursion_size *= 2) {
    std::unique_ptr<RealType[]> res_strassen(new RealType[full_array_size]);
    memset(res_strassen.get(), 0, full_array_size * sizeof(RealType));
    TimeMultiply(MultiplyStrassenRecursionSize, a.get(), b.get(), matrix_size,
                 max_recursion_size, res_strassen.get(),
                 "MultiplyStrassen size = " + std::to_string(matrix_size) +
                     " recursion size = " + std::to_string(max_recursion_size) +
                     " ");
  }
  return 0;
}
