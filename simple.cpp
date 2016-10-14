// Author: Ivan Kotenkov <koteniv1.fit.cvut.cz>

#include "strassen.h"

void MultiplySimple(RealType* a, RealType* b, int n, RealType* res) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        res[i * n + j] += a[i * n + k] * b[k * n + j];
      }
    }
  }
}
