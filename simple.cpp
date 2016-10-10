// Author: Ivan Kotenkov <koteniv1.fit.cvut.cz>

void MultiplySimple(double* a, double* b, int n, double* res) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        res[i * n + j] += a[i * n + k] * b[k * n + j];
      }
    }
  }
}
