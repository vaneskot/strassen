// Author: Ivan Kotenkov <koteniv1.fit.cvut.cz>

#include <assert.h>
#include <memory>

enum class SumType {
  SUM,
  DIFF
};

class PartialMatrix {
 public:
  PartialMatrix(double *data, int full_size, int i_start, int j_start,
                int partial_size)
      : data_(data), full_size_(full_size), i_start_(i_start),
        j_start_(j_start), partial_size_(partial_size) {}

  double Get(int i, int j) const {
    return data_[(i + i_start_) * full_size_ + j + j_start_];
  }
  void Set(int i, int j, double value) {
    data_[(i + i_start_) * full_size_ + j + j_start_] = value;
  }

  PartialMatrix GetSubmatrix(int i, int j) const {
    assert(0 <= i && i <= 1);
    assert(0 <= j && j <= 1);
    const int block_size = partial_size_ / 2;
    return PartialMatrix(data_, full_size_, i_start_ + i * block_size,
                         j_start_ + j * block_size, block_size);
  }

 private:
  friend void MatrixSum(const PartialMatrix &left, const PartialMatrix &right,
                        SumType type, PartialMatrix *res);
  friend void MultiplyStrassen(const PartialMatrix &left,
                               const PartialMatrix &right, PartialMatrix *res);
  friend void MultiplySimple(const PartialMatrix &left,
                             const PartialMatrix &right, PartialMatrix *res);

  double *data_;
  int full_size_;
  int i_start_;
  int j_start_;
  int partial_size_;
};

void MatrixSum(const PartialMatrix &left, const PartialMatrix &right,
               SumType type, PartialMatrix *res) {
  const int partial_size = left.partial_size_;
  assert(partial_size == right.partial_size_ &&
         partial_size == res->partial_size_);
  for (int i = 0; i < partial_size; ++i) {
    for (int j = 0; j < partial_size; ++j) {
      double l = left.Get(i, j);
      double r = right.Get(i, j);
      res->Set(i, j, type == SumType::SUM ? l + r : l - r);
    }
  }
}

// C11 C12 C21 C22
// M1  M3  M2  M1
// M4  M5  M4  -M2
// -M5         M3
// M7          M6

// M1  M2  M3  M4  M5  M6  M7
// C11 C21 C12 C11 C11 C22 C11
// C22 C22 C22 C21 C12

void MultiplyStrassen(const PartialMatrix &left, const PartialMatrix &right,
                      PartialMatrix *res) {
  const int partial_size = left.partial_size_;
  assert(partial_size == right.partial_size_ &&
         partial_size == res->partial_size_);

  if (partial_size == 1) {
    res->Set(0, 0, left.Get(0, 0) * right.Get(0, 0));
    return;
  }

  assert(partial_size % 2 == 0);

  int block_size = partial_size / 2;
  const PartialMatrix a11 = left.GetSubmatrix(0, 0);
  const PartialMatrix a12 = left.GetSubmatrix(0, 1);
  const PartialMatrix a21 = left.GetSubmatrix(1, 0);
  const PartialMatrix a22 = left.GetSubmatrix(1, 1);

  const PartialMatrix b11 = right.GetSubmatrix(0, 0);
  const PartialMatrix b12 = right.GetSubmatrix(0, 1);
  const PartialMatrix b21 = right.GetSubmatrix(1, 0);
  const PartialMatrix b22 = right.GetSubmatrix(1, 1);

  // FIXME(kotenkov): use res for tmp matrices ?
  int tmp_size = block_size * block_size * 7;
  std::unique_ptr<double[]> tmp(new double[tmp_size]);
  memset(tmp.get(), 0, tmp_size * sizeof(double));

  double *m[7];
  for (int i = 0; i < 7; ++i) {
    m[i] = tmp.get() + block_size * block_size * i;
  }
  PartialMatrix m1(m[0], block_size, 0, 0, block_size);
  PartialMatrix m2(m[1], block_size, 0, 0, block_size);
  PartialMatrix m3(m[2], block_size, 0, 0, block_size);
  PartialMatrix m4(m[3], block_size, 0, 0, block_size);
  PartialMatrix m5(m[4], block_size, 0, 0, block_size);
  PartialMatrix m6(m[5], block_size, 0, 0, block_size);
  PartialMatrix m7(m[6], block_size, 0, 0, block_size);

  PartialMatrix c11 = res->GetSubmatrix(0, 0);
  PartialMatrix c12 = res->GetSubmatrix(0, 1);
  PartialMatrix c21 = res->GetSubmatrix(1, 0);
  PartialMatrix c22 = res->GetSubmatrix(1, 1);

  MatrixSum(a11, a22, SumType::SUM, &c11);
  MatrixSum(b11, b22, SumType::SUM, &c12);
  MultiplyStrassen(c11, c12, &m1);

  MatrixSum(a21, a22, SumType::SUM, &c11);
  MultiplyStrassen(c11, b11, &m2);

  MatrixSum(b12, b22, SumType::DIFF, &c11);
  MultiplyStrassen(a11, c11, &m3);

  MatrixSum(b21, b11, SumType::DIFF, &c11);
  MultiplyStrassen(a22, c11, &m4);

  MatrixSum(a11, a12, SumType::SUM, &c11);
  MultiplyStrassen(c11, b22, &m5);

  MatrixSum(a21, a11, SumType::DIFF, &c11);
  MatrixSum(b11, b12, SumType::SUM, &c12);
  MultiplyStrassen(c11, c12, &m6);

  MatrixSum(a12, a22, SumType::DIFF, &c11);
  MatrixSum(b21, b22, SumType::SUM, &c12);
  MultiplyStrassen(c11, c12, &m7);

  MatrixSum(m1, m4, SumType::SUM, &c11);
  MatrixSum(c11, m5, SumType::DIFF, &c11);
  MatrixSum(c11, m7, SumType::SUM, &c11);

  MatrixSum(m3, m5, SumType::SUM, &c12);

  MatrixSum(m2, m4, SumType::SUM, &c21);

  MatrixSum(m1, m2, SumType::DIFF, &c22);
  MatrixSum(c22, m3, SumType::SUM, &c22);
  MatrixSum(c22, m6, SumType::SUM, &c22);
}

void MultiplyStrassen(double *a, double *b, int n, double *c) {
  const PartialMatrix left(a, n, 0, 0, n);
  const PartialMatrix right(b, n, 0, 0, n);
  PartialMatrix res(c, n, 0, 0, n);
  MultiplyStrassen(left, right, &res);
}
