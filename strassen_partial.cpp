// Author: Ivan Kotenkov <koteniv1.fit.cvut.cz>

#include <assert.h>
#include <cstring>

#include <iostream>
#include <memory>

#include "strassen.h"

enum class SumType {
  SUM,
  DIFF
};

class PartialMatrix {
 public:
   PartialMatrix(double *data, IndexType full_size, IndexType i_start,
                 IndexType j_start, IndexType i_border, IndexType j_border,
                 IndexType partial_size)
       : data_(data), full_size_(full_size), i_start_(i_start),
         j_start_(j_start), i_border_(i_border), j_border_(j_border),
         partial_size_(partial_size) {}

   double Get(IndexType i, IndexType j) const {
     const IndexType actual_i = i + i_start_;
     const IndexType actual_j = j + j_start_;
     if (actual_i < i_border_ && actual_j < j_border_)
       return data_[actual_i * full_size_ + actual_j];
     return 0;
  }
  void Set(IndexType i, IndexType j, double value) {
    const IndexType actual_i = i + i_start_;
    const IndexType actual_j = j + j_start_;
    if (actual_i < i_border_ && actual_j < j_border_)
      data_[actual_i * full_size_ + actual_j] = value;
  }

  PartialMatrix GetSubmatrix(IndexType i, IndexType j) const {
    assert(0 <= i && i <= 1);
    assert(0 <= j && j <= 1);
    const IndexType block_size = partial_size_ / 2 + partial_size_ % 2;
    const IndexType new_i_start = i_start_ + i * block_size;
    IndexType new_i_border = new_i_start + block_size;
    if (new_i_border > i_border_)
      new_i_border = i_border_;
    const IndexType new_j_start = j_start_ + j * block_size;
    IndexType new_j_border = new_j_start + block_size;
    if (new_j_border > j_border_)
      new_j_border = j_border_;
    return PartialMatrix(data_, full_size_, new_i_start, new_j_start,
                         new_i_border, new_j_border, block_size);
  }

  void SetMatrix(const PartialMatrix& other) {
    for (int i = 0; i < i_border_ - i_start_; ++i) {
      for (int j = 0; j < j_border_ - j_start_; ++j) {
        data_[(i_start_ + i) * full_size_ + j_start_ + j] = other.Get(i, j);
      }
    }
  }

  void Print() const {
    for (int i = 0; i < partial_size_; ++i) {
      for (int j = 0; j < partial_size_; ++j) {
        std::cout << Get(i, j) << " ";
      }
      std::cout << std::endl;
    }
  }

 private:
  friend void MatrixSum(const PartialMatrix &left, const PartialMatrix &right,
                        SumType type, PartialMatrix *res);
  friend void MultiplyStrassen(const PartialMatrix &left,
                               const PartialMatrix &right, PartialMatrix *res);
  friend void MultiplySimple(const PartialMatrix &left,
                             const PartialMatrix &right, PartialMatrix *res);

  double *data_;
  IndexType full_size_;
  IndexType i_start_;
  IndexType j_start_;
  IndexType i_border_;
  IndexType j_border_;
  IndexType partial_size_;
};

void MatrixSum(const PartialMatrix &left, const PartialMatrix &right,
               SumType type, PartialMatrix *res) {
  const IndexType partial_size = left.partial_size_;
  assert(partial_size == right.partial_size_ &&
         partial_size == res->partial_size_);
  for (IndexType i = 0; i < partial_size; ++i) {
    for (IndexType j = 0; j < partial_size; ++j) {
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
  const IndexType partial_size = left.partial_size_;
  assert(partial_size == right.partial_size_ &&
         partial_size == res->partial_size_);

  if (partial_size == 1) {
    res->Set(0, 0, left.Get(0, 0) * right.Get(0, 0));
    return;
  }

  const IndexType block_size = partial_size / 2 + partial_size % 2;
  const PartialMatrix a11 = left.GetSubmatrix(0, 0);
  const PartialMatrix a12 = left.GetSubmatrix(0, 1);
  const PartialMatrix a21 = left.GetSubmatrix(1, 0);
  const PartialMatrix a22 = left.GetSubmatrix(1, 1);

  const PartialMatrix b11 = right.GetSubmatrix(0, 0);
  const PartialMatrix b12 = right.GetSubmatrix(0, 1);
  const PartialMatrix b21 = right.GetSubmatrix(1, 0);
  const PartialMatrix b22 = right.GetSubmatrix(1, 1);

  // FIXME(kotenkov): use res for tmp matrices ?
  const IndexType tmp_size = block_size * block_size * 8;
  std::unique_ptr<double[]> tmp_data(new double[tmp_size]);
  memset(tmp_data.get(), 0, tmp_size * sizeof(double));

  double* tmp_block[8];
  for (IndexType i = 0; i < 8; ++i) {
    tmp_block[i] = tmp_data.get() + block_size * block_size * i;
  }
  PartialMatrix m1(tmp_block[0], block_size, 0, 0, block_size, block_size,
                   block_size);
  PartialMatrix m2(tmp_block[1], block_size, 0, 0, block_size, block_size,
                   block_size);
  PartialMatrix m3(tmp_block[2], block_size, 0, 0, block_size, block_size,
                   block_size);
  PartialMatrix m4(tmp_block[3], block_size, 0, 0, block_size, block_size,
                   block_size);
  PartialMatrix m5(tmp_block[4], block_size, 0, 0, block_size, block_size,
                   block_size);
  PartialMatrix m6(tmp_block[5], block_size, 0, 0, block_size, block_size,
                   block_size);
  PartialMatrix m7(tmp_block[6], block_size, 0, 0, block_size, block_size,
                   block_size);

  PartialMatrix c11 = res->GetSubmatrix(0, 0);
  PartialMatrix c12 = res->GetSubmatrix(0, 1);
  PartialMatrix c21 = res->GetSubmatrix(1, 0);
  PartialMatrix c22 = res->GetSubmatrix(1, 1);

  // We have one full block that we can use as a temporary block, c11.
  // We need 2 blocks, so we create another here.
  PartialMatrix tmp_matrix(tmp_block[7], block_size, 0, 0, block_size,
                           block_size, block_size);

  // Compute |m1|..|m7| matrices.

  MatrixSum(a11, a22, SumType::SUM, &c11);
  MatrixSum(b11, b22, SumType::SUM, &tmp_matrix);
  MultiplyStrassen(c11, tmp_matrix, &m1);

  MatrixSum(a21, a22, SumType::SUM, &c11);
  MultiplyStrassen(c11, b11, &m2);

  MatrixSum(b12, b22, SumType::DIFF, &c11);
  MultiplyStrassen(a11, c11, &m3);

  MatrixSum(b21, b11, SumType::DIFF, &c11);
  MultiplyStrassen(a22, c11, &m4);

  MatrixSum(a11, a12, SumType::SUM, &c11);
  MultiplyStrassen(c11, b22, &m5);

  MatrixSum(a21, a11, SumType::DIFF, &c11);
  MatrixSum(b11, b12, SumType::SUM, &tmp_matrix);
  MultiplyStrassen(c11, tmp_matrix, &m6);

  MatrixSum(a12, a22, SumType::DIFF, &c11);
  MatrixSum(b21, b22, SumType::SUM, &tmp_matrix);
  MultiplyStrassen(c11, tmp_matrix, &m7);

  // Compute |c11|..|c22| matrices.

  MatrixSum(m1, m4, SumType::SUM, &c11);
  MatrixSum(c11, m5, SumType::DIFF, &c11);
  MatrixSum(c11, m7, SumType::SUM, &c11);

  MatrixSum(m3, m5, SumType::SUM, &c12);

  MatrixSum(m2, m4, SumType::SUM, &c21);

  MatrixSum(m1, m2, SumType::DIFF, &c22);
  MatrixSum(c22, m3, SumType::SUM, &c22);
  MatrixSum(c22, m6, SumType::SUM, &c22);
}

void MultiplyStrassen(double *a, double *b, IndexType n, double *c) {
  const PartialMatrix left(a, n, 0, 0, n, n, n);
  const PartialMatrix right(b, n, 0, 0, n, n, n);
  PartialMatrix res(c, n, 0, 0, n, n, n);
  MultiplyStrassen(left, right, &res);
}
