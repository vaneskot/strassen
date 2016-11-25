// Author: Ivan Kotenkov <koteniv1.fit.cvut.cz>

#include <assert.h>
#include <cstring>

#include <iostream>
#include <memory>

#include "strassen.h"

namespace {
const int kAdditionalBlocksCount = 3;
}  // namespace

class PartialMatrix {
 public:
  PartialMatrix(RealType *data, IndexType full_size, IndexType i_start,
                 IndexType j_start, IndexType i_border, IndexType j_border,
                 IndexType partial_size)
       : data_(data), full_size_(full_size), i_start_(i_start),
         j_start_(j_start), i_border_(i_border), j_border_(j_border),
         partial_size_(partial_size) {
    i_max_ = i_start_ < i_border_ ? i_border_ - i_start_ : 0;
    j_max_ = j_start_ < j_border_ ? j_border_ - j_start_ : 0;
  }

  RealType UnsafeGet(IndexType i, IndexType j) const {
     const IndexType actual_i = i + i_start_;
     const IndexType actual_j = j + j_start_;
     return data_[actual_i * full_size_ + actual_j];
  }

  void UnsafeSet(IndexType i, IndexType j, RealType value) {
    const IndexType actual_i = i + i_start_;
    const IndexType actual_j = j + j_start_;
    data_[actual_i * full_size_ + actual_j] = value;
  }

  RealType Get(IndexType i, IndexType j) const {
    const IndexType actual_i = i + i_start_;
    const IndexType actual_j = j + j_start_;
    if (actual_i < i_border_ && actual_j < j_border_)
      return data_[actual_i * full_size_ + actual_j];
    return 0;
  }

  void Set(IndexType i, IndexType j, RealType value) {
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
    for (int i = 0; i < i_max_; ++i) {
      for (int j = 0; j < j_max_; ++j) {
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
                        PartialMatrix *res);
  friend void MatrixDiff(const PartialMatrix &left, const PartialMatrix &right,
                         PartialMatrix *res);
  friend void MultiplyStrassen(const PartialMatrix &left,
                               const PartialMatrix &right, RealType *tmp_memory,
                               PartialMatrix *res,
                               IndexType max_recursion_size);
  friend void MultiplySimple(const PartialMatrix &left,
                             const PartialMatrix &right, PartialMatrix *res);

  RealType *data_;
  IndexType full_size_;
  IndexType i_start_;
  IndexType j_start_;
  IndexType i_border_;
  IndexType j_border_;
  IndexType i_max_;
  IndexType j_max_;
  IndexType partial_size_;
};

void MultiplySimple(const PartialMatrix &left, const PartialMatrix &right,
                    PartialMatrix *res) {
  const IndexType partial_size = left.partial_size_;
  assert(partial_size == right.partial_size_);
  assert(partial_size == res->partial_size_);
  const IndexType max_i = std::min(left.i_max_, res->i_max_);
  const IndexType max_j = std::min(right.j_max_, res->j_max_);
  const IndexType max_k = std::min(left.j_max_, right.j_max_);

  for (int i = 0; i < max_i; ++i) {
    for (int j = 0; j < max_j; ++j) {
      res->UnsafeSet(i, j, 0.);
      for (int k = 0; k < max_k; ++k) {
        res->UnsafeSet(i, j, res->Get(i, j) + left.Get(i, k) * right.Get(k, j));
      }
    }
    for (int j = max_j; j < res->j_max_; ++j)
      res->UnsafeSet(i, j, 0.);
  }
  for (int i = max_i; i < res->i_max_; ++i) {
    for (int j = 0; j < res->j_max_; ++j) {
      res->UnsafeSet(i, j, 0);
    }
  }
}

void MatrixSum(const PartialMatrix &left, const PartialMatrix &right,
               PartialMatrix *res) {
  const IndexType partial_size = left.partial_size_;
  assert(partial_size == right.partial_size_ &&
         partial_size == res->partial_size_);

  const IndexType max_i_left = std::min(left.i_max_, res->i_max_);
  const IndexType max_i_right = std::min(right.i_max_, res->i_max_);
  const IndexType max_j_left = std::min(left.j_max_, res->j_max_);
  const IndexType max_j_right = std::min(right.j_max_, res->j_max_);
  const IndexType max_i_res = res->i_max_;
  const IndexType max_j_res = res->j_max_;
  const IndexType max_i = std::min(max_i_left, max_i_right);
  const IndexType max_j = std::min(max_j_left, max_j_right);
  const IndexType max_i_left_right = std::max(max_i_left, max_i_right);
  const IndexType max_j_left_right = std::max(max_j_left, max_j_right);

  for (int i = 0; i < max_i; ++i) {
    for (int j = 0; j < max_j; ++j) {
      res->UnsafeSet(i, j, left.UnsafeGet(i, j) + right.UnsafeGet(i, j));
    }
    for (int j = max_j; j < max_j_left; ++j) {
      res->UnsafeSet(i, j, left.UnsafeGet(i, j));
    }
    for (int j = max_j; j < max_j_right; ++j) {
      res->UnsafeSet(i, j, right.UnsafeGet(i, j));
    }
    for (int j = max_j_left_right; j < max_j_res; ++j) {
      res->UnsafeSet(i, j, 0.);
    }
  }
  for (int i = max_i; i < max_i_left; ++i) {
    for (int j = 0; j < max_j_left; ++j) {
      res->UnsafeSet(i, j, left.UnsafeGet(i, j));
    }
    for (int j = max_j_left; j < max_j_res; ++j) {
      res->UnsafeSet(i, j, 0.);
    }
  }
  for (int i = max_i; i < max_i_right; ++i) {
    for (int j = 0; j < max_j_right; ++j) {
      res->UnsafeSet(i, j, right.UnsafeGet(i, j));
    }
    for (int j = max_j_right; j < max_j_res; ++j) {
      res->UnsafeSet(i, j, 0.);
    }
  }
  for (int i = max_i_left_right; i < max_i_res; ++i) {
    for (int j = 0; j < max_j_res; ++j) {
      res->UnsafeSet(i, j, 0.);
    }
  }
}

void MatrixDiff(const PartialMatrix &left, const PartialMatrix &right,
                PartialMatrix *res) {
  const IndexType partial_size = left.partial_size_;
  assert(partial_size == right.partial_size_ &&
         partial_size == res->partial_size_);

  const IndexType max_i_left = std::min(left.i_max_, res->i_max_);
  const IndexType max_i_right = std::min(right.i_max_, res->i_max_);
  const IndexType max_j_left = std::min(left.j_max_, res->j_max_);
  const IndexType max_j_right = std::min(right.j_max_, res->j_max_);
  const IndexType max_i_res = res->i_max_;
  const IndexType max_j_res = res->j_max_;
  const IndexType max_i = std::min(max_i_left, max_i_right);
  const IndexType max_j = std::min(max_j_left, max_j_right);
  const IndexType max_i_left_right = std::max(max_i_left, max_i_right);
  const IndexType max_j_left_right = std::max(max_j_left, max_j_right);

  for (int i = 0; i < max_i; ++i) {
    for (int j = 0; j < max_j; ++j) {
      res->UnsafeSet(i, j, left.UnsafeGet(i, j) - right.UnsafeGet(i, j));
    }
    for (int j = max_j; j < max_j_left; ++j) {
      res->UnsafeSet(i, j, left.UnsafeGet(i, j));
    }
    for (int j = max_j; j < max_j_right; ++j) {
      res->UnsafeSet(i, j, -right.UnsafeGet(i, j));
    }
    for (int j = max_j_left_right; j < max_j_res; ++j) {
      res->UnsafeSet(i, j, 0.);
    }
  }
  for (int i = max_i; i < max_i_left; ++i) {
    for (int j = 0; j < max_j_left; ++j) {
      res->UnsafeSet(i, j, left.UnsafeGet(i, j));
    }
    for (int j = max_j_left; j < max_j_res; ++j) {
      res->UnsafeSet(i, j, 0.);
    }
  }
  for (int i = max_i; i < max_i_right; ++i) {
    for (int j = 0; j < max_j_right; ++j) {
      res->UnsafeSet(i, j, -right.UnsafeGet(i, j));
    }
    for (int j = max_j_right; j < max_j_res; ++j) {
      res->UnsafeSet(i, j, 0.);
    }
  }
  for (int i = max_i_left_right; i < max_i_res; ++i) {
    for (int j = 0; j < max_j_res; ++j) {
      res->UnsafeSet(i, j, 0.);
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
                      RealType* tmp_memory,
                      PartialMatrix *res,
                      IndexType max_recursion_size = 8) {
  const IndexType partial_size = left.partial_size_;
  assert(partial_size == right.partial_size_ &&
         partial_size == res->partial_size_);

  if (partial_size <= max_recursion_size) {
    MultiplySimple(left, right, res);
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

  RealType* tmp_block[kAdditionalBlocksCount];
  for (IndexType i = 0; i < kAdditionalBlocksCount; ++i) {
    tmp_block[i] = tmp_memory + block_size * block_size * i;
  }
  RealType *next_tmp_memory =
      tmp_block[kAdditionalBlocksCount - 1] + block_size * block_size;

  PartialMatrix c11 = res->GetSubmatrix(0, 0);
  PartialMatrix c12 = res->GetSubmatrix(0, 1);
  PartialMatrix c21 = res->GetSubmatrix(1, 0);
  PartialMatrix c22 = res->GetSubmatrix(1, 1);

  // We have one full block that we can use as a temporary block, c11.
  // We need 2 blocks, so we create another here.
  PartialMatrix tmp_matrix(tmp_block[0], block_size, 0, 0, block_size,
                           block_size, block_size);
  PartialMatrix tmp_matrix1(tmp_block[1], block_size, 0, 0, block_size,
                            block_size, block_size);
  PartialMatrix m_matrix(tmp_block[2], block_size, 0, 0, block_size, block_size,
                         block_size);

  // Compute |m1|..|m7| matrices.

  MatrixDiff(a21, a11, &c11);
  MatrixSum(b11, b12, &tmp_matrix);
  MultiplyStrassen(c11, tmp_matrix, next_tmp_memory, &m_matrix,
                   max_recursion_size); // m6

  MatrixDiff(b12, b22, &c11);
  MultiplyStrassen(a11, c11, next_tmp_memory, &tmp_matrix,
                   max_recursion_size); // m3
  c12.SetMatrix(tmp_matrix);
  MatrixSum(m_matrix, tmp_matrix, &c22);

  MatrixSum(a21, a22, &c11);
  MultiplyStrassen(c11, b11, next_tmp_memory, &m_matrix,
                   max_recursion_size); // m2
  c21.SetMatrix(m_matrix);
  MatrixDiff(c22, m_matrix, &c22);

  MatrixSum(a11, a22, &tmp_matrix);
  MatrixSum(b11, b22, &tmp_matrix1);
  MultiplyStrassen(tmp_matrix, tmp_matrix1, next_tmp_memory, &c11,
                   max_recursion_size); // m1
  MatrixSum(c22, c11, &c22);

  MatrixDiff(a12, a22, &tmp_matrix);
  MatrixSum(b21, b22, &tmp_matrix1);
  MultiplyStrassen(tmp_matrix, tmp_matrix1, next_tmp_memory, &m_matrix,
                   max_recursion_size); // m7
  MatrixSum(c11, m_matrix, &c11);

  MatrixDiff(b21, b11, &tmp_matrix);
  MultiplyStrassen(a22, tmp_matrix, next_tmp_memory, &m_matrix,
                   max_recursion_size); // m4
  MatrixSum(c11, m_matrix, &c11);
  MatrixSum(c21, m_matrix, &c21);

  MatrixSum(a11, a12, &tmp_matrix);
  MultiplyStrassen(tmp_matrix, b22, next_tmp_memory, &m_matrix,
                   max_recursion_size); // m5
  MatrixDiff(c11, m_matrix, &c11);
  MatrixSum(c12, m_matrix, &c12);
}

void MultiplyStrassen(RealType *a, RealType *b, IndexType n, RealType *c) {
  const PartialMatrix left(a, n, 0, 0, n, n, n);
  const PartialMatrix right(b, n, 0, 0, n, n, n);
  PartialMatrix res(c, n, 0, 0, n, n, n);
  IndexType tmp_size = 0;
  for (int i = n; i > 1;) {
    i = i / 2 + i % 2;
    tmp_size += i * i;
  }
  tmp_size *=  kAdditionalBlocksCount * sizeof(RealType);
  std::unique_ptr<RealType[]> tmp_memory(new RealType[tmp_size]);
  memset(tmp_memory.get(), 0, tmp_size);
  MultiplyStrassen(left, right, tmp_memory.get(), &res);
}

void MultiplyStrassenRecursionSize(RealType *a, RealType *b, IndexType n,
                                   int max_recursion_size, RealType *c) {
  const PartialMatrix left(a, n, 0, 0, n, n, n);
  const PartialMatrix right(b, n, 0, 0, n, n, n);
  PartialMatrix res(c, n, 0, 0, n, n, n);
  IndexType tmp_size = 0;
  for (int i = n; i > 1;) {
    i = i / 2 + i % 2;
    tmp_size += i * i;
  }
  tmp_size *=  kAdditionalBlocksCount * sizeof(RealType);
  std::unique_ptr<RealType[]> tmp_memory(new RealType[tmp_size]);
  memset(tmp_memory.get(), 0, tmp_size);
  MultiplyStrassen(left, right, tmp_memory.get(), &res, max_recursion_size);
}
