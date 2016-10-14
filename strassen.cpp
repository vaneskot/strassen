// Author: Ivan Kotenkov <koteniv1@fit.cvut.cz>

#include <cstring>

#include <iostream>
#include <memory>

#include "strassen.h"

void PrintMatrix(const RealType* m, int n, int max_elements);
void MultiplySimple(RealType* a, RealType* b, int n, RealType* res);

void AddMatrix(RealType* a, RealType* b, IndexType n, RealType* res) {
  for (IndexType i = 0; i < n * n; ++i) {
    *(res++) = *(a++) + *(b++);
  }
}

void SubtractMatrix(RealType* a, RealType* b, IndexType n, RealType* res) {
  for (IndexType i = 0; i < n * n; ++i) {
    *(res++) = *(a++) - *(b++);
  }
}

void CopyBlock(RealType *from, IndexType from_size,
               IndexType from_i_start, IndexType from_j_start,
               IndexType block_size,
               RealType *to, IndexType to_size,
               IndexType to_i_start, IndexType to_j_start) {
  for (IndexType i = 0; i < block_size; ++i) {
    for (IndexType j = 0; j < block_size; ++j) {
      const int to_i = to_i_start + i;
      const int to_j = to_j_start + j;
      const int from_i = from_i_start + i;
      const int from_j = from_j_start + j;
      if (to_i < to_size && to_j < to_size) {
        to[to_i * to_size + to_j] = from_i < from_size && from_j < from_size
                                        ? from[from_i * from_size + from_j]
                                        : 0.;
      }
    }
  }
}

void MultiplyStrassen(RealType* a, RealType* b, IndexType n, RealType* res) {
  if (n == 1) {
    *res = *a * *b;
    return;
  }

  const IndexType block_size = n / 2 + n % 2;
  static const IndexType kTmpBlocksCount = 19;
  std::unique_ptr<RealType[]> additional_memory(
      new RealType[block_size * block_size * kTmpBlocksCount]);
  memset(additional_memory.get(), 0,
         block_size * block_size * kTmpBlocksCount * sizeof(RealType));
  RealType* memory_blocks[kTmpBlocksCount];
  for (IndexType i = 0; i < kTmpBlocksCount; ++i) {
    memory_blocks[i] = additional_memory.get() + i * block_size * block_size;
  }
  RealType* a11 = memory_blocks[0];
  RealType* a12 = memory_blocks[1];
  RealType* a21 = memory_blocks[2];
  RealType* a22 = memory_blocks[3];
  RealType* a_blocks[] = {a11, a12, a21, a22};
  RealType* b11 = memory_blocks[4];
  RealType* b12 = memory_blocks[5];
  RealType* b21 = memory_blocks[6];
  RealType* b22 = memory_blocks[7];
  RealType* b_blocks[] = {b11, b12, b21, b22};
  RealType* c11 = memory_blocks[8];
  RealType* c12 = memory_blocks[9];
  RealType* c21 = memory_blocks[10];
  RealType* c22 = memory_blocks[11];
  RealType* c_blocks[] = {c11, c12, c21, c22};
  for (int i = 0; i < 4; ++i) {
    CopyBlock(a, n, block_size * (i / 2), block_size * (i % 2), block_size,
              a_blocks[i], block_size, 0, 0);
    CopyBlock(b, n, block_size * (i / 2), block_size * (i % 2), block_size,
              b_blocks[i], block_size, 0, 0);
    CopyBlock(res, n, block_size * (i / 2), block_size * (i % 2), block_size,
              c_blocks[i], block_size, 0, 0);
  }

  RealType* m1 = memory_blocks[12];
  RealType* m2 = memory_blocks[13];
  RealType* m3 = memory_blocks[14];
  RealType* m4 = memory_blocks[15];
  RealType* m5 = memory_blocks[16];
  RealType* m6 = memory_blocks[17];
  RealType* m7 = memory_blocks[18];

  // Compute |m1|..|m7| matrices.

  AddMatrix(a11, a22, block_size, c11);
  AddMatrix(b11, b22, block_size, c12);
  MultiplyStrassen(c11, c12, block_size, m1);

  AddMatrix(a21, a22, block_size, c11);
  MultiplyStrassen(c11, b11, block_size, m2);

  SubtractMatrix(b12, b22, block_size, c11);
  MultiplyStrassen(a11, c11, block_size, m3);

  SubtractMatrix(b21, b11, block_size, c11);
  MultiplyStrassen(a22, c11, block_size, m4);

  AddMatrix(a11, a12, block_size, c11);
  MultiplyStrassen(c11, b22, block_size, m5);

  SubtractMatrix(a21, a11, block_size, c11);
  AddMatrix(b11, b12, block_size, c12);
  MultiplyStrassen(c11, c12, block_size, m6);

  SubtractMatrix(a12, a22, block_size, c11);
  AddMatrix(b21, b22, block_size, c12);
  MultiplyStrassen(c11, c12, block_size, m7);

  // Compute |c11|..|c22| matrices.

  AddMatrix(m1, m4, block_size, c11);
  SubtractMatrix(c11, m5, block_size, c11);
  AddMatrix(c11, m7, block_size, c11);

  AddMatrix(m3, m5, block_size, c12);

  AddMatrix(m2, m4, block_size, c21);

  SubtractMatrix(m1, m2, block_size, c22);
  AddMatrix(c22, m3, block_size, c22);
  AddMatrix(c22, m6, block_size, c22);

  for (int i = 0; i < 4; ++i) {
    CopyBlock(c_blocks[i], block_size, 0, 0, block_size, res, n,
              block_size * (i / 2), block_size * (i % 2));
  }
}
