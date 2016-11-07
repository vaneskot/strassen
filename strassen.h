// Author: Ivan Kotenkov <koteniv1@fit.cvut.cz>

#ifndef STRASSEN_H_
#define STRASSEN_H_

#include "utils.h"

void MultiplyStrassen(RealType* a, RealType* b, IndexType n, RealType* c);
void MultiplyStrassenRecursionSize(RealType *a, RealType *b, IndexType n,
                                   int max_recursion_size, RealType *c);

#endif  // STRASSEN_H_
