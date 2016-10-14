// Author: Ivan Kotenkov <koteniv1@fit.cvut.cz>

#ifndef STRASSEN_H_
#define STRASSEN_H_

using IndexType = unsigned;
using RealType = float;
void MultiplyStrassen(RealType* a, RealType* b, IndexType n, RealType* c);

#endif  // STRASSEN_H_
