// Author: Ivan Kotenkov <koteniv1@fit.cvut.cz>

#ifndef STRASSEN_H_
#define STRASSEN_H_

using IndexType = unsigned;
void MultiplyStrassen(double* a, double* b, IndexType n, double* c);

#endif  // STRASSEN_H_
