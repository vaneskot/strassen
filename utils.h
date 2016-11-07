// Author: Ivan Kotenkov <koteniv1@fit.cvut.cz>

#ifndef UTILS_H_
#define UTILS_H_

#include <chrono>
#include <iostream>
#include <string>

using IndexType = unsigned;
using RealType = float;

void PrintMatrix(const RealType* m, int n, int max_elements);
void FillVector(RealType* v, IndexType n, unsigned seed);

template <typename ClosureType>
void TimeClosure(ClosureType closure, const std::string& label) {
  auto start_time = std::chrono::high_resolution_clock::now();
  closure();
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::duration<RealType>>(
      end_time - start_time);
  std::cout << label << ": " << duration.count() << "s\n";
}


#endif  // UTILS_H_
