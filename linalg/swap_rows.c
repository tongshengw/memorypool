#include "linalg.h"

void swap_rows(double *a, int ncol, int i, int j) {
  if (i == j) return; // No need to swap if indices are the same
  for (int k = 0; k < ncol; ++k) {
    double temp = a[i * ncol + k];
    a[i * ncol + k] = a[j * ncol + k];
    a[j * ncol + k] = temp;
  }
}
