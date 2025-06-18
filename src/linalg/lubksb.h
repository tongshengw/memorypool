/*!
 * \brief Solves the set of n linear equations A.X = B.
 *
 * This routine takes into account the possibility that b will begin with many
 * zero elements, so it is efficient for use in matrix inversion.
 * Adapted from Numerical Recipes in C, 2nd Ed., p. 47.
 *
 * \param[in,out] b[0..n-1] input as the right-hand side vector B, and returns
 * with the solution vector X.
 * \param[in] a[0..n*n-1] row-major input matrix, not as the matrix A but rather
 * as its LU decomposition, determined by the routine ludcmp.
 * \param[in] indx[0..n-1] input the permutation vector returned by ludcmp.
 * \param[in] n size of matrix
 */

template<typename T>
DISPATCH_MACRO void lubksb(T *b, T const *a, int const *indx, int n) {
  int i, ii = 0, ip, j;
  T sum;

  for (i = 0; i < n; i++) {
    ip = indx[i];
    sum = b[ip];
    b[ip] = b[i];
    if (ii)
      for (j = ii - 1; j < i; j++) sum -= a[i*n + j] * b[j];
    else if (sum)
      ii = i + 1;
    b[i] = sum;
  }
  for (i = n - 1; i >= 0; i--) {
    sum = b[i];
    for (j = i + 1; j < n; j++) sum -= a[i*n + j] * b[j];
    b[i] = sum / a[i*n + i];
  }
}
