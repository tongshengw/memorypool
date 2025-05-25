#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*! 
 * \brief vector-vector dot product: a.b
 * \param[in] a[0..n-1] first vector
 * \param[in] b[0..n-1] second vector
 */
double vvdot(double const *a, double const *b, int n);

/*! 
 * \brief matrix-vector dot product: m.v
 * \param[out] r[0..n1-1] output vector
 * \param[in] m[0..n1*n2 -1] row-major sequential storage of n1 x n2 matrix
 * \param[in] v[0..n2-1] input vector
 * \param[in] n1 number of rows in matrix
 * \param[in] n2 number of columns in matrix
 */
void mvdot(double *r, double const *m, double const *v, int n1, int n2);

/*! 
 * \brief LU decomposition
 *
 * Given a row-major sequential storage of matrix a[0..n*n-1], 
 * this routine replaces it by the LU decomposition of a rowwise permutation of itself. 
 * This routine is used in combination with lubksb to solve linear equationsor invert 
 * a matrix. Adapted from Numerical Recipes in C, 2nd Ed., p. 46.
 *
 * \param[in,out] a[0..n*n-1] row-major input matrix, output LU decomposition
 * \param[out] indx[0..n-1] vector that records the row permutation effected by the
 *             partial pivoting. Outputs as +/- 1 depending on whether the number 
 *             of row interchanges was even or odd, respectively. 
 * \param[in] n size of matrix
 */
int ludcmp(double *a, int *indx, int n);

/*! 
 * \brief Solves the set of n linear equations A.X = B. 
 *
 * This routine takes into account the possibility that b will begin with many
 * zero elements, so it is efficient for use in matrix inversion.
 * Adapted from Numerical Recipes in C, 2nd Ed., p. 47.
 *
 * \param[in,out] b[0..n-1] input as the right-hand side vector B, and returns with the
 *                solution vector X.
 * \param[in] a[0..n*n-1] row-major input matrix, not as the matrix A but rather as its LU
 *            decomposition, determined by the routine ludcmp.
 * \param[in] indx[0..n-1] input the permutation vector returned by ludcmp.
 * \param[in] n size of matrix
 */
void lubksb(double *b, double const *a, int const *indx, int n);

/*! 
 * \brief invert a matrix, Y = A^{-1}
 *
 * Using the backsubstitution routines,
 * it is completely straightforward to find the inverse of a matrix
 * column by column.
 *
 * \param[out] y[0..n*n-1] row-major output matrix, Y = A^{-1}
 * \param[in] a[0..n*n-1] row-major LU decomposition matrix
 * \param[in] indx[0..n-1] the permutation vector returned by ludcmp.
 * \param[in] n size of matrix
 */
void luminv(double *y, double const *a, int const *indx, int n);

/*! 
 * \brief solve least square problem A.x = b
 *
 * \param[in,out] b[0..n1-1] right-hand-side vector and output. Input dimension is n1, 
 *                output dimension is n2, requiring n1 >= n2
 * \param[in] a[0..n1*n2-1] row-major input matrix, A
 * \param[in] n1 number of rows in matrix
 * \param[in] n2 number of columns in matrix
 */
void leastsq(double *b, double const *a, int n1, int n2);

#ifdef __cplusplus
} /* extern "C" */
#endif
