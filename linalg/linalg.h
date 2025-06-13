#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <cudamacro.h>
#include <mallocmacro.h>

/*!
 * \brief vector-vector dot product: a.b
 * \param[in] a[0..n-1] first vector
 * \param[in] b[0..n-1] second vector
 */
HD double vvdot(double const *a, double const *b, int n);

/*!
 * \brief matrix-vector dot product: m.v
 * \param[out] r[0..n1-1] output vector
 * \param[in] m[0..n1*n2 -1] row-major sequential storage of n1 x n2 matrix
 * \param[in] v[0..n2-1] input vector
 * \param[in] n1 number of rows in matrix
 * \param[in] n2 number of columns in matrix
 */
HD void mvdot(double *r, double const *m, double const *v, int n1, int n2);

/*!
 * \brief matrix-matrix dot product: a.b
 * \param[out] r[0..n1*n3-1] output matrix in row-major sequential storage
 * \param[in] a[0..n1*n2-1] row-major sequential storage of n1 x n2 matrix
 * \param[in] b[0..n2*n3-1] row-major sequential storage of n2 x n3 matrix
 * \param[in] n1 number of rows in matrix a
 * \param[in] n2 number of columns in matrix a (and rows in matrix b)
 * \param[in] n3 number of columns in matrix b
 */
HD void mmdot(double *r, double const *a, double const *b, int n1, int n2,
              int n3);

/*!
 * \brief LU decomposition
 *
 * Given a row-major sequential storage of matrix a[0..n*n-1],
 * this routine replaces it by the LU decomposition of a rowwise permutation of
 * itself. This routine is used in combination with lubksb to solve linear
 * equationsor invert a matrix. Adapted from Numerical Recipes in C, 2nd Ed.,
 * p. 46.
 *
 * \param[in,out] a[0..n*n-1] row-major input matrix, output LU decomposition
 * \param[out] indx[0..n-1] vector that records the row permutation effected by
 * the partial pivoting. Outputs as +/- 1 depending on whether the number of row
 * interchanges was even or odd, respectively.
 * \param[in] n size of matrix
 */
#define A(i, j) a[(i) * n + (j)]
HD int ludcmp(double *a, int *indx, int n) {
  int i, imax, j, k, d;
  double big, dum, sum, temp;
  double *vv = (double *)malloc(n * sizeof(double));

  d = 1;
  for (i = 0; i < n; i++) {
    big = 0.0;
    for (j = 0; j < n; j++)
      if ((temp = fabs(A(i, j))) > big)
        big = temp;
    if (big == 0.0) {
      fprintf(stderr, "Singular matrix in routine ludcmp");
      exit(1);
    }
    vv[i] = 1.0 / big;
  }
  for (j = 0; j < n; j++) {
    for (i = 0; i < j; i++) {
      sum = A(i, j);
      for (k = 0; k < i; k++)
        sum -= A(i, k) * A(k, j);
      A(i, j) = sum;
    }
    big = 0.0;
    for (i = j; i < n; i++) {
      sum = A(i, j);
      for (k = 0; k < j; k++)
        sum -= A(i, k) * A(k, j);
      A(i, j) = sum;
      if ((dum = vv[i] * fabs(sum)) >= big) {
        big = dum;
        imax = i;
      }
    }
    if (j != imax) {
      for (k = 0; k < n; k++) {
        dum = A(imax, k);
        A(imax, k) = A(j, k);
        A(j, k) = dum;
      }
      d = -d;
      vv[imax] = vv[j];
    }
    indx[j] = imax;
    if (j != n - 1) {
      dum = (1.0 / A(j, j));
      for (i = j + 1; i < n; i++)
        A(i, j) *= dum;
    }
  }
  free(vv);

  return d;
}
#undef A

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
HD void lubksb(double *b, double const *a, int const *indx, int n) {
  int i, ii = 0, ip, j;
  double sum;

  for (i = 0; i < n; i++) {
    ip = indx[i];
    sum = b[ip];
    b[ip] = b[i];
    if (ii)
      for (j = ii - 1; j < i; j++)
        sum -= a[i * n + j] * b[j];
    else if (sum)
      ii = i + 1;
    b[i] = sum;
  }
  for (i = n - 1; i >= 0; i--) {
    sum = b[i];
    for (j = i + 1; j < n; j++)
      sum -= a[i * n + j] * b[j];
    b[i] = sum / a[i * n + i];
  }
}

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
HD void luminv(double *y, double const *a, int const *indx, int n) {
  double *col = (double *)malloc(n * sizeof(double));
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++)
      col[i] = 0.0;
    col[j] = 1.0;
    lubksb(col, a, indx, n);
    for (int i = 0; i < n; i++)
      y[i * n + j] = col[i];
  }
  free(col);
}

/*!
 * \brief solve least square problem min ||A.x - b||
 *
 * \param[in,out] b[0..n1-1] right-hand-side vector and output. Input dimension
 * is n1, output dimension is n2, requiring n1 >= n2
 * \param[in] a[0..n1*n2-1] row-major input matrix, A
 * \param[in] n1 number of rows in matrix
 * \param[in] n2 number of columns in matrix
 */
HD void leastsq(double *b, double const *a, int n1, int n2) {
  double *c = (double *)malloc(n1 * sizeof(double));
  memcpy(c, b, n1 * sizeof(double));

  double *y = (double *)malloc(n2 * n2 * sizeof(double));

  for (int i = 0; i < n2; ++i) {
    // calculate A^T.A
    for (int j = 0; j < n2; ++j) {
      y[i * n2 + j] = 0.;
      for (int k = 0; k < n1; ++k)
        y[i * n2 + j] += a[k * n2 + i] * a[k * n2 + j];
    }

    // calculate A^T.b
    b[i] = 0.;
    for (int j = 0; j < n1; ++j)
      b[i] += a[j * n2 + i] * c[j];
  }

  // calculate (A^T.A)^{-1}.(A^T.b)
  int *indx = (int *)malloc(n2 * sizeof(int));
  ludcmp(y, indx, n2);
  lubksb(b, y, indx, n2);

  free(c);
  free(indx);
  free(y);
}

void populate_aug(double *aug, double const *ata, double const *c, int n2,
                  int nact, int const *ct_indx) {
  // populate A^T.A (upper left block)
  for (int i = 0; i < n2; ++i) {
    for (int j = 0; j < n2; ++j) {
      AUG(i, j) = ATA(i, j);
    }
  }

  // populate C (lower left block)
  for (int i = 0; i < nact; ++i) {
    for (int j = 0; j < n2; ++j) {
      AUG(n2 + i, j) = C(ct_indx[i], j);
    }
  }

  // populate C^T (upper right block)
  for (int i = 0; i < n2; ++i) {
    for (int j = 0; j < nact; ++j) {
      AUG(i, n2 + j) = C(ct_indx[j], i);
    }
  }

  // zero (lower right block)
  for (int i = 0; i < nact; ++i) {
    for (int j = 0; j < nact; ++j) {
      AUG(n2 + i, n2 + j) = 0.0;
    }
  }
}

void populate_rhs(double *rhs, double const *atb, double const *d, int n2,
                  int nact, int const *ct_indx) {
  // populate A^T.b (upper part)
  for (int i = 0; i < n2; ++i) {
    rhs[i] = atb[i];
  }

  // populate d (lower part)
  for (int i = 0; i < nact; ++i) {
    rhs[n2 + i] = d[ct_indx[i]];
  }
}
/*!
 * \brief solve constrained least square problem: min ||A.x - b||, s.t. C.x <= d
 *
 * This subroutine solves the constrained least square problem using the active
 * set method based on the KKT conditions. The first `neq` rows of the
 * constraint matrix `C` are treated as equality constraints, while the
 * remaining rows are treated as inequality constraints.
 *
 * \param[in,out] b[0..n1-1] right-hand-side vector and output. Input dimension
 * is n1, output dimension is n2, requiring n1 >= n2
 * \param[in] a[0..n1*n2-1] row-major input matrix, A
 * \param[in] c[0..n3*n2-1] row-major constraint matrix, C
 * \param[in] d[0..n3-1] right-hand-side constraint vector, d
 * \param[in] n1 number of rows in matrix A
 * \param[in] n2 number of columns in matrix A
 * \param[in] n3 number of rows in matrix C
 * \param[in] neq number of equality constraints, 0 <= neq <= n3
 * \param[in,out] max_iter in: maximum number of iterations to perform, out:
 * number of iterations actually performed
 * \return 0 on success, 1 on invalid input (e.g., neq < 0 or neq > n3),
 *         2 on failure (max_iter reached without convergence).
 */
#define A(i, j) a[(i) * n2 + (j)]
#define ATA(i, j) ata[(i) * n2 + (j)]
#define AUG(i, j) aug[(i) * (n2 + nact) + (j)]
#define C(i, j) c[(i) * n2 + (j)]
HD int leastsq_kkt(double *b, double const *a, double const *c, double const *d,
                   int n1, int n2, int n3, int neq, int *max_iter) {
  // check if n1 > 0, n2 > 0, n3 >= 0
  if (n1 <= 0 || n2 <= 0 || n3 < 0 || n1 < n2) {
    fprintf(
        stderr,
        "Error: n1 and n2 must be positive integers and n3 >= 0, n1 >= n2.\n");
    return 1; // invalid input
  }

  // check if 0 <= neq <= n3
  if (neq < 0 || neq > n3) {
    fprintf(stderr, "Error: neq must be non-negative.\n");
    return 1; // invalid input
  }

  // Allocate memory for the augmented matrix and right-hand side vector
  int size = n2 + n3;
  double *aug = (double *)malloc(size * size * sizeof(double));
  double *ata = (double *)malloc(n2 * n2 * sizeof(double));
  double *atb = (double *)malloc(size * sizeof(double));
  double *rhs = (double *)malloc(size * sizeof(double));

  // evaluation of constraints
  double *eval = (double *)malloc(n3 * sizeof(double));

  // index for the active set
  int *ct_indx = (int *)malloc(n3 * sizeof(int));

  // index array for the LU decomposition
  int *lu_indx = (int *)malloc(size * sizeof(int));

  // populate A^T.A
  for (int i = 0; i < n2; ++i) {
    for (int j = 0; j < n2; ++j) {
      ATA(i, j) = 0.0;
      for (int k = 0; k < n1; ++k) {
        ATA(i, j) += A(k, i) * A(k, j);
      }
    }
  }

  // populate A^T.b
  for (int i = 0; i < n2; ++i) {
    atb[i] = 0.0;
    for (int j = 0; j < n1; ++j) {
      atb[i] += A(j, i) * b[j];
    }
  }

  for (int i = 0; i < n3; ++i) {
    ct_indx[i] = i;
  }

  int nactive = neq;
  int iter = 0;

  while (iter++ < *max_iter) {
    /*printf("============ ");
    printf("nactive = %d, iter = %d\n", nactive, iter);
    printf("CT indices = ");
    for (int i = 0; i < nactive; ++i) {
      printf("%d ", ct_indx[i]);
    }
    printf("| ");
    for (int i = nactive; i < n3; ++i) {
      printf("%d ", ct_indx[i]);
    }
    printf("\n");*/

    int nactive0 = nactive;
    populate_aug(aug, ata, c, n2, nactive, ct_indx);
    populate_rhs(rhs, atb, d, n2, nactive, ct_indx);

    // solve the KKT system
    ludcmp(aug, lu_indx, n2 + nactive);
    lubksb(rhs, aug, lu_indx, n2 + nactive);

    /*printf("Solution vector:\n");
    for (int i = 0; i < n2 + nactive; ++i) {
      printf("%f\n", rhs[i]);
    }*/

    // evaluate the inactive constraints
    for (int i = nactive; i < n3; ++i) {
      int k = ct_indx[i];
      eval[k] = 0.;
      for (int j = 0; j < n2; ++j) {
        eval[k] += C(k, j) * rhs[j];
      }
    }

    /*printf("Evaluation of inactive constraints:\n");
    for (int i = nactive; i < n3; ++i) {
      printf("%d: %f\n", ct_indx[i], eval[ct_indx[i]]);
    }*/

    // remove inactive constraints (three-way swap)
    //           mu < 0
    //           |---------------->|
    //           |<----|<----------|
    //           f     :...m       :...l
    //           |     :   |       :   |
    // | * * * | * * * * | * * * * * | x
    // |-------|---------|-----------|
    // |  EQ   |   INEQ  | INACTIVE  |
    int first = neq;
    int mid = nactive;
    int last = n3;
    while (first < mid) {
      if (rhs[n2 + first] < 0.0) { // inactive constraint
        // swap with the last active constraint
        int tmp = ct_indx[first];
        ct_indx[first] = ct_indx[mid - 1];
        ct_indx[mid - 1] = ct_indx[last - 1];
        ct_indx[last - 1] = tmp;
        --last;
        --mid;
      } else {
        ++first;
      }
    }

    /*printf("After removing inactive constraints:\n");
    printf("CT indices = ");
    for (int i = 0; i < first; ++i) {
      printf("%d ", ct_indx[i]);
    }
    printf("| ");
    for (int i = first; i < n3; ++i) {
      printf("%d ", ct_indx[i]);
    }
    printf("\n");
    printf("first = %d, mid = %d, last = %d\n", first, mid, last);*/

    // add back inactive constraints (two-way swap)
    //                     C.x <= d
    //                     |<----->|
    //                     f       : l
    //                     |       : |
    // | * * * | * * * * | * * * * * x * |
    // |-------|---------|---------------|
    // |  EQ   |   INEQ  |   INACTIVE    |
    while (first < last) {
      int k = ct_indx[first];
      if (eval[k] > d[k]) {
        // add the inactive constraint back to the active set
        ++first;
      } else {
        int tmp = ct_indx[first];
        ct_indx[first] = ct_indx[last - 1];
        ct_indx[last - 1] = tmp;
        --last;
      }
    }

    /*printf("After adding back inactive constraints:\n");
    printf("CT indices = ");
    for (int i = 0; i < first; ++i) {
      printf("%d ", ct_indx[i]);
    }
    printf("| ");
    for (int i = first; i < n3; ++i) {
      printf("%d ", ct_indx[i]);
    }
    printf("\n");
    printf("first = %d, last = %d\n", first, last);*/

    nactive = first;
    if (nactive == nactive0) {
      // no change in active set, we are done
      break;
    }
  }

  // copy to output vector b
  for (int i = 0; i < n2; ++i) {
    b[i] = rhs[i];
  }

  free(aug);
  free(ata);
  free(atb);
  free(rhs);
  free(eval);
  free(ct_indx);
  free(lu_indx);

  if (iter >= *max_iter) {
    *max_iter = iter;
    return 2; // failure to converge
  }

  *max_iter = iter;
  return 0; // success
}
#undef A
#undef ATA
#undef AUG
#undef C

#ifdef __cplusplus
} /* extern "C" */
#endif
