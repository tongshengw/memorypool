#include <stdio.h>
#include <stdlib.h>

#include "linalg.h"

#define A(i, j) a[(i) * n2 + (j)]
#define AUG(i, j) aug[(i) * size + (j)]
#define C(i, j) c[(i) * n2 + (j)]

void populate_aug1(double *aug, double const *a, int n1, int n2) {
  // populate A^T.A (upper left block)
  for (int i = 0; i < n2; ++i) {
    for (int j = 0; j < n2; ++j) {
      AUG(i, j) = 0.0;
      for (int k = 0; k < n1; ++k) {
        AUG(i, j) += A(k, i) * A(k, j);
      }
    }
  }
}

void populate_aug2(double *aug, double const* c, int n2, int n3) {
  // populate C (lower left block)
  for (int i = 0; i < n3; ++i) {
    for (int j = 0; j < n2; ++j) {
      AUG(n2 + i, j) = C(i, j);
    }
  }

  // populate C^T (upper right block)
  for (int i = 0; i < n2; ++i) {
    for (int j = 0; j < n3; ++j) {
      AUG(i, n2 + j) = C(j, i);
    }
  }

  // zero (lower right block)
  for (int i = 0; i < n3; ++i) {
    for (int j = 0; j < n3; ++j) {
      AUG(n2 + i, n2 + j) = 0.0;
    }
  }
}

void populate_rhs1(double *rhs, double const *a, double const *b
                   int n1, int n2) {
  // populate A^T.b (upper part)
  for (int i = 0; i < n2; ++i) {
    rhs[i] = 0.0;
    for (int j = 0; j < n1; ++j) {
      rhs[i] += A(j, i) * b[j];
    }
  }

void populate_rhs2(double *rhs, double const *d, int n3) {
  // populate d (lower part)
  for (int i = 0; i < n3; ++i) {
    rhs[n2 + i] = d[i];
  }
}

int leastsq_kkt(double *b, double const *a, double const* c, double const* d,
                int n1, int n2, int n3, int neq, int max_iter) {
  // check if 0 <= neq <= n3
  if (neq < 0 || neq > n3) {
    fprintf(stderr, "Error: neq must be non-negative.\n");
    exit(EXIT_FAILURE);
  }

  // Allocate memory for the augmented matrix and right-hand side vector
  int size = n2 + n3;
  double *aug = (double *)malloc(size * size * sizeof(double));
  double *rhs = (double *)malloc(size * sizeof(double));

  // evaluation of constraints
  double *eval = (double *)malloc(n3 * sizeof(double));

  // index for the active set
  int *ct_indx = (int *)malloc(n3 * sizeof(int));
  for (int i = 0; i < n3; ++i) {
    ct_indx[i] = i;
  }

  // index array for the LU decomposition
  int *lu_indx = (int *)malloc(size * sizeof(int));

  populate_aug1(aug, a, n1, n2);
  populate_rhs1(rhs, a, b, n1, n2);

  int nactive = n3;
  int iter = 0;

  while (iter++ < max_iter) {
    int nactive0 = nactive;
    populate_aug2(aug, c, n2, nactive);
    populate_rhs2(rhs, d, nactive);

    // solve the KKT system
    ludcmp(aug, lu_indx, n2 + nactive);
    lubksb(rhs, aug, lu_indx, n2 + nactive);

    // evaluate the constraints
    mvdoc(eval, c, rhs, nactive, n2);

    // determine active constraints
    int first = neq;
    int last = n3 - 1;
    while (first < last) {
      if (rhs[n2 + first] < 0.0) { // inactive constraint
        // swap with the last active constraint
        int tmp = ct_indx[first];
        ct_indx[first] = ct_indx[last];
        ct_indx[last] = tmp;

        // swap rows in constraint matrix
        swap_rows(c, n2, first, last);
        swap_rows(d, 1, first, last);
        --last;
      } else {
        ++first;
      }
    }

    // add back inactive constraints
    first = nactive;
    last = n3 - 1;
    while (first < last) {
      int j = ct_indx[first];
      if (eval[j] > d[j]) {
        // add the inactive constraint back to the active set
        int tmp = ct_indx[first];
        ct_indx[first] = ct_indx[last];
        ct_indx[last] = tmp;
        --last;

        // swap rows in constraint matrix
        swap_rows(c, n2, first, last);
        swap_rows(d, 1, first, last);
      } else {
        ++first;
      }
    }

    if (nactive == nactive0) {
      // no change in active set, we are done
      break;
    }
  }

  if (iter >= max_iter) {
    return 1; // indicate failure to converge
  }

  free(aug);
  free(rhs);
  free(eval);
  free(ct_indx);
  free(lu_indx);
}

#undef A
#undef AUG
