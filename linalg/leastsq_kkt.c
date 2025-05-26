#include <stdio.h>
#include <stdlib.h>

#include "linalg.h"

#define A(i, j) a[(i) * n2 + (j)]
#define ATA(i, j) ata[(i) * n2 + (j)]
#define AUG(i, j) aug[(i) * (n2 + n3) + (j)]
#define C(i, j) c[(i) * n2 + (j)]

void populate_aug(double *aug, double const* ata, double const* c,
                  int n2, int n3, int const *ct_indx) {
  // populate A^T.A (upper left block)
  for (int i = 0; i < n2; ++i) {
    for (int j = 0; j < n2; ++j) {
      AUG(i, j) = ATA(i, j);
    }
  }

  // populate C (lower left block)
  for (int i = 0; i < n3; ++i) {
    for (int j = 0; j < n2; ++j) {
      AUG(n2 + i, j) = C(ct_indx[i], j);
    }
  }

  // populate C^T (upper right block)
  for (int i = 0; i < n2; ++i) {
    for (int j = 0; j < n3; ++j) {
      AUG(i, n2 + j) = C(ct_indx[j], i);
    }
  }

  // zero (lower right block)
  for (int i = 0; i < n3; ++i) {
    for (int j = 0; j < n3; ++j) {
      AUG(n2 + i, n2 + j) = 0.0;
    }
  }
}

void populate_rhs(double *rhs, double const *atb, double const *d,
                  int n2, int n3, int const *ct_indx) {
  // populate A^T.b (upper part)
  for (int i = 0; i < n2; ++i) {
    rhs[i] = atb[i];
  }

  // populate d (lower part)
  for (int i = 0; i < n3; ++i) {
    rhs[n2 + i] = d[ct_indx[i]];
  }
}

int leastsq_kkt(double *b, double const *a, double const* c, double const* d,
                int n1, int n2, int n3, int neq, int max_iter) {
  // check if n1 > 0, n2 > 0, n3 > 0
  if (n1 <= 0 || n2 <= 0 || n3 <= 0) {
    fprintf(stderr, "Error: n1, n2, and n3 must be positive integers.\n");
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

  int nactive = n3;
  int iter = 0;

  while (iter++ < max_iter) {
    int nactive0 = nactive;
    populate_aug(aug, ata, c, n2, nactive, ct_indx);
    populate_rhs(rhs, atb, d, n2, nactive, ct_indx);

    // solve the KKT system
    ludcmp(aug, lu_indx, n2 + nactive);
    lubksb(rhs, aug, lu_indx, n2 + nactive);

    // evaluate the constraints
    for (int i = 0; i < nactive; ++i) {
      int k = ct_indx[i];
      eval[k] = 0.;
      for (int j = 0; j < n2; ++j) {
        eval[k] += C(k, j) * rhs[j];
      }
    }

    // remote inactive constraints (three-way swap)
    int first = neq;
    int mid = nactive - 1;
    int last = n3 - 1;
    while (first < mid) {
      if (rhs[n2 + first] < 0.0) { // inactive constraint
        // swap with the last active constraint
        int tmp = ct_indx[first];
        ct_indx[first] = ct_indx[mid];
        ct_indx[mid] = ct_indx[last];
        ct_indx[last] = tmp;
        --last;
        --mid;
      } else {
        ++first;
      }
    }

    // add back inactive constraints (two-way swap)
    while (first < last) {
      int k = ct_indx[first];
      if (eval[k] > d[k]) {
        // add the inactive constraint back to the active set
        int tmp = ct_indx[first];
        ct_indx[first] = ct_indx[last];
        ct_indx[last] = tmp;
        --last;
      } else {
        ++first;
      }
    }

    nactive = first;
    if (nactive == nactive0) {
      // no change in active set, we are done
      break;
    }
  }

  if (iter >= max_iter) {
    return 2; // failure to converge
  }

  free(aug);
  free(ata);
  free(atb);
  free(rhs);
  free(eval);
  free(ct_indx);
  free(lu_indx);

  return 0; // success
}

#undef A
#undef ATA
#undef AUG
#undef C
