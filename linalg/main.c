#include <stdio.h>

#include "linalg.h"

// test vvdot
void test_vvdot()
{
  printf("Testing vvdot...\n");
  double a[3] = {1.0, 2.0, 3.0};
  double b[3] = {4.0, 5.0, 6.0};
  double result = vvdot(a, b, 3);
  printf("Dot product: %f\n", result);
  printf("Expected: 32.000000\n");
  printf("\n");
}

// test mvdot
void test_mvdot()
{
  printf("Testing mvdot...\n");

  double m[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  printf("Matrix = \n");
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      printf("%f ", m[i * 3 + j]);
    }
    printf("\n");
  }

  double v[3] = {7.0, 8.0, 9.0};
  printf("Vector = \n");
  for (int i = 0; i < 3; i++) {
    printf("%f\n", v[i]);
  }

  double r[2];
  mvdot(r, m, v, 2, 3);
  printf("Matrix-vector product: %f %f\n", r[0], r[1]);
  printf("\n");
}

// test mmdot
void test_mmdot()
{
  printf("Testing mmdot...\n");
  double a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double b[6] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  double r[4];
  int n1 = 2;
  int n2 = 3;
  int n3 = 2;

  printf("Matrix A= \n");
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      printf("%f ", a[i * n2 + j]);
    }
    printf("\n");
  }

  printf("Matrix B= \n");
  for (int i = 0; i < n2; i++) {
    for (int j = 0; j < n3; j++) {
      printf("%f ", b[i * n3 + j]);
    }
    printf("\n");
  }

  mmdot(r, a, b, n1, n2, n3);
  
  printf("Matrix product R= \n");
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n3; j++) {
      printf("%f ", r[i * n3 + j]);
    }
    printf("\n");
  }
  printf("\n");
}

// test ludcmp
void test_ludcmp()
{
  printf("Testing ludcmp...\n");
  double a[4] = {1.0, 2.0, 3.0, 4.0};
  int indx[2];
  int n = 2;
  int d = ludcmp(a, indx, n);
  printf("LU decomposition: d = %d\n", d);
  printf("LU matrix:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f ", a[i * n + j]);
    }
    printf("\n");
  }
  printf("Permutation vector: ");
  for (int i = 0; i < n; i++) {
    printf("%d ", indx[i]);
  }
  printf("\n");
  printf("\n");
}

// test lubksb
void test_lubksb()
{
  printf("Testing lubksb...\n");
  double b[2] = {5.0, 11.0};
  double a[4] = {1.0, 2.0, 3.0, 4.0};
  int indx[2];
  int n = 2;

  printf("matrix A= \n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f ", a[i * n + j]);
    }
    printf("\n");
  }

  printf("vector b= \n");
  for (int i = 0; i < n; i++) {
    printf("%f\n", b[i]);
  }

  ludcmp(a, indx, n);
  lubksb(b, a, indx, n);
  printf("Solution vector x= \n");
  for (int i = 0; i < n; i++) {
    printf("%f\n", b[i]);
  }
  printf("\n");
}

// test luminv
void test_luminv()
{
  printf("Testing luminv...\n");
  double a[4] = {1.0, 2.0, 3.0, 4.0};
  int indx[2];
  int n = 2;
  double y[4];

  printf("matrix A= \n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f ", a[i * n + j]);
    }
    printf("\n");
  }

  ludcmp(a, indx, n);
  luminv(y, a, indx, n);
  printf("Inverse matrix Y= \n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f ", y[i * n + j]);
    }
    printf("\n");
  }
  printf("\n");
}

// test leastsq
void test_leastsq()
{
  printf("Testing leastsq...\n");
  double a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double b[3] = {7.0, 8.0, 9.0};
  int n1 = 3;
  int n2 = 2;

  printf("Matrix A= \n");
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      printf("%f ", a[i * n2 + j]);
    }
    printf("\n");
  }

  printf("Vector B= \n");
  for (int i = 0; i < n1; i++) {
    printf("%f\n", b[i]);
  }

  leastsq(b, a, n1, n2);
  printf("Least squares solution: \n");
  for (int i = 0; i < n2; i++) {
    printf("%f\n", b[i]);
  }
  printf("\n");
}

// test leastsq_kkt
void test_leastsq_kkt()
{
  printf("Testing leastsq_kkt...\n");
  double a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double c[4] = {7.0, 8.0, 9.0, 10.0};
  double d[2] = {11., 15.};
  double b[3] = {7.0, 8.0, 9.0};
  int n1 = 3;
  int n2 = 2;
  int n3 = 2;
  int neq = 1;
  
  printf("Matrix A= \n");
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      printf("%f ", a[i * n2 + j]);
    }
    printf("\n");
  }

  printf("Vector B= \n");
  for (int i = 0; i < n1; i++) {
    printf("%f\n", b[i]);
  }

  printf("Matrix C= \n");
  for (int i = 0; i < n3; i++) {
    for (int j = 0; j < n2; j++) {
      printf("%f ", c[i * n2 + j]);
    }
    printf("\n");
  }
  printf("neq = %d\n", neq);

  printf("Vector D= \n");
  for (int i = 0; i < n3; i++) {
    printf("%f\n", d[i]);
  }

  int max_iter = 20;
  leastsq_kkt(b, a, c, d, n1, n2, n3, neq, &max_iter);
  
  printf("Constrained least squares solution: \n");
  for (int i = 0; i < n2; i++) {
    printf("%f\n", b[i]);
  }
  printf("\n");
  printf("Number of iterations: %d\n", max_iter);
}

int main(int argc, char *argv[])
{
  test_vvdot();
  test_mvdot();
  test_mmdot();
  test_ludcmp();
  test_lubksb();
  test_luminv();
  test_leastsq();
  test_leastsq_kkt();
}
