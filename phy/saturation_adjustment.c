#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <linalg/linalg.h>
#include "thermo.h"

int saturation_adjustment(
    double *temp,
    double *conc,
    double h0,
    double const *stoich,
    int nspecies,
    int nreaction,
    double const *enthalpy_offset,
    double const *cp_const,
    user_func1 const *logsvp_func,
    user_func1 const *logsvp_func_ddT,
    user_func1 const *enthalpy_extra,
    user_func1 const *enthalpy_extra_ddT,
    double logsvp_eps,
    int *max_iter)
{
  const double Rgas = 8.31446; // J/(mol*K)
  
  // check positive temperature
  if (*temp <= 0) {
    fprintf(stderr, "Error: Non-positive temperature.\n");
    return 1; // error: non-positive temperature
  }

  // check non-negative concentration
  for (int i = 0; i < nspecies; i++) {
    if (conc[i] < 0) {
      fprintf(stderr, "Error: Negative concentration for species %d.\n", i);
      return 1; // error: negative concentration
    }
  }

  // check dimensions
  if (nspecies <= 0 || nreaction <= 0) {
    fprintf(stderr, "Error: nspecies and nreaction must be positive integers.\n");
    return 1; // error: invalid dimensions
  }

  // check non-negative cp
  for (int i = 0; i < nspecies; i++) {
    if (cp_const[i] < 0) {
      fprintf(stderr, "Error: Negative heat capacity for species %d.\n", i);
      return 1; // error: negative heat capacity
    }
  }

  double *enthalpy = (double*)malloc(nspecies * sizeof(double));
  double *enthalpy_ddT = (double*)malloc(nspecies * sizeof(double));
  double *logsvp = (double*)malloc(nreaction * sizeof(double));
  double *logsvp_ddT = (double*)malloc(nreaction * sizeof(double));

  // weight matrix
  double *weight = (double*)malloc(nreaction * nspecies * sizeof(double));

  // U matrix
  double *umat = (double*)malloc(nreaction * nreaction * sizeof(double));

  // right-hand-side vector
  double *rhs = (double*)malloc(nreaction * sizeof(double));

  // active set
  int *reaction_set = (int*)malloc(nreaction * sizeof(int));
  for (int i = 0; i < nreaction; i++) {
    reaction_set[i] = i;
  }

  // evaluate enthalpy and its derivative
  for (int i = 0; i < nspecies; i++) {
    enthalpy[i] = enthalpy_offset[i] + cp_const[i] * (*temp);
    if (enthalpy_extra[i]) {
      enthalpy[i] += enthalpy_extra[i](*temp);
    }
    enthalpy_ddT[i] = cp_const[i];
    if (enthalpy_extra_ddT[i]) {
      enthalpy_ddT[i] += enthalpy_extra_ddT[i](*temp);
    }
  }

  // active stoichiometric matrix
  double *stoich_active = (double*)malloc(nspecies * nreaction * sizeof(double));

  int iter = 0;
  int kkt_err = 0;
  while (iter++ < *max_iter) {
    printf("======= Iteration %d\n", iter);

    // evaluate log vapor saturation pressure and its derivative
    for (int j = 0; j < nreaction; j++) {
      double stoich_sum = 0.0;
      for (int i = 0; i < nspecies; i++)
        if (stoich[i * nreaction + j] < 0) { // reactant
          stoich_sum += (-stoich[i * nreaction + j]);
        }
      printf("stoich_sum = %f\n", stoich_sum);
      logsvp[j] = logsvp_func[j](*temp) - stoich_sum * log(Rgas * (*temp));
      logsvp_ddT[j] = logsvp_func_ddT[j](*temp) - stoich_sum / (*temp);
      printf("logsvp [%d] = %f, logsvp_ddT[%d] = %f\n", j, logsvp[j], j, logsvp_ddT[j]);
    }

    // calculate heat capacity
    double heat_capacity = 0.0;
    for (int i = 0; i < nspecies; i++) {
      heat_capacity += enthalpy_ddT[i] * conc[i];
    }

    // populate weight matrix, rhs vector and active set
    int first = 0;
    int last = nreaction;
    while (first < last) {
      int j = reaction_set[first];
      double log_conc_sum = 0.0;
      double prod = 1.0;

      // active set condition variables
      for (int i = 0; i < nspecies; i++) {
        if (stoich[i * nreaction + j] < 0) {  // reactant
          log_conc_sum += (-stoich[i * nreaction + j]) * log(conc[i]);
        } else if (stoich[i * nreaction + j] > 0) { // product
          prod *= conc[i];
        }
      }
      printf("log_conc_sum = %f, prod = %f\n", log_conc_sum, prod);

      // active set, weight matrix and rhs vector
      if ((log_conc_sum < (logsvp[j] - logsvp_eps) && prod > 0.) ||
          (log_conc_sum > (logsvp[j] + logsvp_eps))) {
        for (int i = 0; i < nspecies; i++) {
          weight[first * nspecies + i] = logsvp_ddT[j] * enthalpy[i] / heat_capacity;
          if (stoich[i * nreaction + j] < 0) {
            weight[first * nspecies + i] += (-stoich[i * nreaction + j]) / conc[i];
          }
        }
        rhs[first] = logsvp[j] - log_conc_sum;
        first++;
      } else {
        int tmp = reaction_set[first];
        reaction_set[first] = reaction_set[last - 1];
        reaction_set[last - 1] = tmp;
        last--;
      }
    }

    if (first == 0) {
      // all reactions are in equilibrium, no need to adjust saturation
      break;
    }
    printf("nactive reactions = %d\n", first);
    printf("weight matrix:\n");
    for (int j = 0; j < first; j++) {
      for (int i = 0; i < nspecies; i++) {
        printf("%f ", weight[j * nspecies + i]);
      }
      printf("\n");
    }

    // form active stoichiometric and constraint matrix
    int nactive = first;
    for (int i = 0; i < nspecies; i++)
      for (int k = 0; k < nactive; k++) {
        int j = reaction_set[k];
        stoich_active[i * nactive + k] = stoich[i * nreaction + j];
      }
    printf("active stoichiometric matrix:\n");
    for (int i = 0; i < nspecies; i++) {
      for (int j = 0; j < first; j++) {
        printf("%f ", stoich_active[i * nreaction + j]);
      }
      printf("\n");
    }

    mmdot(umat, weight, stoich_active, nactive, nspecies, nactive);
    printf("umatrix:\n");
    for (int j = 0; j < nactive; j++) {
      for (int i = 0; i < nactive; i++) {
        printf("%f ", umat[j * nactive + i]);
      }
      printf("\n");
    }

    for (int i = 0; i < nspecies; i++)
      for (int k = 0; k < nactive; k++) {
        stoich_active[i * nactive + k] *= -1;
      }
    printf("constraint matrix:\n");
    for (int i = 0; i < nspecies; i++) {
      for (int j = 0; j < nactive; j++) {
        printf("%f ", stoich_active[i * nactive + j]);
      }
      printf("\n");
    }
    printf("b vector (rhs):\n");
    for (int i = 0; i < nactive; i++) {
      printf("%f\n", rhs[i]);
    }
    printf("d vector (conc):\n");
    for (int i = 0; i < nspecies; i++) {
      printf("%f\n", conc[i]);
    }

    // solve constrained optimization problem (KKT)
    int max_kkt_iter = *max_iter;
    kkt_err = leastsq_kkt(rhs, umat, stoich_active, conc,
                          nactive, nactive, nspecies, 0, &max_kkt_iter);
    if (kkt_err != 0) break;
    printf("KKT solution:\n");
    for (int i = 0; i < nactive; i++) {
      printf("%f ", rhs[i]);
    }
    printf("\n");

    // rate -> conc
    for (int i = 0; i < nspecies; i++) {
      for (int k = 0; k < nactive; k++) {
        conc[i] -= stoich_active[i * nactive + k] * rhs[k];
      }
    }

    // print concentrations
    printf("Updated concentrations:\n");
    for (int i = 0; i < nspecies; i++) {
      printf("conc[%d] = %f\n", i, conc[i]);
    }

    // temperature iteration
    double temp0 = 0.;
    do {
      double zh = 0.;
      double zc = 0.;

      // re-evaluate enthalpy and its derivative
      for (int i = 0; i < nspecies; i++) {
        enthalpy[i] = enthalpy_offset[i] + cp_const[i] * (*temp);
        if (enthalpy_extra[i]) {
          enthalpy[i] += enthalpy_extra[i](*temp);
        }
        enthalpy_ddT[i] = cp_const[i];
        if (enthalpy_extra_ddT[i]) {
          enthalpy_ddT[i] += enthalpy_extra_ddT[i](*temp);
        }
        zh += enthalpy[i] * conc[i];
        zc += enthalpy_ddT[i] * conc[i];
      }
      printf("zh = %f, zc = %f\n", zh, zc);

      temp0 = *temp;
      (*temp) += (h0 - zh) / zc;
    } while (fabs(*temp - temp0) > 1e-4);

    printf("********** temp = %f\n", *temp);
    printf("enthalpy = [");
    for (int i = 0; i < nspecies; i++) {
      printf("%f ", enthalpy[i]);
    }
    printf("]\n");
    printf("enthalpy_ddT = [");
    for (int i = 0; i < nspecies; i++) {
      printf("%f ", enthalpy_ddT[i]);
    }
    printf("]\n");
  }
  
  free(enthalpy);
  free(enthalpy_ddT);
  free(logsvp);
  free(logsvp_ddT);
  free(weight);
  free(rhs);
  free(umat);
  free(reaction_set);
  free(stoich_active);

  if (iter >= *max_iter) {
    fprintf(stderr, "Saturation adjustment did not converge after %d iterations.\n", *max_iter);
    return 2; // failure to converge
  } else {
    *max_iter = iter;
    return kkt_err; // success or KKT error
  }
} 
