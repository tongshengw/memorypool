#include <stdlib.h>
#include <math.h>

#include <linalg/linalg.h>
#include "saturation_adjustment.h"

int saturation_adjustment(
    double *temp,
    double *conc,
    double h0,
    double const *stoich,
    int nspecies,
    int nreaction,
    double const *enthalpy_offset,
    double const *cp_multiplier,
    user_func1 const *logsvp_func,
    user_func1 const *logsvp_func_ddT,
    user_func1 const *enthalpy_func,
    user_func1 const *enthalpy_func_ddT,
    int *max_iter)
{
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

  // active stoichiometric matrix
  double *stoich_active = (double*)malloc(nspecies * nreaction * sizeof(double));

  int iter = 0;
  int kkt_err = 0;
  while (iter++ < *max_iter) {
    // temperature iteration
    double temp0;
    do {
      double zh = 0.;
      double zc = 0.;
      // evaluate enthalpy and its derivative
      for (int i = 0; i < nspecies; i++) {
        enthalpy[i] = enthalpy_offset[i];
        if (enthalpy_func[i] != NULL) {
          enthalpy[i] += enthalpy_func[i](*temp);
        }
        enthalpy_ddT[i] = cp_multiplier[i];
        if (enthalpy_func_ddT[i] != NULL) {
          enthalpy_ddT[i] *= enthalpy_func_ddT[i](*temp);
        }
        zh += enthalpy[i] * conc[i];
        zc += enthalpy_ddT[i] * conc[i];
      }

      temp0 = *temp;
      (*temp) += (h0 - zh) / zc;
    } while (fabs(*temp - temp0) > 1e-4);

    // evaluate log vapor saturation pressure and its derivative
    for (int i = 0; i < nreaction; i++) {
      logsvp[i] = logsvp_func[i](*temp);
      logsvp_ddT[i] = logsvp_func_ddT[i](*temp);
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

      // active set, weight matrix and rhs vector
      if ((log_conc_sum < logsvp[j] && prod > 0.) ||
          (log_conc_sum > logsvp[j])) {
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

    // form active stoichiometric and constraint matrix
    int nactive = first;
    for (int i = 0; i < nspecies; i++)
      for (int k = 0; k < nactive; k++) {
        int j = reaction_set[k];
        stoich_active[i * nactive + k] = stoich[i * nreaction + j];
      }

    mmdot(umat, weight, stoich_active, nactive, nspecies, nactive);

    for (int i = 0; i < nspecies; i++)
      for (int k = 0; k < nactive; k++) {
        stoich_active[i * nactive + k] *= -1;
      }

    // solve constrained optimization problem (KKT)
    kkt_err = leastsq_kkt(rhs, umat, stoich_active, conc,
                          nactive, nactive, nspecies, 0, max_iter);
    if (kkt_err != 0) break;

    // rate -> conc
    for (int i = 0; i < nspecies; i++) {
      for (int k = 0; k < nactive; k++) {
        conc[i] -= stoich_active[i * nactive + k] * rhs[k];
      }
    }
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
    return 2; // failure to converge
  } else {
    return kkt_err; // success or KKT error
  }
} 
