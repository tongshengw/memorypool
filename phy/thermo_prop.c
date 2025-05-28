#include <stdlib.h>
#include <stdio.h>

#include <math.h>

#include "thermo.h"

double thermo_prop(
    double temp,
    double const *conc,
    int nspecies,
    double const *offset,
    double const *first_derivative,
    user_func1 const *extra)
{
  double prop = 0.;
  for (int i = 0; i < nspecies; i++) {
    if (conc[i] < 0) {
      fprintf(stderr, "Error: Negative concentration for species %d.\n", i);
      return 0; // error: negative concentration
    }
    double propi = offset[i] + first_derivative[i] * temp;
    if (extra[i]) {
      propi += extra[i](temp);
    }
    prop += conc[i] * propi;
  }
  return prop;
}
