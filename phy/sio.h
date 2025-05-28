#pragma once

#include <math.h>

static inline double logsvp_SiO_Visscher(double T) {
  auto log10p = 8.203 - 25898.9 / T;
  return log(1.E5) + log(10.) * log10p;
}

static inline double logsvp_ddT_SiO_Visscher(double T) {
  return 25898.9 / (T * T);
}
