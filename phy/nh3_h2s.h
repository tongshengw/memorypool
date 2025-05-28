#pragma once

// C/C++
#include <math.h>

const double Pcgs_of_atm = 1013250.0;  // atmospheres to dynes/cm**2

inline double svp_nh3_h2s_Umich(double T) {
  double const GOLB2 = (14.83 - (4715.0 / T));
  return (pow(10.0, GOLB2)) * Pcgs_of_atm * Pcgs_of_atm;
}

static inline double logsvp_NH3_H2S_Lewis(double T) {
  return (14.82 - 4705. / T) * log(10.) + 2. * log(101325.);
}

static inline double logsvp_ddT_NH3_H2S_Lewis(double T) {
  return 4705. * log(10.) / (T * T);
}
