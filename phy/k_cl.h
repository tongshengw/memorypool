#pragma once

inline double sat_vapor_p_KCl_Lodders(double T) {
  double logp = 7.611 - 11382. / T;
  return 1.E5 * exp(logp);
}
