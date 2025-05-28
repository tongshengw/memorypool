#pragma once

#include <math.h>

static inline double logsvp_ideal(double t, double beta, double gamma) {
  return (1. - 1. / t) * beta - gamma * log(t);
}

static inline double logsvp_ddT_ideal(double t, double beta, double gamma) {
  return beta / (t * t) - gamma / t;
}
