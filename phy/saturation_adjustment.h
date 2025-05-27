#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef double (*user_func1)(double temp);

/*! \brief Calculate the saturation adjustment
 * 
 * Given an initial guess of temperature and concentrations, this function adjusts
 * the temperature and concentrations to satisfy the saturation condition.
 *
 * \param[in,out] temp in: initial temperature, out: adjusted temperature.
 * \param[in,out] conc in: initial concentrations for each species, out: adjusted concentrations.
 * \param[in] h0 initial enthalpy.
 * \param[in] stoich reaction stoichiometric matrix, nspecies x nreaction.
 * \param[in] nspecies number of species in the system.
 * \param[in] nreaction number of reactions in the system.
 * \param[in] enthalpy_offset offset for enthalpy calculations.
 * \param[in] cp_multiplier multiplier for heat capacity calculations.
 * \param[in] logsvp_func user-defined functions for logarithm of saturation vapor pressure.
 * \param[in] logsvp_func_ddT user-defined functions for derivative of logsvp 
 *            with respect to temperature.
 * \param[in] enthalpy_func user-defined functions for enthalpy calculation 
 *            in addition to the offset.
 * \param[in] enthalpy_func_ddT user-defined functions for enthalpy derivative 
 *            with respect to temperature to be multiplied by cp_multiplier.
 * \param[in,out] max_iter maximum number of iterations allowed for convergence.
 */
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
    int *max_iter);

#ifdef __cplusplus
} /* extern "C" */
#endif
