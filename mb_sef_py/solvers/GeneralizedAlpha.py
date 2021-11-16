import numpy as np
from scipy.sparse.linalg import spsolve

from .SolverParameters import TimeIntegrationParameters
from ..core.TypeOfVariables import TypeOfVariables
from ..core.Model import TypeOfAnalysis, dict_of_assembly_coefs


class GeneralizedAlpha:
    def __init__(self, model, tip=TimeIntegrationParameters(), logger=None):
        self.model = model
        self.tip = tip
        self.logger = logger

        self.number_of_iterations = None

    @staticmethod
    def parameters(rho_inf, h):
        parameters = {
            'alpha_f': rho_inf / (1. + rho_inf),
            'alpha_m': (2. * rho_inf - 1.) / (rho_inf + 1.),
            'gamma': 0.5 * (3. - rho_inf) / (1. + rho_inf),
            'beta': 1. / ((1. + rho_inf) * (1. + rho_inf)),
        }
        parameters.update({
            'gamma_p': parameters['gamma'] / (parameters['beta'] * h),
            'beta_p': (1. - parameters['alpha_m']) / (parameters['beta'] * h * h * (1. - parameters['alpha_f'])),
        })
        return parameters

    def solve(self):
        self.model.initialize(TypeOfAnalysis.DYNAMIC)

        if self.logger:
            self.logger.initialize(self.model, self)

        assembly_coefs = dict_of_assembly_coefs()

        # Initial acceleration
        n_motion = self.model.get_number_of_motion_dofs()
        assembly_coefs['coef_k'] = assembly_coefs['coef_c'] = 0.
        assembly_coefs['coef_m'] = assembly_coefs['coef_b'] = 1.
        self.model.assemble_res_st(assembly_coefs, self.tip)
        st = self.model.build_iteration_matrix_from_sparse_representation()
        corr = - spsolve(st, self.model.res)
        self.model.v_dot = corr[:n_motion]
        a_n = corr[:n_motion]
        self.model.inc[n_motion:] = corr[n_motion:]
        self.model.kinematic_update([TypeOfVariables.LAGRANGE_MULTIPLIER])
        #

        if self.logger:
            self.logger.log_step(0)

        gap = GeneralizedAlpha.parameters(self.tip.rho, self.tip.h)
        number_of_steps = int(self.tip.T / self.tip.h)
        mean_number_of_iterations = 0.
        max_number_of_iterations = 0.
        assembly_coefs['coef_k'] = assembly_coefs['coef_b'] = 1.
        assembly_coefs['coef_c'] = gap['gamma_p']
        assembly_coefs['coef_m'] = gap['beta_p']

        for step in range(number_of_steps):
            self.model.advance_time_step(self.tip.h)
            print('time: ', self.model.time, '; step: ', step)

            self.model.inc[:n_motion] = self.tip.h * self.model.v[:] + \
                                        (0.5 - gap['beta']) * self.tip.h * self.tip.h * a_n[:]
            self.model.v += self.tip.h * (1. - gap['gamma']) * a_n[:]
            a_n = (gap['alpha_f'] * self.model.v_dot[:] - gap['alpha_m'] * a_n[:]) / (1. - gap['alpha_m'])
            self.model.inc[:n_motion] += gap['beta'] * self.tip.h * self.tip.h * a_n[:]
            self.model.v += gap['gamma'] * self.tip.h * a_n[:]

            self.model.v_dot *= 0.
            self.model.inc[n_motion:] *= 0.
            self.model.kinematic_update()

            self.number_of_iterations = 0
            while self.number_of_iterations < self.tip.nit_max:
                ref = self.model.assemble_res_st(assembly_coefs, self.tip)

                norm_res_forces = np.linalg.norm(self.model.res[:n_motion])
                norm_res_cons = np.linalg.norm(self.model.res[n_motion:])
                print('nit: ', self.number_of_iterations,
                      '; nres_f: ', norm_res_forces, ' / ', ref.norm_forces,
                      '; nres_c: ', norm_res_cons, ' / ', ref.norm_constraints)

                if norm_res_forces <= self.tip.tol_res_forces * (1. + ref.norm_forces) and \
                   norm_res_cons <= self.tip.tol_res_constraints * (1. + ref.norm_constraints):
                    break

                st = self.model.build_iteration_matrix_from_sparse_representation()
                corr = spsolve(st, self.model.res)
                self.model.inc -= corr[:]
                self.model.v -= gap['gamma_p'] * corr[:n_motion]
                self.model.v_dot -= gap['beta_p'] * corr[:n_motion]
                self.model.kinematic_update()

                self.number_of_iterations += 1

            a_n += (1. - gap['alpha_f'])/(1. - gap['alpha_m']) * self.model.v_dot[:]

            max_number_of_iterations = max(max_number_of_iterations, self.number_of_iterations)
            mean_number_of_iterations += self.number_of_iterations
            if self.logger:
                self.logger.log_step(step+1)

        if self.logger:
            self.logger.finalize()
        mean_number_of_iterations = mean_number_of_iterations / number_of_steps
        print('mean nit: ', mean_number_of_iterations, '; max nit: ', max_number_of_iterations)
