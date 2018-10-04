import copy
import itertools as it
import logging
import numpy as np

from beluga.bvpsol.algorithms.BaseAlgorithm import BaseAlgorithm
from beluga.ivpsol import Propagator, integrate_quads, Trajectory, reconstruct
from multiprocessing_on_dill import pool

import dill


class Shooting(BaseAlgorithm):
    r"""
    Shooting algorithm for solving boundary value problems.

    Given a system of ordinary differential equations :eq:`ordinarydifferentialequation`, define the sensitivities as

    .. math::
        A(t) = \left[\frac{\partial \mathbf{f}}{\partial \mathbf{x}}, \frac{\partial \mathbf{f}}{\partial \mathbf{p}}\right]

    Then, the state-transition matrix is defined as the following set of first-order differential equations

    .. math::
        \begin{aligned}
            \Delta_0 &= \left[Id_M, \mathbf{0}\right] \\
            \dot{\Delta} &= A(t)\Delta
        \end{aligned}

    Sensitivities of the boundary conditions are

    .. math::
        \begin{aligned}
            M &= \frac{\partial \mathbf{\Phi}}{\partial \mathbf{x}_0} \\
            P &= \frac{\partial \mathbf{\Phi}}{\partial \mathbf{p}} \\
            Q_0 &= \frac{\partial \mathbf{\Phi}}{\partial \mathbf{q}_0} \\
            Q_f &= \frac{\partial \mathbf{\Phi}}{\partial \mathbf{q}_f}
        \end{aligned}

    The Jacobian matrix is then the concatenation of these sensitivities

    .. math::
        J = \left[M, P, Q_0+Q_f \right]

    +------------------------+-----------------+-----------------+
    | Valid kwargs           | Default Value   | Valid Values    |
    +========================+=================+=================+
    | ivp_args               | {}              | see `ivpsol`    |
    +------------------------+-----------------+-----------------+
    | tolerance              | 1e-4            | > 0             |
    +------------------------+-----------------+-----------------+
    | max_error              | 100             | > 0             |
    +------------------------+-----------------+-----------------+
    | max_iterations         | 100             | > 0             |
    +------------------------+-----------------+-----------------+
    | num_arcs               | 1               | > 0             |
    +------------------------+-----------------+-----------------+
    | num_cpus               | 1               | > 0             |
    +------------------------+-----------------+-----------------+

    """

    def __new__(cls, *args, **kwargs):
        obj = super(Shooting, cls).__new__(cls, *args, **kwargs)

        obj.ivp_args = kwargs.get('ivp_args', dict())
        obj.tolerance = kwargs.get('tolerance', 1e-4)
        obj.max_error = kwargs.get('max_error', 100)
        obj.max_iterations = kwargs.get('max_iterations', 100)
        obj.num_arcs = kwargs.get('num_arcs', 1)
        obj.num_cpus = kwargs.get('num_cpus', 1)

        obj.pool = None
        obj.stm_ode_func = None
        obj.bc_func_ms = None

        return obj

    def __init__(self, *args, **kwargs):
        # Set up the boundary condition function
        if self.boundarycondition_function is not None:
            self.bc_func_ms = self._bc_func_multiple_shooting(bc_func=self.boundarycondition_function)

        if self.num_cpus > 1:
            self.pool = pool.Pool(processes=self.num_cpus)

    @staticmethod
    def _bc_jac_multi(t_list, nBCs, nquads, phi_full_list, y_list, q_list, parameters, nondynamical_params, aux, quad_func, bc_func, StepSize=1e-6):
        parameters = np.array(parameters)
        h = StepSize
        t0 = t_list[0][0]
        tf = t_list[-1][-1]
        y0 = np.array([traj[0] for traj in y_list])[0]
        yf = np.array([traj[-1] for traj in y_list])[0]
        if nquads > 0:
            q0 = q_list[0][0]
            qf = q_list[-1][-1]
        else:
            q0 = []
            qf = []
        nOdes = y0.shape[0]
        num_arcs = len(phi_full_list)

        fx = bc_func(t0, y0, q0, tf, yf, qf, parameters, nondynamical_params, aux)
        nBCs = len(fx)

        M = np.zeros((nBCs, nOdes))
        P1 = np.zeros((nBCs, parameters.size))
        P2 = np.zeros((nBCs, nondynamical_params.size))
        Ptemp = np.zeros((nBCs, parameters.size))
        J = np.zeros((nBCs, (nOdes)*num_arcs + parameters.size + nondynamical_params.size))
        dx = np.zeros((nOdes + parameters.size, num_arcs))

        for arc_idx, phi in zip(it.count(), phi_full_list):
            for ii in range(nOdes):
                dx[ii, arc_idx] = dx[ii, arc_idx] + h
                dy = np.dot(phi, dx)
                gamma = Trajectory(np.hstack(t_list), np.vstack(y_list) + dy[:,:,arc_idx])
                if nquads > 0:
                    gamma = reconstruct(quad_func, gamma, q0, parameters, aux, (0,))
                    f = bc_func(t0, gamma.y[0], gamma.q[0], tf, gamma.y[-1], gamma.q[-1], parameters, nondynamical_params, aux)
                else:
                    f = bc_func(t0, gamma.y[0], [], tf, gamma.y[-1], [], parameters, nondynamical_params, aux)
                M[:, ii] = (f-fx)/h
                dx[ii, arc_idx] = dx[ii, arc_idx] - h
            J_i = M
            J_slice = slice(nOdes*arc_idx, nOdes*(arc_idx+1))
            J[:,J_slice] = J_i

        for arc_idx, phi in zip(it.count(), phi_full_list):
            for ii in range(parameters.size):
                parameters[ii] = parameters[ii] + h
                jj = ii + nOdes
                dx[jj, arc_idx] = dx[jj, arc_idx] + h
                dy = np.dot(phi, dx)
                gamma = Trajectory(np.hstack(t_list), np.vstack(y_list) + dy[:, :, arc_idx])
                if nquads > 0:
                    gamma = reconstruct(quad_func, gamma, q0, parameters, aux, (0,))
                    f = bc_func(t0, gamma.y[0], gamma.q[0], tf, gamma.y[-1], gamma.q[-1], parameters, nondynamical_params, aux)
                else:
                    f = bc_func(t0, gamma.y[0], [], tf, gamma.y[-1], [], parameters, nondynamical_params, aux)
                Ptemp[:, ii] = (f-fx)/h
                dx[jj, arc_idx] = dx[jj, arc_idx] - h
                parameters[ii] = parameters[ii] - h
            P1 += Ptemp
            Ptemp = np.zeros((nBCs, parameters.size))

        Ptemp = np.zeros((nBCs, nondynamical_params.size))
        for ii in range(nondynamical_params.size):
            nondynamical_params[ii] = nondynamical_params[ii] + h
            if nquads > 0:
                f = bc_func(t0, y_list[0][0,:], q_list[0][0,:], tf, y_list[-1][-1,:], q_list[-1][-1,:], parameters, nondynamical_params, aux)
            else:
                f = bc_func(t0, y_list[0][0,:], [], tf, y_list[-1][-1,:], [], parameters, nondynamical_params, aux)
            Ptemp[:, ii] = (f-fx)/h
            nondynamical_params[ii] = nondynamical_params[ii] - h
        P2 += Ptemp
        J_i = np.hstack((P1, P2))
        J_slice = slice(nOdes * num_arcs, nOdes * num_arcs + (parameters.size + nondynamical_params.size))
        J[:, J_slice] = J_i
        return J

    def _bc_func_multiple_shooting(self, bc_func=None):
        def _bc_func(t0, y0, q0, tf, yf, qf, paramGuess, nondynamical_parameters, aux):
            bc1 = np.array(bc_func(t0, y0, q0, tf, yf, qf, paramGuess, nondynamical_parameters, aux)).flatten()
            # narcs = y0.shape[1]
            # bc2 = np.array([y0[:, ii + 1] - yf[:, ii] for ii in range(narcs - 1)]).flatten()
            # bc = np.hstack((bc1,bc2))
            return bc1

        def wrapper(t0, y0, q0, tf, yf, qf, paramGuess, nondynamical_parameters, aux):
            return _bc_func(t0, y0, q0, tf, yf, qf, paramGuess, nondynamical_parameters, aux)

        return wrapper

    @staticmethod
    def make_stmode(odefn, nOdes, StepSize=1e-6):
        Xh = np.eye(nOdes)*StepSize

        def _stmode_fd(t, _X, p, const, arc_idx):
            """ Finite difference version of state transition matrix """
            nParams = p.size
            F = np.empty((nOdes, nOdes+nParams))
            phi = _X[nOdes:].reshape((nOdes, nOdes+nParams))
            X = _X[0:nOdes]  # Just states

            # Compute Jacobian matrix, F using finite difference
            fx = np.squeeze([odefn(t, X, p, const, arc_idx)])

            for i in range(nOdes):
                fxh = odefn(t, X + Xh[i, :], p, const, arc_idx)
                F[:, i] = (fxh-fx)/StepSize

            for i in range(nParams):
                p[i] += StepSize
                fxh = odefn(t, X, p, const, arc_idx)
                F[:, i+nOdes] = (fxh - fx).real / StepSize
                p[i] -= StepSize

            phiDot = np.dot(np.vstack((F, np.zeros((nParams, nParams + nOdes)))), np.vstack((phi, np.hstack((np.zeros((nParams, nOdes)), np.eye(nParams))))))[:nOdes, :]
            return np.hstack((fx, np.reshape(phiDot, (nOdes * (nOdes + nParams)))))

        return _stmode_fd

    def compute_y_and_stm(self, tspan_list, ya, q0, stm0, p, aux, n_odes):
        phi_full_list = []
        t_list = []
        y_list = []
        q_list = []

        nquads = q0.size
        y0stm = np.zeros((len(stm0) + n_odes))
        yb = np.zeros_like(ya)

        for arc_idx, tspan in enumerate(tspan_list):
            y0stm[:n_odes] = ya[arc_idx, :]
            y0stm[n_odes:] = stm0[:]
            q0 = []
            sol_ivp = self.prop(self.stm_ode_func, self.quadrature_function, tspan, y0stm, q0, p, aux, 0)  # TODO: arc_idx is hardcoded as 0 here, this'll change with path constraints. I51
            t = sol_ivp.t
            yy = sol_ivp.y
            qq = sol_ivp.q
            t_list.append(t)
            y_list.append(yy[:, :n_odes])
            if nquads > 0:
                q_list.append(qq[:, :nquads])
            yb[arc_idx, :] = yy[-1, :n_odes]
            phi_full = np.reshape(yy[:, n_odes:], (len(t), n_odes, n_odes + len(p)))
            phi_full_list.append(np.copy(phi_full))

        return t_list, y_list, q_list, phi_full_list, ya, yb

    def compute_y(self, tspan_list, ya, q0, p, aux):
        t_list = []
        y_list = []
        q_list = []
        nquads = q0.size
        yb = np.zeros_like(ya)

        for arc_idx, tspan in enumerate(tspan_list):
            y0 = ya[arc_idx, :]
            sol_ivp = self.prop(self.derivative_function, self.quadrature_function, tspan, y0, q0, p, aux, 0)
            t = sol_ivp.t
            yy = sol_ivp.y
            qq = sol_ivp.q
            t_list.append(t)
            y_list.append(yy[:, :])
            if nquads > 0:
                q_list.append(qq[:, :nquads])
            yb[arc_idx, :] = yy[-1, :]

        return t_list, y_list, q_list, ya, yb

    @staticmethod
    def shoot_single(data_in):

        _prop = dill.loads(data_in[0])
        _f = dill.loads(data_in[1])

        return _prop(_f, *data_in[2:])

    def compute_y_parallel(self, tspan_list, ya, p, aux):

        data_in = []
        for tspan, y0 in zip(tspan_list, ya):
            data_in += [(self.prop_byte, self.deriv_func_byte, None, tspan, y0, [], p, aux, 0)]

        sols_ivp = list(self.pool.map(self.shoot_single, data_in))

        t_list = []
        y_list = []
        yb = []
        for sol_ivp in sols_ivp:
            t = sol_ivp.t
            yy = sol_ivp.y
            y_list.append(yy[:, :])
            t_list.append(t)
            yb += [yy[-1, :]]

        return t_list, y_list, ya, yb

    def solve(self, solinit):
        """
        Solve a two-point boundary value problem using the shooting method.

        :param deriv_func: The ODE function.
        :param quad_func: The quad func.
        :param bc_func: The boundary conditions function.
        :param solinit: An initial guess for a solution to the BVP.
        :return: A solution to the BVP.
        """

        # Make a copy of sol and format inputs
        sol = copy.deepcopy(solinit)
        sol.t = np.array(sol.t, dtype=np.float64)
        sol.y = np.array(sol.y, dtype=np.float64)
        sol.q = np.array(sol.q, dtype=np.float64)
        sol.dynamical_parameters = np.array(sol.dynamical_parameters, dtype=np.float64)
        sol.nondynamical_parameters = np.array(sol.nondynamical_parameters, dtype=np.float64)

        # Extract some info from the guess structure
        y0g = sol.y[0, :]
        if self.quadrature_function is None or all(np.isnan(sol.q)):
            q0g = np.array([])
        else:
            q0g = sol.q[0, :]

        nOdes = y0g.shape[0]
        nquads = q0g.shape[0]
        paramGuess = sol.dynamical_parameters
        nondynamical_parameter_guess = sol.nondynamical_parameters

        # Make the state-transition ode matrix
        if self.stm_ode_func is None:
            self.stm_ode_func = self.make_stmode(self.derivative_function, y0g.shape[0])

        # Set up the boundary condition function
        if self.bc_func_ms is None:
            self.bc_func_ms = self._bc_func_multiple_shooting(bc_func=self.boundarycondition_function)

        if sol.arcs is None:
            sol.arcs = [(0, len(solinit.t)-1)]

        arc_seq = sol.aux['arc_seq']
        num_arcs = len(arc_seq)

        self.prop = Propagator(**self.ivp_args)
        # sol.set_interpolate_function('cubic')
        ya = None
        tspan_list = []
        t0 = sol.t[0]
        ti = np.linspace(sol.t[0], sol.t[-1], self.num_arcs+1)
        for ii in range(len(ti) - 1):
            tspan_list.append((t0, ti[ii+1]))
            if ya is None:
                ya = np.array([sol(t0)[0]])
            else:
                ya = np.vstack((ya, sol(t0)[0]))
            t0 = ti[ii+1]

        num_arcs = self.num_arcs

        r_cur = None

        if solinit.dynamical_parameters is None:
            nParams = 0
        else:
            nParams = solinit.dynamical_parameters.size

        # Initial state of STM is an identity matrix with an additional column of zeros per dynamical parameter
        stm0 = np.hstack((np.eye(nOdes), np.zeros((nOdes,nParams)))).reshape(nOdes*(nOdes+nParams))
        n_iter = 1  # Initialize iteration counter
        converged = False  # Convergence flag

        while not converged and n_iter <= self.max_iterations:
            t_list, y_list, q_list, phi_full_list, ya, yb = self.compute_y_and_stm(tspan_list, ya, q0g, stm0, paramGuess, sol.aux, nOdes)

            if nquads == 0:
                res = self.bc_func_ms(t_list[0][0], ya[0], [], t_list[-1][-1], yb[0], [], paramGuess, nondynamical_parameter_guess, sol.aux)
            else:
                res = self.bc_func_ms(t_list[0][0], ya[0], q_list[0][0, :], t_list[-1][-1], yb[0], q_list[-1][-1, :], paramGuess, nondynamical_parameter_guess, sol.aux)

            r1 = np.linalg.norm(res)
            logging.debug('Residual: ' + str(r1))
            try:
                if any(np.isnan(res)):
                    print(res)
                    raise RuntimeError("Nan in residual")

                if r1 > self.max_error:
                    raise RuntimeError('Error exceeded max_error')
            except RuntimeError as err:
                logging.warning(err)
                import traceback
                traceback.print_exc()
                break

            # Compute Jacobian of boundary conditions
            nBCs = len(res)

            J = self._bc_jac_multi(t_list, nBCs, nquads, phi_full_list, y_list, q_list, paramGuess, nondynamical_parameter_guess, sol.aux, self.quadrature_function, self.bc_func_ms)

            try:
                raw_step = np.linalg.solve(J, -res)
            except np.linalg.LinAlgError as err:
                logging.warning(err)
                raw_step, *_ = np.linalg.lstsq(J, -res)

            a = 1e-4
            reduct = 0.5
            ll = 1.0

            if r_cur is None:
                r_cur = r1
            r_try = float('Inf')

            ya_copy = copy.deepcopy(ya)
            p_copy = copy.deepcopy(paramGuess)
            p2_copy = copy.deepcopy(nondynamical_parameter_guess)
            while (r_try >= (1 - a*ll)*r_cur) and r_try > self.tolerance:
                step = ll*raw_step
                d_ya = np.reshape(step[:nOdes * num_arcs], (num_arcs, nOdes), order='C')
                ya = ya_copy + d_ya

                if nParams > 0:
                    dp1 = step[nOdes * num_arcs:nOdes * num_arcs + paramGuess.size]
                    dp2 = step[nOdes * num_arcs + paramGuess.size:]
                    paramGuess = p_copy + dp1
                    nondynamical_parameter_guess += p2_copy + dp2

                t_list, y_list, q_list, ya, yb = self.compute_y(tspan_list, ya, q0g, paramGuess, sol.aux)
                if nquads == 0:
                    res_try = self.bc_func_ms(t_list[0][0], ya[0], [], t_list[-1][-1], yb[0], [], paramGuess, nondynamical_parameter_guess, sol.aux)
                else:
                    res_try = self.bc_func_ms(t_list[0][0], ya[0], q_list[0][0, :], t_list[-1][-1], yb[0], q_list[-1][-1, :], paramGuess, nondynamical_parameter_guess, sol.aux)

                r_try = np.linalg.norm(res_try)
                ll *= reduct
                if ll <= 0.1:
                    r1 = r_try
                    break

            r_cur = r_try
            logging.debug('Iteration #' + str(n_iter))
            # Solution converged if BCs are satisfied to tolerance
            if r_cur <= self.tolerance and n_iter > 1:
                logging.info("Converged in " + str(n_iter) + " iterations.")
                converged = True
            n_iter += 1

        if converged:
            sol.arcs = []
            timestep_ctr = 0
            for arc_idx, tt in enumerate(t_list):
                sol.arcs.append((timestep_ctr, timestep_ctr+len(tt)-1))

            sol.t = np.hstack(t_list)
            sol.y = np.row_stack(y_list)
            if nquads > 0:
                sol.q = np.row_stack(q_list)
            sol.dynamical_parameters = paramGuess
            sol.nondynamical_parameters = nondynamical_parameter_guess

        else:
            # Return a copy of the original guess if the problem fails to converge
            sol = copy.deepcopy(solinit)

        sol.converged = converged
        return sol
