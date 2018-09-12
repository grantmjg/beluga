import copy
import itertools as it
import logging
import numpy as np
from copy import deepcopy

from beluga.bvpsol.algorithms.BaseAlgorithm import BaseAlgorithm
from beluga.ivpsol import Propagator
from multiprocessing_on_dill import pool


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

    """

    def __init__(self, cached=True, derivative_method='fd', ivp_args={}, tolerance=1e-4, max_error=100,
                 max_iterations=100, num_arcs=1, num_cpus=1, use_numba=False, verbose=True):

        """
            #     Initializes a new Shooting object.
            #
            #     :param args: Unused
            #     :param kwargs: Additional parameters accepted by the solver.
            #     :return: Shooting object.
            #
            #     +------------------------+-----------------+-----------------+
            #     | Valid kwargs           | Default Value   | Valid Values    |
            #     +========================+=================+=================+
            #     | cached                 | True            | Bool            |
            #     +------------------------+-----------------+-----------------+
            #     | derivative_method      | 'fd'            | {'fd','csd'}    |
            #     +------------------------+-----------------+-----------------+
            #     | ivp_args               | {}              | see `ivpsol`    |
            #     +------------------------+-----------------+-----------------+
            #     | tolerance              | 1e-4            | > 0             |
            #     +------------------------+-----------------+-----------------+
            #     | max_error              | 100             | > 0             |
            #     +------------------------+-----------------+-----------------+
            #     | max_iterations         | 100             | > 0             |
            #     +------------------------+-----------------+-----------------+
            #     | num_arcs               | 1               | > 0             |
            #     +------------------------+-----------------+-----------------+
            #     | num_cpus               | 1               | > 0             |
            #     +------------------------+-----------------+-----------------+
            #     | use_numba              | False           | Bool            |
            #     +------------------------+-----------------+-----------------+
            #     | verbose                | False           | Bool            |
            #     +------------------------+-----------------+-----------------+
            #     """

        self.cached = cached
        self.derivative_method = derivative_method
        self.ivp_args = ivp_args
        self.tolerance = tolerance
        self.max_error = max_error
        self.max_iterations = max_iterations
        self.num_arcs = num_arcs
        self.num_cpus = num_cpus
        self.use_numba = use_numba
        self.verbose = verbose

        self.stm_ode_func = None
        self.saved_code = True

        if self.num_cpus > 1:
            self.pool = pool.Pool(processes=self.num_cpus)
        else:
            self.pool = None

        if self.derivative_method not in ['fd']:
            raise ValueError("Invalid derivative method specified. Valid options are 'csd' and 'fd'.")

        self.deriv_func = None
        self.quad_func = None

        # Make the state-transition ode matrix
        self.stm_ode_func = None

        # Set up the boundary condition function
        self.bc_func = None

        self.prop = Propagator(**self.ivp_args)

    def load_problem_info(self, deriv_func, quad_func, bc_func, n_odes):

        self.deriv_func = deriv_func
        self.quad_func = quad_func

        # Make the state-transition ode matrix
        self.stm_ode_func = self.make_stmode(deriv_func, n_odes)

        # Set up the boundary condition function
        self.bc_func = self._bc_func_multiple_shooting(bc_func=bc_func)

    @staticmethod
    def _bc_jac_multi(t_list, nBCs, phi_full_list, y_list, parameters, aux, bc_func, StepSize=1e-7):
        p = np.array(parameters)
        nParams = p.size
        h = StepSize
        ya = np.array([traj[0] for traj in y_list]).T
        yb = np.array([traj[-1] for traj in y_list]).T
        nOdes = ya.shape[0]
        num_arcs = len(phi_full_list)
        fx = bc_func(t_list[0][0], ya, [], t_list[-1][-1], yb, [], p, aux)
        nBCs = len(fx)

        M = np.zeros((nBCs, nOdes))
        P = np.zeros((nBCs, p.size))
        Ptemp = np.zeros((nBCs, p.size))
        J = np.zeros((nBCs, nOdes*num_arcs+p.size))
        dx = np.zeros((nOdes+nParams, num_arcs))

        for arc_idx, phi in zip(it.count(), phi_full_list):
            # Evaluate for all arcs
            for i in range(nOdes):
                dx[i, arc_idx] = dx[i, arc_idx] + h
                dy = np.dot(phi[-1], dx)
                f = bc_func(t_list[0][0], ya + dx[:nOdes], [], t_list[-1][-1], yb + dy, [], p, aux)
                M[:, i] = (f-fx)/h
                dx[i, arc_idx] = dx[i, arc_idx] - h
            J_i = M
            J_slice = slice(nOdes*arc_idx, nOdes*(arc_idx+1))
            J[:, J_slice] = J_i

        for arc_idx, phi in zip(it.count(), phi_full_list):
            for i in range(p.size):
                p[i] = p[i] + h
                j = i + nOdes
                dx[j, arc_idx] = dx[j, arc_idx] + h
                dy = np.dot(phi[-1], dx)
                f = bc_func(t_list[0][0], ya, [], t_list[-1][-1], yb + dy, [], p, aux)
                Ptemp[:, i] = (f-fx)/h
                dx[j, arc_idx] = dx[j, arc_idx] - h
                p[i] = p[i] - h
            P += Ptemp
            Ptemp = np.zeros((nBCs, p.size))

        J_i = P
        J_slice = slice(nOdes * num_arcs, nOdes * num_arcs + nParams)
        J[:, J_slice] = J_i

        return J

    @staticmethod
    def _bc_func_multiple_shooting(bc_func=None):
        def _bc_func(t0, y0, q0, tf, yf, qf, paramGuess, aux):
            bc1 = np.array(bc_func(t0, y0, q0, tf, yf, qf, paramGuess, aux)).flatten()
            narcs = y0.shape[1]
            bc2 = np.array([y0[:, ii + 1] - yf[:, ii] for ii in range(narcs - 1)]).flatten()
            bc = np.hstack((bc1, bc2))
            return bc

        def wrapper(t0, y0, q0, tf, yf, qf, paramGuess, aux):
            return _bc_func(t0, y0, q0, tf, yf, qf, paramGuess, aux)

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

        def wrapper(t, _X, p, const, arc_idx):  # needed for scipy
            return _stmode_fd(t, _X, p, const, arc_idx)
        return wrapper

    def compute_y_and_stm(self, tspan_list, ya, stm0, p, aux, n_odes):

        phi_full_list = []
        t_list = []
        y_list = []

        y0stm = np.zeros((len(stm0)+n_odes))
        yb = np.zeros_like(ya)

        for arc_idx, tspan in enumerate(tspan_list):
            y0stm[:n_odes] = ya[arc_idx, :]
            y0stm[n_odes:] = stm0[:]
            q0 = []
            sol_ivp = self.prop(self.stm_ode_func, None, tspan, y0stm, q0, p, aux, 0)  # TODO: arc_idx is hardcoded as 0 here, this'll change with path constraints. I51
            t = sol_ivp.t
            yy = sol_ivp.y
            y_list.append(yy[:, :n_odes])
            t_list.append(t)
            yb[arc_idx, :] = yy[-1, :n_odes]
            phi_full = np.reshape(yy[:, n_odes:], (len(t), n_odes, n_odes + len(p)))
            phi_full_list.append(np.copy(phi_full))

        return t_list, y_list, phi_full_list, ya, yb

    def compute_y(self, tspan_list, ya, p, aux):

        t_list = []
        y_list = []

        yb = np.zeros_like(ya)

        for arc_idx, tspan in enumerate(tspan_list):
            y0 = ya[arc_idx, :]
            q0 = []
            sol_ivp = self.prop(self.deriv_func, None, tspan, y0, q0, p, aux, 0)
            t = sol_ivp.t
            yy = sol_ivp.y
            y_list.append(yy[:, :])
            t_list.append(t)
            yb[arc_idx, :] = yy[-1, :]

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
        sol.parameters = np.array(sol.parameters, dtype=np.float64)

        # Extract some info from the guess structure
        y0g = sol.y[0, :]
        nOdes = y0g.shape[0]
        paramGuess = sol.parameters

        if sol.arcs is None:
            sol.arcs = [(0, len(solinit.t)-1)]

        arc_seq = sol.aux['arc_seq']
        num_arcs = len(arc_seq)

        # TODO: These are specific to an old implementation of path constraints see I51
        left_idx, right_idx = map(np.array, zip(*sol.arcs))
        ya = sol.y[left_idx, :]
        yb = sol.y[right_idx, :]
        tmp = np.arange(num_arcs+1, dtype=np.float32)*sol.t[-1]  # TODO: I51
        tspan_list = [(a, b) for a, b in zip(tmp[:-1], tmp[1:])]  # TODO: I51

        # sol.set_interpolate_function('cubic')
        ya = None
        tspan_list = []
        t0 = sol.t[0]
        ti = np.linspace(sol.t[0], sol.t[-1], self.num_arcs+1)
        for ii in range(len(ti)-1):
            tspan_list.append((t0, ti[ii+1]))
            if ya is None:
                ya = np.array([sol(t0)[0]])
            else:
                ya = np.vstack((ya, sol(t0)[0]))
            t0 = ti[ii+1]

        num_arcs = self.num_arcs

        r_cur = None

        if solinit.parameters is None:
            nParams = 0
        else:
            nParams = solinit.parameters.size

        # Initial state of STM is an identity matrix with an additional column of zeros per parameter
        stm0 = np.hstack((np.eye(nOdes), np.zeros((nOdes, nParams)))).reshape(nOdes*(nOdes+nParams))
        n_iter = 1  # Initialize iteration counter
        converged = False  # Convergence flag

        while not converged and n_iter <= self.max_iterations:

            if n_iter == 1:
                if not self.saved_code:
                    self.save_code()
                    self.saved_code = True

            t_list, y_list, phi_full_list, ya, yb = \
                self.compute_y_and_stm(tspan_list, ya, stm0, paramGuess, sol.aux, nOdes)

            # Determine the error vector
            res = self.bc_func(t_list[0][0], ya.T, [], t_list[-1][-1], yb.T, [], paramGuess, sol.aux)

            r1 = max(abs(res))

            if self.verbose:
                logging.debug('Residual: ' + str(r1))

            try:
                # Break cycle if there are any NaNs in our error vector
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
            J = self._bc_jac_multi(t_list, nBCs, phi_full_list, y_list, paramGuess, sol.aux, self.bc_func)

            try:
                raw_step = np.linalg.solve(J, -res)
            except np.linalg.LinAlgError as err:
                logging.warning(err)
                raw_step, *_ = np.linalg.lstsq(J, -res)

            a = 1e-4
            reduct = 0.5

            ll = 1
            if r_cur is None:
                r_cur = r1
            r_try = float('Inf')

            ya_copy = deepcopy(ya)
            p_copy = deepcopy(paramGuess)

            while (r_try >= (1 - a*ll)*r_cur) and (r_try > self.tolerance):

                step = ll*raw_step

                d_ya = np.reshape(step[:nOdes * num_arcs], (num_arcs, nOdes), order='C')
                ya = ya_copy + d_ya

                if nParams > 0:
                    dp_try = step[nOdes * num_arcs:]
                    paramGuess = p_copy + dp_try

                t_list, y_list, ya, yb = self.compute_y(tspan_list, ya, paramGuess, sol.aux)

                res_try = self.bc_func(t_list[0][0], ya.T, [], t_list[-1][-1], yb.T, [], paramGuess, sol.aux)
                r_try = max(abs(res_try))

                ll *= reduct
                if ll <= 0.1:
                    r1 = r_try
                    break

            r_cur = r_try

            # Solution converged if BCs are satisfied to tolerance
            if r1 <= self.tolerance and n_iter > 1:
                if self.verbose:
                    logging.info("Converged in "+str(n_iter)+" iterations.")
                converged = True
                break

            n_iter += 1
            logging.debug('Iteration #' + str(n_iter))

        if converged:
            sol.arcs = []
            timestep_ctr = 0
            for arc_idx, tt in enumerate(t_list):
                sol.arcs.append((timestep_ctr, timestep_ctr+len(tt)-1))

            sol.t = np.hstack(t_list)
            sol.y = np.row_stack(y_list)
            sol.parameters = paramGuess

        else:
            # Return a copy of the original guess if the problem fails to converge
            sol = copy.deepcopy(solinit)

        sol.converged = converged
        return sol
