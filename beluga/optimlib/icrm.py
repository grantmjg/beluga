from .optimlib import *
import sympy as sym


def ocp_to_bvp(ocp, guess):
    ws = init_workspace(ocp, guess)
    problem_name = ws['problem_name']
    independent_variable = ws['indep_var']
    states = ws['states']
    controls = ws['controls']
    constants = ws['constants']
    constants_of_motion = ws['constants_of_motion']
    constraints = ws['constraints']
    quantities = ws['quantities']
    initial_cost = ws['initial_cost']
    terminal_cost = ws['terminal_cost']
    path_cost = ws['path_cost']
    quantity_vars, quantity_list, derivative_fn = process_quantities(quantities)
    augmented_initial_cost = make_augmented_cost(initial_cost, constraints, location='initial')
    initial_lm_params = make_augmented_params(constraints, location='initial')
    augmented_terminal_cost = make_augmented_cost(terminal_cost, constraints, location='terminal')
    terminal_lm_params = make_augmented_params(constraints, location='terminal')
    hamiltonian, costates = make_ham_lamdot(states, path_cost, derivative_fn)
    bc_initial = make_boundary_conditions(constraints, states, costates, augmented_initial_cost, derivative_fn, location='initial')
    bc_terminal = make_boundary_conditions(constraints, states, costates, augmented_terminal_cost, derivative_fn, location='terminal')
    bc_terminal = make_time_bc(constraints, bc_terminal)
    dHdu = make_dhdu(hamiltonian, controls, derivative_fn)
    nondyn_parameters = initial_lm_params + terminal_lm_params
    costate_eoms, bc_list = make_constrained_arc_fns(states, costates, controls, nondyn_parameters, constants, quantity_vars, hamiltonian)
    dae_states, dae_equations, dae_bc, guess, temp_dgdX, temp_dgdU = make_control_dae(states, costates, controls, dHdu, guess, derivative_fn)

    # Generate the problem data
    tf_var = sympify('tf')

    dgdX = []
    for i, row in enumerate(temp_dgdX.tolist()):
        for j, expr in enumerate(row):
            if expr != 0:
                dgdX.append('dgdX[{},{}] = {}'.format(i, j, expr))

    dgdU = []
    for i, row in enumerate(temp_dgdU.tolist()):
        for j, expr in enumerate(row):
            if expr != 0:
                dgdU.append('dgdU[{},{}] = {}'.format(i, j, expr))

    out = ws
    out['states'] = states
    out['costates'] = costates
    out['initial_lm_params'] = initial_lm_params
    out['terminal_lm_params'] = terminal_lm_params
    out['problem_data'] = {'method': 'icrm',
        'problem_name': problem_name,
        'aux_list': [{'type': 'const', 'vars': [str(k) for k in constants]}],
        'state_list':[str(x) for x in it.chain(states, costates)],
        'deriv_list': [tf_var * state.eom for state in states] + [tf_var * costate.eom for costate in costates] + [tf_var*dae_eom for dae_eom in dae_equations],
        'states': states,
        'costates': costates,
        'constants': constants,
        'constants_of_motion': constants_of_motion,
        'dynamical_parameters': [tf_var],
        'nondynamical_parameters': nondyn_parameters,
        'controls': controls,
        'quantity_vars': quantity_vars,
        'dae_var_list': [str(dae_state) for dae_state in dae_states],
        'dae_eom_list': ['(tf)*(' + str(dae_eom) + ')' for dae_eom in dae_equations],
        'dae_var_num': len(dae_states),
        'costate_eoms': costate_eoms,
        'hamiltonian': hamiltonian,
        'num_states': 2 * len(states),
        'dHdu': [str(_) for _ in it.chain(dHdu)],
        'bc_initial': [str(_) for _ in bc_initial],
        'bc_terminal': [str(_) for _ in it.chain(bc_terminal, dae_bc)],
        'num_bc': len(bc_initial) + len(bc_terminal) + len(dae_bc),
        'control_options': [],
        'control_list': [str(u) for u in controls],
        'num_controls': len(controls),
        'ham_expr': str(hamiltonian),
        'quantity_list': quantity_list,
        'dgdX': dgdX,
        'dgdU': dgdU,
        'nOdes': 2 * len(states) + len(dae_states)}
    return out


def make_control_dae(states, costates, controls, dhdu, guess, derivative_fn):
    """
    Make's control law for dae (ICRM) formulation.

    :param states:
    :param costates:
    :param controls:
    :param constraints:
    :param dhdu:
    :param xi_init_vals:
    :param guess:
    :param derivative_fn:
    :return:
    """

    g = dhdu
    X = [state for state in states] + [costate for costate in costates]
    U = [c for c in controls]

    xdot = sym.Matrix([sympify(state.eom) for state in states] + [sympify(lam.eom) for lam in costates])
    # Compute Jacobian
    dgdX = sym.Matrix([[derivative_fn(g_i, x_i) for x_i in X] for g_i in g])
    dgdU = sym.Matrix([[derivative_fn(g_i, u_i) for u_i in U] for g_i in g])

    udot = dgdU.LUsolve(-dgdX*xdot) # dgdU * udot + dgdX * xdot = 0

    dae_states = U
    dae_equations = list(udot)
    dae_bc = g

    guess.dae_num_states = len(U)

    yield dae_states
    yield dae_equations
    yield dae_bc
    yield guess
    yield dgdX
    yield dgdU
