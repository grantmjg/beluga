import numbers as num

from sympy import *
from beluga.utils import sympify

import copy

class Scaling(dict):
    excluded_aux = ['function']

    def __init__(self):
        self.units = {}
        self.units_sym = []
        self.scale_func = {}
        self.problem_data = {}

    """Defines scaling for a set of units"""
    def unit(self, unit_str, unit_scale):
        """Adds scaling factor for a given unit
        Allows method chaining
        """
        self.units[unit_str] = unit_scale
        return self

    def initialize(self, ws):
        """Initializes the scaling process.

        Parameters
        ----------
        ws - dict
            Workspace processed by OCP workflow
        """
        self.problem_data = ws['problem_data']

        # Generate scaling functions for states, costates
        # constants, constraints, lagrange multipliers

        # Convert units to symbols
        self.units_sym = symbols(list(self.units))

        # Growing list TODO: Put inside utils
        # TODO: Automate the following sections

        # Scaling functions for constants
        self.scale_func['const'] = {str(const): self.create_scale_fn(const.unit)
                                    for const in ws['constants']}

        # Cost function used for scaling costates
        cost_keys = ['path_cost', 'terminal_cost', 'initial_cost']
        cost_used = next(key for key in cost_keys if str(ws[key]) != '0')
        if len(cost_used) < 1:
            raise ValueError('At least one cost function must be specified as nonzero!')

        if cost_used == 'path_cost':
            cost_unit = str(ws[cost_used].unit) + '*' + str(ws['indep_var'].unit)
        else:
            cost_unit = str(ws[cost_used].unit)

        # Scaling functions for states & costates (combined)
        self.scale_func['states'] = {}
        self.scale_func['states'] = {str(state): self.create_scale_fn(state.unit)
                                     for state in ws['states']}
        costate_units = {str(costate): self.create_scale_fn('('+cost_unit+')/('+str(state.unit)+')')
                         for state, costate in zip(ws['states'],ws['costates']) }
        self.scale_func['states'].update(costate_units)

        self.scale_func['initial'] = self.scale_func['states']
        self.scale_func['terminal'] = self.scale_func['states']

        # Scaling functions for constraint multipliers and other parameters
        self.scale_func['parameters'] = {}

        # Scaling function for the independent variable
        # TODO: Fix hardcoding
        self.scale_func['parameters']['tf'] = self.create_scale_fn(ws['indep_var'].unit)

        self.scale_func['constraints'] = {}

        for c_type, c_list in ws['constraints'].items():
            for c, mul_var in zip(c_list, ws.get(c_type+'_lm_params',[])):
                mul_unit = '('+cost_unit+')/('+str(c.unit)+')'
                self.scale_func['parameters'][str(mul_var)] = self.create_scale_fn(mul_unit)

    def create_scale_fn(self,unit_expr):
        return lambdify(self.units_sym,sympify(unit_expr))

    def compute_base_scaling(self,sol,scale_expr):
        if isinstance(scale_expr,num.Number):
            # If scaling factor is a number, use it
            return scale_expr
        else:
            variables = [(aux_name,aux_val)
                    for aux_type in sol.aux
                    if isinstance(sol.aux[aux_type],dict)
                    for (aux_name,aux_val) in sol.aux[aux_type].items()
                    ]
            # Have to do in this order to override state values with arrays
            variables += [(state,max(abs(sol.y[:, idx])))
                            for idx,state in enumerate(self.problem_data['state_list'])]

            var_dict = dict(variables)

            # Evaluate expression to get scaling factor
            return float(sympify(scale_expr).subs(var_dict,dtype=float).evalf())

    def compute_scaling(self, sol):
        from collections import OrderedDict
        # Units should be stored in order to be used as function arguments
        self.scale_factors = OrderedDict()
        # Evaluate scaling factors for each base unit
        for (unit,scale_expr) in self.units.items():
            self.scale_factors[unit] = self.compute_base_scaling(sol, scale_expr)

        # Ordered list of unit scaling factors for use as function parameters
        scale_factor_list = [v for (k,v) in self.scale_factors.items()]

        # Find scaling factors for each entity in problem
        self.scale_vals = {}

        for var_type,var_funcs in self.scale_func.items():
            # If there are no sub items, use the scale factor directly
            if callable(var_funcs):
                self.scale_vals[var_type] = var_funcs(*scale_factor_list)
            else:
                # Else call scaling function for each sub item
                self.scale_vals[var_type] = {}
                for var_name,var_func in var_funcs.items():
                    self.scale_vals[var_type][var_name] = var_func(*scale_factor_list)

    def scale(self, sol):
        """Scales a BVP solution"""
        solout = copy.deepcopy(sol)

        # Additional aux entries for initial and terminal BCs
        extras = [{'type':'initial','vars':self.problem_data['state_list']},
                  {'type':'terminal','vars':self.problem_data['state_list']}]

        # Scale the states and costates
        for idx, state in enumerate(self.problem_data['state_list']):
            solout.y[:, idx] /= self.scale_vals['states'][state]

        # Scale auxiliary variables
        for aux in (self.problem_data['aux_list']+extras):
            if aux['type'] not in Scaling.excluded_aux:
                for var in aux['vars']:
                    solout.aux[aux['type']][var] /= self.scale_vals[aux['type']][var]

        # Scale parameters
        for idx, param in enumerate([str(p) for p in self.problem_data['dynamical_parameters']]):
            solout.dynamical_parameters[idx] /= self.scale_vals['parameters'][param]

        return solout

    def unscale(self, sol):
        """ Unscales a solution object"""
        solout = copy.deepcopy(sol)

        # Additional aux entries for initial and terminal BCs
        extras = [{'type': 'initial', 'vars': self.problem_data['state_list']},
                  {'type': 'terminal', 'vars': self.problem_data['state_list']}]

        # Scale the states and costates
        for idx, state in enumerate(self.problem_data['state_list']):
            solout.y[:, idx] *= self.scale_vals['states'][state]

        # Scale auxiliary variables
        for aux in (self.problem_data['aux_list']+extras):
            if aux['type'] not in Scaling.excluded_aux:
                for var in aux['vars']:
                    solout.aux[aux['type']][var] *= self.scale_vals[aux['type']][var]

        # Scale parameters
        for idx, param in enumerate([str(p) for p in self.problem_data['dynamical_parameters']]):
            solout.dynamical_parameters[idx] *= self.scale_vals['parameters'][param]

        return solout
