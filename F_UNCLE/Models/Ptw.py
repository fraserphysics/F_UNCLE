# !/usr/bin/pthon2
"""Preston-Tonks-Wallace Flow Stress Model

This module implements the PTW flow stress model

Authors
-------

- Stephen A. Andrews (SA)
- Diane E. Vaughan (DEV)

Version History
---------------

 0.0: 13/05/2016 - Initial class creation

References
----------

[1] Preston, D. L.; Tonks, D. L. & Wallace, D. C. Model of plastic deformation
    for extreme loading conditions Journal of Applied Physics, 2003, 93, 211-220

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# =========================
# Python Standard Libraries
# =========================
import sys
import os
import unittest
from math import pi, erf, log
import pdb
# =========================
# Python Installed Packages
# =========================

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Module Packages
# =========================

from ..Utils.Struc import Struc
from ..Utils.PhysicsModel import PhysicsModel

# =========================
# Main Code
# =========================

class Ptw(PhysicsModel):
    """PTW Flow stress model

    **Usage**

    1. Instantiate a Ptw object with desired options
    2. *optional* set the options as desired
    3. Call the Ptw object with a valid temperature, strain rate and material
       specification
    4. Call the object again, material must be specified each time

    """

    def __init__(self, name="Ptw flow stress", *args, **kwargs):
        """Instantiates the structure

        Args:
            name(str): Name of the structure
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Return:
            None

        """

        def_opts = {
            'materials': [list,
                          ['Cu', 'U', 'Ta', 'V', 'Mo', 'Be', 'SS_304',
                           'SS_21-6-9'],
                          None, None, 'None', 'List of available materials'
                         ],
            'temp_check': [float, 300.0, 0.0, 2E3, 'K',
                           'Dummy option to check temp input'],
            'str_rt_chk': [float, 0.0, 0.0, 1E12, 's**-1',
                           'Dummy option to check strain rate input'],
            'matname': [(str), 'Cu', None, None, '-',
                        'Current material specification'],
            'rho': [float, float('nan'), 0, None, 'kg/m**3',
                    'Density'],
            'm_weight': [float, float('nan'), 0, None, 'g/gmole',
                         'Molecular weight'],
            'Tm': [float, float('nan'), 0, None, '??',
                   'Melt temperature'],
            'theta': [float, float('nan'), 0, None, '??',
                      'Ptw theta'],
            'p': [float, float('nan'), 0, None, '??',
                  'Ptw p'],
            's_o': [float, float('nan'), 0, None, '??',
                    'Flow stress at zero temperature'],
            's_inf': [float, float('nan'), 0, None, '??',
                      'Flow stress at infinite temperature'],
            'kappa': [float, float('nan'), 0, None, '??',
                      'Ptw kappa'],
            'gamma': [float, float('nan'), 0, None, '??',
                      'Ptw gamma'],
            'y_o': [float, float('nan'), 0, None, '??',
                    'Yield stress at zero temperature'],
            'y_inf': [float, float('nan'), 0, None, '??',
                      'Yield stress at infinite temperature'],
            'y_1': [float, float('nan'), 0, None, '??',
                    'Ptw y_1'],
            'y_2': [float, float('nan'), 0, None, '??',
                    'Ptw y_2'],
            'beta': [float, float('nan'), 0, None, '??',
                     'Ptw beta'],
            'G_o': [float, float('nan'), 0, None, '??',
                    'Shear modulus at zero temperature'],
            'alpha': [float, float('nan'), 0, None, '??',
                      'Ptw alpha'],
            'alpha_p': [float, float('nan'), 0, None, '??',
                        'Ptw alpha p']}

        Struc.__init__(self, name, def_opts=def_opts, *args, **kwargs)

    def __call__(self, temp, strain_rate, material, **overrides):
        """Solves for the yield stress and flow stress at the given condition

        Args:
           temp(float): Temperature, in degrees Kelvin
	   strain_rate(float): Strain rate, in sec**-1
	   material(str): Key for material type
        Keyword Args:
           **overrides(dict): Passed as a chain of keyword arguments. These
                              arguments override any material property

        Return:
           flow_stress(float): Flow stress in ??Pa??
           yield_stress(float): Yield stress in ??Pa??

        """

        g_modu, t_norm, psi_norm = self.__pre__(temp, strain_rate, material,
                                                **overrides)

        s_o = self.get_option('s_o')
        s_inf = self.get_option('s_inf')
        kappa = self.get_option('kappa')
        beta = self.get_option('beta')
        y_o = self.get_option('y_o')
        y_inf = self.get_option('y_inf')
        y_1 = self.get_option('y_1')
        y_2 = self.get_option('y_2')

        if psi_norm == 0.0:
            erf_psi_norm = 0.0
        else:
            erf_psi_norm = erf(kappa * t_norm * log(psi_norm**(-1)))
        # end

        glide_flow_stress = s_o - (s_o - s_inf) * erf_psi_norm

        shock_flow_stress = s_o * psi_norm**beta
        glide_yield_stress = y_o - (y_o - y_inf) * erf_psi_norm

        shock_yield_stress = y_1 * psi_norm**y_2

        flow_stress = max((glide_flow_stress, shock_flow_stress))
        yield_stress = max((glide_yield_stress,
                            min((shock_yield_stress, shock_flow_stress))))

        flow_stress *= g_modu
        yield_stress *= g_modu

        return flow_stress, yield_stress

    def get_stress_strain(self, temp, strain_rate, material, min_strain=0.05,
                          max_strain=0.7, **overrides):
        """Returns the stress strain relationship for the material

        """

        t_s, t_y = self(temp, strain_rate, material, **overrides)

        g_modu, t_norm, psi_norm = self.__pre__(temp, strain_rate, material,
                                                **overrides)

        t_s /= g_modu
        t_y /= g_modu

        s_o = self.get_option('s_o')
        p = self.get_option('p')
        theta = self.get_option('theta')

        ratio = p * (t_s - t_y) / (s_o - t_y)

        strain = np.linspace(min_strain, max_strain, 100)

        stress = -p * theta * strain / ((s_o - t_y) * (np.exp(ratio) - 1))
        stress = np.exp(stress)
        stress *= (1 - np.exp(-ratio))
        stress = np.log(1 - stress)
        stress *= p**-1 * (s_o - t_y)
        stress += t_s

        stress *= g_modu

#        if np.any(np.isnan(stress)): pdb.set_trace()

        return stress, strain

    def get_stress(self, strain, strain_rate, temp, material, **overrides):
        """Returns the stress in the material material

        Args:
            strain(float or np.array): The strain in the material
            strain_rate(float): The train rate in the material
            temp(float): The temperature of the material
            mat(str): A valid material specifier

        Keyword Args:
            valid materials properties can be passed as kwargs to
            override default values

        Return:
            (float or np.array): The stress in the material

        """

        t_s, t_y = self(temp, strain_rate, material, **overrides)

        g_modu = self.__pre__(temp, strain_rate, material, **overrides)[0]

        t_s /= g_modu
        t_y /= g_modu

        s_o = self.get_option('s_o')
        p = self.get_option('p')
        theta = self.get_option('theta')

        ratio = p * (t_s - t_y) / (s_o - t_y)

        poisson = 0.23
        yeild_strain = t_y / (g_modu * 2 * (1 + poisson))

        if strain < yeild_strain:
            stress = strain * g_modu * 2 * (1 + poisson)
        else:
            stress = -p * theta * strain / ((s_o - t_y) * (np.exp(ratio) - 1))
            stress = np.exp(stress)
            stress *= (1 - np.exp(-ratio))
            stress = np.log(1 - stress)
            stress *= p**-1 * (s_o - t_y)
            stress += t_s

            stress *= g_modu
        # end

        return stress

    def __pre__(self, temp, strain_rate, material, **overrides):
        """Performs data conditioning

        Performs the following tasks

            1. Ensure the temperature input is in a valid range
            2. Ensures the strain rate input is in a valid range
            3. Ensure a valid material has been specified
            4. Populates the materials database
            5. Applies the overrides
            6. Returns the normalizing factors

        Args:
            temp(float): Temperature, in degrees Kelvin
            strain_rate(float): Strain rate, in sec**-1
            material(str): Key for material type

        Keyword Args:
           **overrides(dict): Passed as a chain of keyword arguments. These
                              arguments override any material property

        Returns
           (float): The shear modulus
           (float): The characteristic temperature. Temp normalized by melt temp
           (float): The characteristic timescale. Time for a vibration to pass
           through one atom

        """

        # uses the Struc object to perform the checks
        self.set_option('temp_check', temp)
        self.set_option('str_rt_chk', strain_rate)

        if material not in self.get_option('materials'):
            raise IOError('{} invalid material specified'
                          .format(self.get_inform(1)))
        else:
            self.get_mat_data(material)
        # end

        for key in overrides:
            try:
                self.set_option(key, overrides[key])
            except KeyError:
                raise KeyError("{} {} is not a valid material property key"
                               .format(self.get_inform(1), key))
            # end
        # end

        t_melt = self.get_melt_temperature(temp, strain_rate)
        self.set_option('Tm', t_melt)

        g_modu = self.get_shear_modulus(temp, strain_rate, t_melt)

        rho = self.get_option('rho')
        m_weight = self.get_option('m_weight')
        gamma = self.get_option('gamma')

        # Natural time scale
        ksi_dot = 0.5 * (4.0 * pi * rho / (3 * m_weight))**(1.0 / 3.0)\
            * (g_modu / rho)**0.5

        return g_modu, temp / t_melt, gamma * strain_rate / ksi_dot

    def get_mat_data(self, material):
        """Gets the material data corresponding to the given material

        Args:
            material(str): Key for material type

        """

        mat_data = np.array(
            [(0.025, 0.055, 0.02, 0.023, 0.014, 0.04, 0.02, 0.02),
             (2.0, 1.0, 0.0, 0.0, 0.0, 1.4, 10.0, 8.0),
             (0.0085, 0.03, 0.012, 0.013, 0.00945, 0.007, 0.05, 0.05),
             (0.00055, 0.00015, 0.00325, 0.00405, 0.0038, 0.0012, 0.0075,
              0.0075),
             (0.11, 0.13, 0.6, 0.4, 0.41, 0.14, 0.3, 0.3),
             (0.00001, 0.002, 0.00004, 0.00006, 0.000008, 0.00001, 0.001,
              0.001),
             (0.0001, 0.00075, 0.01, 0.0105, 0.00795, 0.0015, 0.0069, 0.0125),
             (0.0001, 0.00075, 0.00125, 0.00155, 0.0023, 0.0005, 0.0015,
              0.00225),
             (0.094, 0.03, 0.012, 0.013, 0.00945, 0.007, 0.05, 0.05),
             (0.575, 0.27, 0.4, 0.42, 0.36, 0.25, 0.46, 0.41),
             (0.25, 0.25, 0.23, 0.23, 0.23, 0.25, 0.25, 0.25),
             (518, 938, 722, 499, 1303, 1524, 895, 862),
             (0.20, 0.56, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23),
             (0.43, 0.72, 0.48, np.nan, 0.41, np.nan, 0.66, 0.37),
             (8.933, 19.07, 16.75, np.nan, np.nan, np.nan, np.nan, np.nan),
             (64.54, 238.04, 180.948, np.nan, np.nan, np.nan, np.nan, np.nan)
            ], dtype={'names': self.get_option('materials'),
                      'formats': ['f8'] * 8})

        self.set_option('matname', material)
        self.set_option('theta', mat_data[material][0])
        self.set_option('p', mat_data[material][1])
        self.set_option('s_o', mat_data[material][2])
        self.set_option('s_inf', mat_data[material][3])
        self.set_option('kappa', mat_data[material][4])
        self.set_option('gamma', mat_data[material][5])
        self.set_option('y_o', mat_data[material][6])
        self.set_option('y_inf', mat_data[material][7])
        self.set_option('y_1', mat_data[material][8])
        self.set_option('y_2', mat_data[material][9])
        self.set_option('beta', mat_data[material][10])
        self.set_option('G_o', mat_data[material][11] * 1E8)
        self.set_option('alpha', mat_data[material][12])
        self.set_option('alpha_p', mat_data[material][13])
        self.set_option('rho', mat_data[material][14] * 1E3)
        self.set_option('m_weight', mat_data[material][15])

    def get_shear_modulus(self, temp, strain_rate, t_melt):
        """Gets the shear modulus for the operating conditions

        Args:
            temp(float): Temperature, in degrees Kelvin
            strain_rate(float): Strain rate, in sec**-1
            t_melt(float): Melt temperature of the material in Kelvin

        """

        return self.get_option('G_o') *\
            (1 - self.get_option('alpha') * temp / t_melt)

    def get_melt_temperature(self, temp, strain_rate):
        """Gets the melt temperature for the operating conditions

        Args:
            temp(float): Temperature, in degrees Kelvin
            strain_rate(float): Strain rate, in sec**-1

        Returns:
            (float): The melt temperature in Kelvin

        """

        melt_temp = {
            'Cu': 1084.88,
            'U': 1406 - 273.15,
            'Ta': 2996,
            'V': 1910,
            'Mo': 2610,
            'Be': 1283,
            'SS_304': 1425,
            'SS_21-6-9': 1425,
        }

        return melt_temp[self.get_option('matname')] + 273.15


#-------------------------
# Local Variables:
# eval: (python-mode)
# eval: (flycheck-mode)
# End:
