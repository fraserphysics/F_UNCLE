"""

For now, this is a copy of Stick.py

Authors
-------

- Stephen Andrews (SA)
- Andrew M. Fraiser (AMF)

Revisions
---------

0 -> Initial class creation (06-06-2016)

ToDo
----

None


"""

# =========================
# Python Standard Libraries
# =========================

import sys
import os
import pdb
import copy
import unittest

# =========================
# Python Packages
# =========================
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Custom Packages
# =========================
if __name__ == '__main__':
    sys.path.append(os.path.abspath('./../../'))
    from F_UNCLE.Utils.Experiment import Experiment
    from F_UNCLE.Models.Isentrope import EOSBump, EOSModel, Isentrope, Spline
else:
    from ..Utils.Experiment import Experiment
    from ..Models.Isentrope import EOSBump, EOSModel, Isentrope, Spline
# end


# =========================
# Main Code
# =========================


class Stick(Experiment):
    """A toy physics model representing a gun type experiment

    Attributes:
        const(dict): A dictionary of conversion factors

    """
    def __init__(self, eos, name='Gun Toy Computational Experiment', *args, **kwargs):
        """Instantiate the Experiment object

        Args:
            eos(Isentrope): The equation of state model used in the toy computational
                            experiment

        Keyword Args:
            name(str): A name. (Default = 'Gun Toy Computational Experiment')

        """

        if isinstance(eos, Isentrope):
            self.eos = eos
        else:
            raise TypeError('{:} Equation of state model must be an Isentrope object'
                            .format(self.get_inform(2)))
        # end

        def_opts = {
            'x_i': [float, 0.4, 0.0, None, 'cm',
                    'Initial position of projectile'],
            'x_f': [float, 3.0, 0.0, None, 'cm',
                    'Final/muzzle position of projectile'],
            'm': [float, 500.0, 0.0, None, 'g',
                  'Mass of projectile'],
            'mass_he': [float, 4, 0.0, None, 'g',
                        'The initial mass of high explosives used to drive\
                        the projectile'],
            'area': [float, 1.0, 0.0, None, 'cm**2',
                     'Projectile cross section'],
            'sigma': [float, 1.0e0, 0.0, None, '??',
                      'Variance attributed to v measurements'],
            't_min': [float, 1.0e-6, 0.0, None, 'sec',
                      'Range of times for t2v spline'],
            't_max': [float, 1.0e-2, 0.0, None, 'sec',
                      'Range of times for t2v spline'],
            'n_t': [int, 250, 0, None, '',
                    'Number of times for t2v spline']
        }

        self.const = {'newton2dyne': 1e5,
                      'cm2km': 1.0e5}

        Experiment.__init__(self, name=name, def_opts=def_opts, *args, **kwargs)

    def update(self, model=None):
        """Update the analysis with a new model
        """
        if model is None:
            pass
        elif isinstance(model, Isentrope):
            self.eos = copy.deepcopy(model)
        else:
            raise TypeError('{}: Model must be an isentrope for update'
                            .format(self.get_inform(1)))
        # end

    def _on_str(self, *args, **kwargs):
        """Print method of the gun model

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Return:
            (str): A string representing the object
        """

        out_str = '\n'
        out_str += 'Equation of State Model\n'
        out_str += '-----------------------\n'
        out_str += str(self.eos)

        return out_str
