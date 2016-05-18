#!/usr/bin/pyton
"""

pyExperiment

Abstract class for experiments, both physical and computational

Authors
-------

- Stephen Andrews (SA)
- Andrew M. Fraiser (AMF)

Revisions
---------

0 -> Initial class creation (03-16-2016)

ToDo
----

None


"""

# =========================
# Python Standard Libraries
# =========================

import sys
import os
# import pdb
# import copy
import unittest

# =========================
# Python Packages
# =========================
# import numpy as np

# from scipy.interpolate import InterpolatedUnivariateSpline as IU_Spline
# For scipy.interpolate.InterpolatedUnivariateSpline. See:
# https://github.com/scipy/scipy/blob/v0.14.0/scipy/interpolate/fitpack2.py


# =========================
# Custom Packages
# =========================
sys.path.append(os.path.abspath('./../'))
from FUNCLE.utils.pyStruc import Struc


# =========================
# Main Code
# =========================

class Experiment(Struc):
    """Abstract class for experiments
    """
    def __init__(self, name='Experiment', *args, **kwargs):

        def_opts = {}

        if 'def_opts' in kwargs:
            def_opts.update(kwargs.pop('def_opts'))
        #end

        Struc.__init__(self, name=name, def_opts=def_opts, *args, **kwargs)
    # end
# end

class TestExperiment(unittest.TestCase):
    """Test of the experiment class
    """

    def test_instantiation(self):
        """Tests that the cass can instantiate correctlt
        """
        exp = Experiment()

        self.assertIsInstance(exp, Experiment)

        print '\n'+str(exp)
    # end
# end

if __name__ == '__main__':
    unittest.main(verbosity=4)
