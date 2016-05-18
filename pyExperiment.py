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
import pdb
import copy

# =========================
# Python Packages
# =========================
import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline as IU_Spline
# For scipy.interpolate.InterpolatedUnivariateSpline. See:
# https://github.com/scipy/scipy/blob/v0.14.0/scipy/interpolate/fitpack2.py


# =========================
# Custom Packages
# =========================

import pyStruc.Struc as Struc


# =========================
# Main Code
# =========================

class Experiment(Struc):

    def __init__(self, name = 'Experiment', *args, **kwargs):

        def_opts = {}

        if def_opts in kwargs:
            def_opts.update(kwargs.pop('def_opts'))
        #end

        Struc.__init__(self, name = name, def_opts = def_opts, *args, **kwargs)
    # end
# end


if __name__ == '__main__':
    import unittest

    class test_Experiment(unittest.TestCase):

        def test_instantiation(self):
            exp = Experiment()

            if not hasattr(exp, 'informs'):
                raise Exception()

        # end
    # end

    unittest.main(verbosity = 4)
