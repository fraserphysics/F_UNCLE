#!/usr/bin/pyton
"""

test_Experiment

Tests of the Experiment abstract class

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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# =========================
# Python Standard Libraries
# =========================

import unittest
import sys
import os
import copy
import warnings
import time

# =========================
# Python Packages
# =========================
import numpy as np
from numpy.linalg import inv
import numpy.testing as npt

# =========================
# Custom Packages
# =========================
from ..Struc import Struc
from ..PhysicsModel import PhysicsModel
from .test_PhysicsModel import SimpleModel
from ..Simulation import Simulation
from ..Experiment import Experiment


