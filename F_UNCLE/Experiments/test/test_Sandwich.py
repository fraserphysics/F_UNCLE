"""Tests of the ToySandwich class
"""
import sys
import os
import unittest

import matplotlib.pyplot as plt

from ...Models.Isentrope import EOSBump, EOSModel, Isentrope, Spline
from ...Models.SimpleStr import SimpleStr
from ..Sandwich import ToySandwich, ToySandwichExperiment
from ..Cylinder import ToyCylinder#, ToyCylinderExperiment

class TestToySandwich(unittest.TestCase):

    def setUp(self):
        """
        """
        self.eos_model = EOSModel(lambda v: 2.56E9/v**3, spline_max = 2.0)
        self.eos_true = EOSBump()


    def tearDown(self):
        """
        """

        pass

    def test_instantiate_simulation(self):
        """Test that the simulation is properly instantiated with defaults
        """

        sandwich = ToySandwich()

    def test_instantiate_experiment(self):
        """Test tha the experiment is instantiated with defaults
        """

        exp = ToySandwichExperiment(model=self.eos_true)


    def test_run_simulation(self):
        """Tests that a call to the simulations solves correctly
        """
        sandwich = ToySandwich()
        data = sandwich({'eos': self.eos_model})


    def test_plot(self):
        """Test plotting the sandwich data
        """
        sandwich = ToySandwich()
        data1 = sandwich({'eos': self.eos_model})
        data2 = sandwich({'eos': self.eos_true})

        fig = sandwich.plot(data = [data1, data2],
                            labels=["EOS Model", "EOS True"],
                            linestyles=['-k', '--k', 'r'])

        fig.savefig('sandwich_test.pdf')
        
    
    def test_run_experiment(self):
        """Test that a call to the experiment solves correctly
        """

        exp = ToySandwichExperiment(model=self.eos_true)
        expdata = exp()

    
    def test_sim_compare(self):
        """Test comparing two simulations
        """

        sandwich = ToySandwich()
        data1 = sandwich({'eos': self.eos_model})
        data2 = sandwich({'eos': self.eos_true})
       
        sandwich.compare(data1, data2)
    def test_exp_compare(self):
        """Test comparing a sim to an experiment
        """

        sandwich = ToySandwich()
        simdata = sandwich({'eos': self.eos_model})

        exp = ToySandwichExperiment(model=self.eos_true)
        expdata = exp()

        aligned_sim = exp.align(simdata)
        epsilon = exp.compare(aligned_sim)

        fig = sandwich.plot(data = [aligned_sim, expdata],
                            labels=["Simulation", "Experiment"],
                            linestyles=['-k', '--k', 'r'])

        fig.savefig('sandwich_exp.pdf')
        

class TestToyCylinder(unittest.TestCase):
    """Test of the toy clyinder problem
    """

    def setUp(self):
        """
        """
        self.eos_model = EOSModel(lambda v: 2.56E9/v**3, spline_max = 2.0)
        self.eos_true = EOSBump()
        self.str_model = SimpleStr([101E9, 0.0])
    def test_instantiate_simulation(self):
        """Test that the Cylinder can be instantiated
        """

        cyl = ToyCylinder()
        
    def test_run_simulation(self):
        """Test that the Cylinder can be run
        """

        cyl = ToyCylinder()
        
        data = cyl({'eos': self.eos_model,
                    'strength': self.str_model})

        fig = cyl.plot(data=[data])
        ax1 = fig.gca()
        ax2 = ax1.twinx()
        ax2.plot(1E6 * data[0], data[1][2])
        fig.savefig("cylinder_sim.pdf")
