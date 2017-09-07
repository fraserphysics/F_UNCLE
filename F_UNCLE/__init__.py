"""The FUNCLE module

Functional UNcertainty Constrained by Law and Experiment


User's Guide
============

Definitions
-----------

Physics Process

   "A *physics process* is a relation between two or more physical
   quantities, e.g. the rate at which plasma ion and electron
   temperatures equilibrate as a function of the individual
   temperatures, particle masses, and number densities. Note that a
   physics process need not be time dependent, e.g. the shear modulus
   as a function of temperature and density.":cite:`VaughanPUBS2014`

Physics Model

   "A physics model is an approximate mathematical representation of a
   physics process.  As an example of a physics model, consider the
   plastic constitutive relation of a metal. This relates the basic
   quantities governing plastic deformation, e.g. the flow stress,
   plastic strain, strain rate, temperature, and density. It is an
   approximate representation of the very complicated mesoscale
   physics responsible for the flow of a solid, and is therefore a
   physics model.":cite:`VaughanPUBS2014`

Degree of Freedom

   The mathematical representation of a physics model is dependent on
   a set of variables whose values can alter the behavior of the
   model. These variables are referred to as the degrees of freedom or
   DOF

Experiment

   An experiment represents data measured from a real physical
   experiments. Experiments used in F_UNCLE are chosen so that their
   results are dominated by a single physics process which can be
   represented by a physics model Simulation

Simulation

   A simulation which represents the physical system from which the
   experimental data were obtained. The simulation is dependent on one
   or more physics models and acts as a map between the model degree
   of freedom space and the experimental result space




F_UNCLE Parent Classes
----------------------

These concepts are implemented in the following three fundamental
F_UNCLE classes

Physics Model
.............

:py:class:`F_UNCLE.Utils.PhysicsModel.PhysicsModel`. This class
represents a physics model. The degrees of freedom of a `PhysicsModel`
are immutable. Setting the degrees of freedom of a `PhysicsModel` will
return a independent copy of itself with updated DOF.

    - :py:func:`F_UNCLE.Utils.PhysicsModel.PhysicsModel.get_dof`: Returns
      a list of the model degrees of freedom

    - :py:func:`F_UNCLE.Utils.PhysicsModel.PhysicsModel.set_dof`: Generates
      a new Physics Model with the specified degrees of freedom

    - :py:func:`F_UNCLE.Utils.PhyscisModel.PhysicsModel.get_sigma`: Returns
      a square matrix representing the co-variance of the model degrees of
      freedom

    - :py:func:`F_UNCLE.Utils.PhyscisModel.PhysicsModel.shape`: Returns
      an integer representing the number of model degrees of freedom

Experiment
..........

:py:class:`F_UNCLE.Utils.Experiment.Experiment`. This class represents
the results for a single experiment performed under a single set of
conditions.

   - :py:func:`F_UNCLE.Utils.Experiment.Experiment.__call__`: Returns
     a tuple with the following elements:

      0. The independent variable of the simulation (i.e. time)

      1. List of arrays of dependent variables. Element zero is the
         element for comparison, the last element is the labels for
         the previous elements

      2. Dictionary of additional simulation result
         attributes. Contains at least the key `mean_fn` which is a
         functional representation of the relationship between the
         independent variable and the dependent variable for
         comparison

    - :py:func:`F_UNCLE.Utils.Experiment.Experiment.get_sigma`:
      Returns an nxn co-variance matrix for the experimental data

    - :py:func:`F_UNCLE.Utils.Experiment.Experiment.shape`: Returns an
      integer representing the number of experimental data points

    - :py:func:`F_UNCLE.Utils.Experiment.Experiment.get_fisher_matrix`:
      This returns the fisher information matrix of the experiment given
      the simulation's sensitivity matrix

The `Experiment` object provides some internal routines to ease
comparison of simulations and experiments

   - :py:func:`F_UNCLE.Utils.Experiment.Experiment.align`: This takes
     a set of simulation data and returns a copy of it with the
     simulation data aligned so it is evaluated at each independent
     data value of the experiment. This 'aligned' tuple of simulation
     data has a new key `tau` which is the shift in independent
     variable (likely time) required to align the simulation to the
     experiment. In addition, the `mean_fn` attribute has been modified
     so it returns values aligned to the experimental data.

  - :py:fun:`F_UNCLE.Utils.Experiment.Experiment.compare`: This takes
    a set of simulation data and returns the error between the
    experiment and the simulation. The simulation can either be
    aligned or not. The return value is the experimental value *less*
    the aligned simulation value

Simulation
..........

:py:class:`F_UNCLE.Utils.Simulation.Simulation`. This class wraps some
computer simulation which is dependent on a PhysicsModel

    - :py:func:`F_UNCLE.Utils.Simulation.Simulation.compare`: Compares
      two simulations results and returns the difference

    - :py:func:`F_UNCLE.Utils.Simulations.Simulations.__call__`: Returns
      the results of the simulation as a tuple

        0. The independent variable of the simulation (i.e. time)

        1. List of arrays of dependent variables. Element zero is the
           element for comparison, the last element is the labels for
           the previous elements

        2. Dictionary of additional simulation result
           attributes. Contains at least the key `mean_fn` which is a
           functional representation of the relationship between the
           independent variable and the dependent variable for
           comparison

    - :py:func:`F_UNCLE.Utils.Simulations.Simulations.get_sens`: This
      returns a sensitivity matrix of the simulation response to
      changes in the required model degrees of freedom. The response
      is evaluated at the independent data points of the provided
      `initial_data` and the deltas for each finite difference step
      are evaluated using the `compare` method.

Optimization Methods
....................

The goal of F_UNCLE is to determine the set of model degrees of
freedom which minimizes the error between a set of Experiments and
associated Simulations.

:py:class:`F_UNCLE.Opt.Bayesian.Bayesian`:



"""
