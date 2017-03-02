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

   The mathematical representation of a physics model is dependant on
   a set of variables whose values can alter the behaviour of the
   model. These variables are refered to as the degrees of freedom or
   DOF

Experiment

   An experiment represents data measured from a real physical
   experiments. Experiments used in F_UNCLE are chosen so that thier
   results are dominated by a single physics process which can be
   represented by a physics model Simulation

Simulation

   A simulation which represents the physical system from which the
   experimental data were obtained. The simulation is dependent on one
   or more physics models and acts as a map between the model degree
   of freedom space and the experimental result space




F_UNCLE Parent Classes
----------------------

These concepts are implemented in the following three fundemental
F_UNCLE classes

Physics Model
.............

:pyclass:`F_UNCLE.Utils.PhysicsModel.PhysicsModel`. This class
represents a physics model. The degrees of freedom of a `PhysicsModel`
are immutable. Setting the degrees of freedom of a `PhysicsModel` will
return a independent copy of itself with updated DOF.

 - :pymethod:`F_UNCLE.Utils.PhysicsModel.PhysicsModel.get_dof`. Returns
   a list of the model degrees of freedom

 - :pymethod:`F_UNCLE.Utils.PhysicsModel.PhysicsModel.set_dof`. Generates
   a new Physics Model with the specified degrees of freedom

 - :pymethod:`F_UNCLE.Utils.PhyscisModel.PhysicsModel.get_sigma`. Returns
   a square matrix representingthe covariance of the model degrees of
   freedom

 - :pymethod:`F_UNCLE.Utils.PhyscisModel.PhysicsModel.shape`. Returns
   an integer representing the number of model degrees of freedom

Experiment
..........

:pyclass:`F_UNCLE.Utils.Experiment.Experiment`. This class represents
the results for a single experiment performed under a single set of
conditions.

- :pymethod:`F_UNCLE.Utils.Experiment.Experiment.__call__`: Returns
  the 

  0. The independent varialbe of the simulation (i.e. time)

  1. List of arrays of dependant variables. Element zero is the
     element for comparisson, the last element is the labels for the
     previous elements

  2. Dictionary of additonal simulation result attribures. Contains at
     least the key `mean_fn` which is a functional representation of
     the relationship between the independant variable and the
     dependent variable for comparisson


Sumulation
..........

:pyclass:`F_UNCLE.Utils.Simulation.Simulation`. This class wraps some
computer simulatin which is dependent on a PhysicsModel

- :pymethod:`F_UNCLE.Utils.Simulation.Simulation.compare`: Compares
  two simulations results and returns the difference

- :pymethod:`F_UNCLE.Utils.Simulations.Simulations.__call__`: Returns
  the results of the simulation as a tuple

   0. The independent varialbe of the simulation (i.e. time)

  1. List of arrays of dependant variables. Element zero is the
     element for comparisson, the last element is the labels for the
     previous elements

  2. Dictionary of additonal simulation result attribures. Contains at
     least the key `mean_fn` which is a functional representation of
     the relationship between the independant variable and the
     dependent variable for comparisson

     

"""

# from pyStruc import Struc
# from pyContainer import Container
# from pyIsentrope import Isentrope
# from pyIsentrope import EOSBump
# from pyIsentrope import EOSModel
# from pyIsentrope import Spline
# from pyExperiment import Experiment

# __all__ = ['Struc', 'Container']
# def start():
#    """This starts the module
#    """


###---------------
### Local Variables:
### mode: rst
### End:
