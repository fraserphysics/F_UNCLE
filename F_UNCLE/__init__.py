"""The FUNCLE module

Functional UNcertainty Constrained by Law and Experiment

Authors
-------

 - Andrew Fraser (AF)
 - Stephen A Andrews (SA)

Revision History
----------------

0.0 - Initial class creation (09-10-2015)

License
-------

Directories
-----------

The root project directory has the following sub-directories:

docs
   Files for making the documentation

F_UNCLE
   Code described in this documentation

reports
   Latex source for papers and presentations based on F_UNCLE

test
   Vacuous root for nosetest

The F_UNCLE directory has the following sub-directories:

examples
   Scripts that make figures for documents in root/reports
   
Experiments
   Code to simulate toy experiments such as: Cylinder, Gun, Sphere and
   Stick with classes that are sub-classes of classes defined in the
   Models and Utils directories

Models
   Scripts that define particular "Physics Models" that inherit
   from Utils/PhysicsModel.py

Opt
   Code for optimizing a posteriori probability and reporting Fisher
   information

Utils
   Scripts with base classes for Experiments and Models
   

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
