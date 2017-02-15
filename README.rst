
Functional Uncertainty Constrained by Law and Experiment
========================================================

.. image:: https://travis-ci.org/steve855/F_UNCLE.svg
   :alt: travis CI build status
   :target: https://travis-ci.org/steve855/F_UNCLE/

.. image:: https://readthedocs.org/projects/f-uncle/badge/?version=latest
   :alt: readthedocs status
   :target: http://f-uncle.readthedocs.io/en/latest/

Prerequisites
-------------

To run the tests, certain software must be installed.  The required
packages are described in the file foo.  Using the conda package
manager, one can create an environment by foobar.

Testing
-------

From the root directory of `F_UNCLE` run:

   $ nosetests -all-modules

Documentation
-------------

Documentation is available on `readthedocs
<http://f-uncle.readthedocs.io/en/latest/>`_

Los Alamos National Laboratory (LANL) Release Information
---------------------------------------------------------

The initial version of this code was written at LANL.  It was released
in 2016 with Los Alamos Computer Code index LA-CC-16-034 and the
following description:

The F_UNCLE project is for quantitatively characterizing uncertainty
about functions and how physical laws and experiments constrain that
uncertainty.  The project is an evolving platform consisting of
program and text files that numerically demonstrate and explore
quantitative characterizations of uncertainty about functions and how
either historical or proposed experiments have or could constrain that
uncertainty.  No actual physical measurements are distributed as part
of the software.

Authors
-------

- Stephen Andrews
- Andrew Fraser

License
-------

F_UNCLE: Tools for understanding functional uncertainty
Copyright (C) 2016 Andrew M. Fraser and Stephen A. Andrews

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Files and Directories
---------------------

The root project directory has the following files and sub-directories:

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

__init__.py
   File that enables importing directory as module and contains text
   for documentation
   