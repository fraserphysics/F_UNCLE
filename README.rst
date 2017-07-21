
Functional Uncertainty Constrained by Law and Experiment
========================================================

.. image:: https://travis-ci.org/fraserphysics/F_UNCLE.svg
   :alt: travis CI build status
   :target: https://travis-ci.org/fraserphysics/F_UNCLE/

.. image:: https://readthedocs.org/projects/f-uncle/badge/?version=latest
   :alt: readthedocs status
   :target: http://f-uncle.readthedocs.io/en/latest/

Prerequisites
-------------

To run the tests, certain software must be installed.  

The minimum required python packages to run F_UNCLE are shown in the
file /pip_req.txt. These packages can either be installed manually or
the conda package manager can be used to create an environment
containing all the required packages. In the root F_UNCLE folder run
the following commands::

   conda create -n funcle python=3.5 --file pip_req.txt
   source activate funcle

The conda environment will now be active and provide all the packages
needed to run F_UNCLE

To run the tests, several other packages are needed, they are given in
`pip_test_req.txt`. These packages can either be installed manually or
added to the conda environment with the following commands from the
root F_UNCLE directory::

    source activate funcle
    conda install --file pip_test_req.txt

To build the documentation, further packages are needed, they are
given in `pip_doc_req.txt` and can be installed as above

Testing
-------

From the root directory of `F_UNCLE` run::

   pytest F_UNCLE

This will discover, run and report on all test in the F_UNCLE module

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
   
