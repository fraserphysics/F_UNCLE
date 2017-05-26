# /usr/bin/pyton
"""

pyStruc.py

Contains the Struc abstract class definition

Authors
-------

- Stephen Andrews (SA)

Revisions
---------

0 -> Initial class creation (10-09-2015)

To Do
-----

- Nothing




"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Allows the unicode type when running in python3
try:
    unicode
except NameError:
    unicode = str


# =========================
# Python Standard Libraries
# =========================
import unittest
import copy
from math import ceil
# =========================
# Python Packages
# =========================

# =========================
# Main Code
# =========================


class Struc(object):
    """Abstract object to contain properties and warnings

    **Attributes**

    Attributes:
        name(str): The name of the object
        def_opts(dict): Default options and bounds
        informs(dict): Important user information prompts
        warns(dict): Optional warnings
        options(dict): The options as set by the user

    **Methods**
    """
    def __init__(self, name, def_opts=None, informs=None, warns=None,
                 *args, **kwargs):
        """Instantiates the structure

        Args:
            name(str): Name of the structure
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Keyword Args:
            def_opts(dict):
                Dictionary of the default options for the structure
                Formatted as follows::

                     {option_name(str):
                      [type(Type), default(num), lower_bound(num),
                       upper_bound(num), unit(str), note(str)]
                     }
            informs(dict): Dictionary of the default informs for the structure
            warns(dict): Dictionary of the warnings for the structure

        Return:
            None

        """

        if isinstance(name, (str, unicode)):
            self.name = name
        else:
            raise TypeError("Structure name must be a string")
        # end

        self.informs = {
            0: "{:} completed successfully:".format(name),
            1: "{:} failed:".format(name),
            2: "{:} error:".format(name)
        }

        if informs is not None:
            self.informs.update(informs)
        # end

        self.warns = {
            0: "{:} warning:".format(name)
        }
        if warns is not None:
            self.warns.update(warns)
        # end

        # ..note:: No checks on default options formatting, see header
        if def_opts is not None:
            self.def_opts = def_opts
        else:
            self.def_opts = {}
        # end

        self.options = {}

        # Sets the options, giving priority to options passed as keyword
        # arguments
        for arg in self.def_opts:
            if arg in kwargs:
                self.set_option(arg, kwargs[arg])
            else:
                self.options[arg] = copy.deepcopy(self.def_opts[arg][1])
            # end
        # end

    def get_option(self, name):
        """Returns the option corresponding the the given name

        Args:
            name(str): Name of the option

        Return:
            Value of the option corresponding to 'name'

        """

        options = self.options

        if name in options:
            opt = options[name]
        else:
            raise KeyError("{:} Unknown option {:}"
                           . format(self.get_inform(1), name))
        # end

        return opt

    def set_option(self, name, value):
        """Sets the option corresponding to the given name to a specified value.

        Enforces the following checks

        1. name is valid
        2. value is of correct type
        3. value is within bounds
        4. **Not Implemented** value has correct units

        Args:
            name(str): Name of the option to set
            value: Value of the option to set

        Return:
            None
        """
        def_opts = self.def_opts

        if name not in def_opts:
            raise KeyError("{:} Unknown option {:}"
                           .format(self.get_inform(1), name))
        # end

        if not len(def_opts[name]) == 6:
            raise IndexError("{:}, option {:} has not been defined with 5"
                             "elements".format(self.get_inform(1), name))
        # Type testing
        # If a float is needed and an int is given, change to a float
        req_type = def_opts[name][0]
        if req_type == str:
            req_type = (str, unicode)
        # end

        if not isinstance(value, req_type):
            if isinstance(value, int) and def_opts[name][0] == float:
                value *= 1.0
            else:
                raise TypeError("{:} Wrong type for option {:}"
                                .format(self.get_inform(1), name))
            # end
        # end

        # Bounds testing, bounds can be (float, None) (None float)
        # or (None None)
        def bounds_test(name, value):
            if def_opts[name][2] is not None and value < def_opts[name][2]:
                raise ValueError("{:} option {:} out of bounds"
                                 .format(self.get_inform(1), name))
            elif def_opts[name][3] is not None and value > def_opts[name][3]:
                raise ValueError("{:} option {:} out of bounds"
                                 .format(self.get_inform(1), name))
            else:
                pass
            # end

        if isinstance(value, (float, int)):
            bounds_test(name, value)
        elif isinstance(value, (list, tuple)):
            for val in value:
                bounds_test(name, val)
            # end
        # end

        # NotImplementedError -> units test

        self.options[name] = value

    def get_inform(self, err_id):
        """Returns an inform corresponding to the error code

        Args:
            err_id(int): Error ID number

        Returns:
            (str): String containing the error message

        """
        informs = self.informs
        if not isinstance(err_id, int):
            raise TypeError('{}: error index must be an integer'
                            .format(self.name))
        elif err_id not in informs:
            raise KeyError('{}: unknown error code {:d}'
                           .format(self.name, err_id))
        else:
            return informs[err_id]
        # end

    def get_warn(self, warn_id):
        """Returns an inform corresponding to the warning code

        Args:
            warn_id(int): Warning ID number

        Return:
            (str): String containing the warning message

        """

        warns = self.warns
        if not isinstance(warn_id, int):
            raise TypeError('{}: warning index must be an integer'
                            .format(self.name))
        elif warn_id not in warns:
            raise KeyError('{}: unknown warning code {:d}'
                           .format(self.name, warn_id))
        else:
            return warns[warn_id]
        # end

    def __str__(self, inner=False, *args, **kwargs):
        """Returns a string representation of the object

        Args:
            inner(bool): Flag if the string is being called within another
                         string function
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Return:
            (str): A string describing the object
        """
        def_opts = self.def_opts
        opts = self.options
        out_str = '\n'
        # if not inner: out_str += "="*80 + "\n"
        out_str += "{:^80s}\n".format(self.name)
        out_str += "{:^80s}\n".format("^" * len(self.name))
        out_str += "Options\n"
        out_str += "-------\n"
        out_str += "{:<15s}{:<10s}{:12s}{:<10s}{:<10s}{:<23s}\n"\
                   .format("Name", "Value", "Units", " Min",
                           " Max", "Description")
        out_str += "{:<15s}{:<10s}{:12s}{:<10s}{:<10s}{:<23s}\n"\
                   .format("....", ".....", ".....", " ...",
                           " ...", "...........")
        for key in def_opts:
            if def_opts[key][2] is None:
                lower_bound = float('nan')
            else:
                lower_bound = def_opts[key][2]
            # end

            if def_opts[key][3] is None:
                upper_bound = float('nan')
            else:
                upper_bound = def_opts[key][3]
            # end

            # Divide long descriptions to span many lines
            try:
                descr = []
                for i in range(int(ceil(len(def_opts[key][5]) / 23.0))):
                    descr.append(def_opts[key][5][(i) * 23:(i + 1) * 23])
                # end
            except Exception as inst:
                raise inst
            # end

            if isinstance(opts[key], (float, int)):
                out_str += "{:<15s}{:< 10g}{:12s}{:< 10g}{:< 10g}{:<23s}\n".\
                           format(key, opts[key], def_opts[key][4], lower_bound,
                                  upper_bound, descr[0])
                if len(descr) > 1:
                    for line in descr[1:]:
                        out_str += " " * 57 + line + "\n"
                    # end
                # end
            elif isinstance(opts[key], str):
                out_str += "{:<15s}{:<10s}{:12s}{:< 10g}{:< 10g}{:<23s}\n".\
                           format(key, opts[key], def_opts[key][4], lower_bound,
                                  upper_bound, descr[0])
                if len(descr) > 1:
                    for line in descr[1:]:
                        out_str += " " * 57 + line + "\n"
                    # end
                # end
            elif isinstance(opts[key], (tuple, list)) and len(opts[key]) > 0:
                # Print out lists

                # Print first row

                if isinstance(opts[key][0], (float, int)):
                    out_str += "{:<15s}{:< 10g}{:12s}{:< 10g}{:< 10g}{:<23s}\n"\
                               .format(key, opts[key][0],
                                       def_opts[key][4], lower_bound,
                                       upper_bound, descr[0])
                else:
                    out_str += "{:<15s}{:<10s}{:12s}{:< 10g}{:< 10g}{:<23s}\n"\
                               .format(key, opts[key][0],
                                       def_opts[key][4], lower_bound,
                                       upper_bound, descr[0])
                # end

                # Print following rows
                if len(opts[key]) > 1:
                    for i in range(len(opts[key]) - 1):
                        # Accounts for multi-line descriptions
                        if (i + 1) < len(descr):
                            descr_line = descr[i + 1]
                        else:
                            descr_line = ""
                        # end
                        if isinstance(opts[key][i + 1], (int, float)):
                            out_str += "{:<15s}{:< 10g}{:32s}{:<23s}\n"\
                                       .format('', opts[key][i + 1], '',
                                               descr_line)
                        else:
                            out_str += "{:<15s}{:<10s}{:32s}{:<23s}\n"\
                                       .format('', opts[key][i + 1], '',
                                               descr_line)
                        # end
                    # end
                # end
                # Print extra lines if the description takes more lines
                # than the option
                if len(opts[key]) < len(descr):
                    for i in range(len(descr) - len(opts[key])):
                        out_str += " " * 57 + descr[len(opts[key]) + i] + "\n"
                    # end
                # end
            # end
        # end
        out_str += self._on_str(*args, **kwargs)
        # if not inner: out_str += "="*80 + "\n"
        return out_str

    def _on_str(self, *args, **kwargs):
        """Print methods of children

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Return:
            (str): A string representing the object
        """

        return ''

    def write_tex_var(self, name, value, units):
        """Returns a string which will define a variable in latex
        
        Args:

            name(str): The variable name, do not include the leading slash
            value(str): The string representation of the number
            units(rstr): A *RAW* sting giving the correct latex syntax for the
                         units
        """
        if units is None:
            return '\\newcommand{{\\{:s}}}'\
                '{{\\num{{{:s}}}}}\n'\
                .format(name, value)
        else:
            return '\\newcommand{{\\{:s}}}'\
                '{{\\SI{{{:s}}}{{{:s}}}}}\n'\
                .format(name, value, units)

        
    def plot(self, axes=None, fig=None, linestyles=[], labels=[]):
        """Plots the object

        Args:
            axes(plt.Axes): The Axes on which to plot the figure, if None,
                creates a new figure object on which to plot.
            fig(plt.Figure): The Figure on which to plot, cannot specify
                figure and axes
            linstyles(list): Strings for the linestlyes
            labels(list): Strings for the plot labels

        Return:
            (plt.Figure): A reference to the figure containing the plot

        """

        raise NotImplementedError("Plotting not defined")

    def write_to_file(self, filename):
        """Writes the object to a file

        Args:
            filename(string): A path to a writable location

        Return:
            None

        """
        out_str = str(self)

        try:
            with open(filename, 'r') as fid:
                fid.write(out_str)
            # end
        except IOError:
            raise IOError("{:} Could not write to file {:}"
                          .format(self.get_inform(1), filename))
        # end
