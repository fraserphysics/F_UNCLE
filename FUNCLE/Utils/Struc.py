#/usr/bin/pyton
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

        if isinstance(name, str):
            self.name = name
        else:
            raise TypeError("Structure name must be a string")
        #end

        self.informs = {
            0 : "{:} completed successfully:".format(name),
            1 : "{:} failed:".format(name),
            2 : "{:} error:".format(name)
            }

        if not informs is None:
            self.informs.update(informs)
        #end

        self.warns = {
            0 : "{:} warning:".format(name)
            }
        if not warns is None:
            self.warns.update(warns)
        #end

        # ..note:: No checks on default options formatting, see header
        if not def_opts is None:
            self.def_opts = def_opts
        else:
            self.def_opts = {}
        #end

        self.options = {}

        # Sets the options, giving priority to options passed as keyword
        # arguments
        for arg in self.def_opts:
            if arg in kwargs:
                self.set_option(arg, kwargs[arg])
            else:
                self.options[arg] = copy.deepcopy(self.def_opts[arg][1])
            #end
        #end

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
            raise KeyError("{:} Unknown option {:}".\
                           format(self.get_inform(1), name))
        #end

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

        if not name in def_opts:
            raise KeyError("{:} Unknown option {:}".\
                           format(self.get_inform(1), name))
        #end

        # Type testing
        # If a float is needed and an int is given, change to a float
        if not isinstance(value, def_opts[name][0]):
            if isinstance(value, int) and def_opts[name][0] == float:
                value *= 1.0
            else:
                raise TypeError("{:} Wrong type for option {:}".\
                                format(self.get_inform(1), name))
            #end
        #end

        # Bounds testing, bounds can be (float, None) (None float)
        # or (None None)
        if isinstance(value, (float, int)):
            if not def_opts[name][2] is None  and value < def_opts[name][2]:
                raise ValueError("{:} option {:} out of bounds".\
                                 format(self.get_inform(1), name))
            elif not def_opts[name][3] is None and value > def_opts[name][3]:
                raise ValueError("{:} option {:} out of bounds".\
                                 format(self.get_inform(1), name))
            else:
                pass
            #end
        if isinstance(value, (list, tuple)):
            for i in xrange(len(value)):
                if not isinstance(value[i], (float, int)):
                    continue
                elif not def_opts[name][2] is None and value[i] < def_opts[name][2]:
                    raise ValueError("{:} option {:} out of bounds".\
                                     format(self.get_inform(1), name))
                elif not def_opts[name][3] is None and value[i] > def_opts[name][3]:
                    raise ValueError("{:} option {:} out of bounds".\
                                     format(self.get_inform(1), name))
                else:
                    pass
                #end
        #end

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
            raise TypeError('{}: error index must be an integer'.\
                            format(self.name))
        elif not err_id in informs:
            raise KeyError('{}: unknown error code {:d}'.\
                           format(self.name, err_id))
        else:
            return informs[err_id]
        #end

    def get_warn(self, warn_id):
        """Returns an inform corresponding to the warning code

        Args:
            warn_id(int): Warning ID number

        Return:
            (str): String containing the warning message

        """

        warns = self.warns
        if not isinstance(warn_id, int):
            raise TypeError('{}: warning index must be an integer'.\
                            format(self.name))
        elif not warn_id in warns:
            raise KeyError('{}: unknown warning code {:d}'.\
                           format(self.name, warn_id))
        else:
            return warns[warn_id]
        #end

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
        out_str += "{:^80s}\n".format("^"*len(self.name))
        out_str += "Options\n"
        out_str += "-------\n"
        out_str += "{:<15s}{:<10s}{:12s}{:<10s}{:<10s}{:<23s}\n".\
                    format("Name", "Value", "Units", " Min", " Max", "Description")
        out_str += "{:<15s}{:<10s}{:12s}{:<10s}{:<10s}{:<23s}\n".\
                    format("....", ".....", ".....", " ...", " ...", "...........")
        for key in def_opts:
            
            if def_opts[key][2] is None:
                lower_bound = float('nan')
            else:
                lower_bound = def_opts[key][2]
            #end

            if def_opts[key][3] is None:
                upper_bound = float('nan')
            else:
                upper_bound = def_opts[key][3]
            #end

            ## Divide long descriptions to span many lines
            try:
                descr = []
                for i in xrange(int(ceil(len(def_opts[key][5])/23.0))):
                    descr.append(def_opts[key][5][(i)*23:(i+1)*23])
                #end
            except:
                import pdb
                pdb.set_trace()
                
            if isinstance(opts[key], (float, int)):
                out_str += "{:<15s}{:< 10g}{:12s}{:< 10g}{:< 10g}{:<23s}\n".\
                           format(key, opts[key], def_opts[key][4], lower_bound,
                                  upper_bound, descr[0])
                if len(descr) > 1:
                    for line in descr[1:]:
                        out_str += " "*57 + line + "\n"
                    #end
                #end
            elif isinstance(opts[key], str):
                out_str += "{:<15s}{:<10s}{:12s}{:< 10g}{:< 10g}{:<23s}\n".\
                           format(key, opts[key], def_opts[key][4], lower_bound,
                                  upper_bound, descr[0])
                if len(descr) > 1:
                    for line in descr[1:]:
                        out_str += " "*57 + line + "\n"
                    #end
                #end
            elif isinstance(opts[key], (tuple, list)) and len(opts[key])>0:
                # Print out lists

                # Print first row

                if isinstance(opts[key][0], (float, int)):
                    out_str += "{:<15s}{:< 10g}{:12s}{:< 10g}{:< 10g}{:<23s}\n".\
                                format(key, opts[key][0],
                                       def_opts[key][4], lower_bound,
                                       upper_bound, descr[0])
                else:
                    out_str += "{:<15s}{:<10s}{:12s}{:< 10g}{:< 10g}{:<23s}\n".\
                                format(key, opts[key][0],
                                       def_opts[key][4], lower_bound,
                                       upper_bound, descr[0])
                #end

                # Print following rows
                if len(opts[key]) > 1:
                    for i in xrange(len(opts[key])-1):
                        # Accounts for multi-line descriptions
                        if (i+1) < len(descr):
                            descr_line = descr[i+1]
                        else:
                            descr_line = ""
                        #end
                        if isinstance(opts[key][i+1], (int, float)):
                            out_str += "{:<15s}{:< 10g}{:32s}{:<23s}\n".\
                                       format('', opts[key][i+1], '', descr_line)
                        else:
                            out_str += "{:<15s}{:<10s}{:32s}{:<23s}\n".\
                                       format('', opts[key][i+1], '', descr_line)
                        #end
                    #end
                #end
                # Print extra lines if the description takes more lines
                # than the option
                if len(opts[key]) < len(descr):
                    for i in xrange(len(descr)-len(opts[key])):
                        out_str += " "*57 + descr[len(opts[key])+i] + "\n"
                    #end
                #end
            #end
        #end
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

    def plot(self, axis=None, hardcopy=None):
        """Returns creates

        Args:
            axis(plt.Axes): The axis on which to plot the figure, if None,
                creates a new figure object on which to plot.
            hard-copy(bool): If a string, write the figure to the file specified

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
            #end
        except IOError:
            raise IOError("{:} Could not write to file {:}".\
                          format(self.get_inform(1), filename))
        #end

class TestObject(unittest.TestCase):
    """
    Test of the Struc object
    """
    def setUp(self):
        self.def_opts = {
            'apples':[int, 1, 0, 10, '-', 'The number of apples'],
            'bears':[int, 2, 0, None, '-', 'The number of bears'],
            'canaries':[float, 0.1, -0.3, 0.25, 'ft/s',
                        'The number of canaries. This is also a test of very'+\
                        ' long titles'],
            'ducks':[float, 0.0, None, None, '-', 'Ducks are unlimited'],
            'list_test':[list, [1, 2, 3, 4, 5], 0, 6, '-', 'Short title list'],
            'list_test3':[list, ['string', 'string2'], None, None, '-',
                          'Test of list with a title which wraps more lines'+\
                          ' than the list is long, add some more text to be safe'],
            'list_test2':[list, [1, 2, 3, 4, 5], None, None, '-',
                          'Short title list with a very very long title which'+\
                          ' will wrap several lines']
            }
        self.name = "Test Structure Object"


    def test_standard_instantiation(self):
        """

        Test normal usage

        """

        my_struc = Struc(self.name, def_opts=self.def_opts)
        print ''
        print my_struc
        self.assertIsInstance(my_struc, Struc)

    def test_inst_bad_option(self):
        """

        Structure should ignore the unknown option "potatoes"

        """

        my_struc = Struc(self.name, def_opts=self.def_opts, potatoes=5)
        self.assertIsInstance(my_struc, Struc)

    def test_bad_set_option(self):
        """

        Structure should raise a KeyError when given an unknown option

        """
        my_struc = Struc(self.name, def_opts=self.def_opts)

        with self.assertRaises(KeyError):
            my_struc.set_option("potatoes", 5)
        #end

    def test_inst_bad_type(self):
        """

        Structure should raise a type error when int apples is set to a float

        """

        with self.assertRaises(TypeError):
            Struc(self.name, def_opts=self.def_opts, apples=1.0)
        #end

    def test_list_set_option(self):
        """Structure should raise a value error if list item set above bound
        """

        my_struc = Struc(self.name, def_opts=self.def_opts)

        my_struc.set_option('list_test', [0, 1, 2])

        with self.assertRaises(ValueError):
            my_struc.set_option('list_test', [-1, 1, 2])
        #end

        my_struc.set_option('list_test', [1, 2, 6])
        with self.assertRaises(ValueError):
            my_struc.set_option('list_test', [1, 2, 7])
        #end

    def test_inst_over_bound(self):
        """

        Structure should raise a value error if apples is set above its bounds

        """

        with self.assertRaises(ValueError):
            Struc(self.name, def_opts=self.def_opts, apples=11)
        #end

        my_struc = Struc(self.name, def_opts=self.def_opts)

        with self.assertRaises(ValueError):
            my_struc.set_option('apples', 11)
        #end

    def test_inst_at_bounds(self):
        """

        Structure should accept a value *at* the upper and lower bound for ints
        and floats

        """

        Struc(self.name, def_opts=self.def_opts, apples=10)
        Struc(self.name, def_opts=self.def_opts, apples=0)
        Struc(self.name, def_opts=self.def_opts, canaries=-0.3)
        Struc(self.name, def_opts=self.def_opts, canaries=0.25)


    def test_inst_under_bound(self):
        """

        Structure should raise value error if apples is below its bounds

        """

        with self.assertRaises(ValueError):
            Struc(self.name, def_opts=self.def_opts, apples=-1)
        #end

    def test_no_bounds(self):
        """

        Tests the bounds of options which are unbounded. Should be able to be
        set to very large or small values

        """
        my_struc = Struc(self.name, def_opts=self.def_opts, ducks=-1E21)
        my_struc = Struc(self.name, def_opts=self.def_opts, ducks=1E21)


    def test_get_option(self):
        """

        get_option should return the default value if in a vanilla instantiation

        """
        my_struc = Struc(self.name, def_opts=self.def_opts)

        self.assertEqual(my_struc.get_option('apples'),
                         self.def_opts['apples'][1])


    def test_bad_get_option(self):
        """

        get_option should raise a KeyError if an invalid name is given

        """

        my_struc = Struc(self.name, def_opts=self.def_opts)

        with self.assertRaises(KeyError):
            my_struc.get_option('potato')


    def test_get_inform(self):
        """

        get_inform should return a given string for error code 0

        """

        my_struc = Struc(self.name, def_opts=self.def_opts)

        self.assertEqual(my_struc.get_inform(0),
                         "Test Structure Object completed successfully:")


    def test_bad_get_inform(self):
        """

        get_inform  should raise an error if the index is not an int or is out
        of bounds

        """
        my_struc = Struc(self.name, def_opts=self.def_opts)

        with self.assertRaises(TypeError):
            my_struc.get_inform('one')
        #end

        with self.assertRaises(KeyError):
            my_struc.get_inform(3)
        #end

    def test_get_warn(self):
        """

        get_warn should return a given string for error code 0

        """

        my_struc = Struc(self.name, def_opts=self.def_opts)

        self.assertEqual(my_struc.get_warn(0), "Test Structure Object warning:")


    def test_bad_get_warn(self):
        """

        get_warn  should raise an error if the index is not an int or is out
        of bounds

        """
        my_struc = Struc(self.name, def_opts=self.def_opts)

        with self.assertRaises(TypeError):
            my_struc.get_warn('one')
        #end

        with self.assertRaises(KeyError):
            my_struc.get_warn(2)
        #end


    #end

if __name__ == '__main__':
    unittest.main(verbosity=4)
