# /usr/bin/pyton
"""

test_Utils.py

Test classes for misc utils

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
# =========================
# Python Packages
# =========================

# =========================
# External Packages
# =========================
if __name__ == '__main__':
    sys.path.append(os.path.abspath('./../../'))
    from F_UNCLE.Utils.Struc import Struc
    from F_UNCLE.Utils.Container import Container
else:
    from .Struc import Struc
    from .Container import Container
# end


class TestObject(unittest.TestCase):
    """
    Test of the Struc object
    """
    def setUp(self):
        self.def_opts = {
            'apples': [int, 1, 0, 10, '-', 'The number of apples'],
            'bears': [int, 2, 0, None, '-', 'The number of bears'],
            'canaries': [float, 0.1, -0.3, 0.25, 'ft/s',
                         'The number of canaries. This is also a test of very'
                         'long titles'],
            'ducks': [float, 0.0, None, None, '-', 'Ducks are unlimited'],
            'list_test': [list, [1, 2, 3, 4, 5], 0, 6, '-', 'Short title list'],
            'list_test3': [list, ['string', 'string2'], None, None, '-',
                           'Test of list with a title which wraps more lines '
                           'than the list is long, add some more text to be'
                           'safe'],
            'list_test2': [list, [1, 2, 3, 4, 5], None, None, '-',
                           'Short title list with a very very long title which'
                           'will wrap several lines']
        }
        self.name = "Test Structure Object"

    def test_standard_instantiation(self):
        """Test normal usage
        """

        my_struc = Struc(self.name, def_opts=self.def_opts)
        print('')
        print(my_struc)
        self.assertIsInstance(my_struc, Struc)

    def test_inst_bad_option(self):
        """Structure should ignore the unknown option "potatoes"
        """

        my_struc = Struc(self.name, def_opts=self.def_opts, potatoes=5)
        self.assertIsInstance(my_struc, Struc)

    def test_bad_set_option(self):
        """Structure should raise a KeyError when given an unknown option
        """
        my_struc = Struc(self.name, def_opts=self.def_opts)

        with self.assertRaises(KeyError):
            my_struc.set_option("potatoes", 5)
        # end

    def test_inst_bad_type(self):
        """Structure should raise a type error when int is set to a float
        """

        with self.assertRaises(TypeError):
            Struc(self.name, def_opts=self.def_opts, apples=1.0)
        # end

    def test_list_set_option(self):
        """Structure should raise a value error if list item set above bound
        """

        my_struc = Struc(self.name, def_opts=self.def_opts)

        my_struc.set_option('list_test', [0, 1, 2])

        with self.assertRaises(ValueError):
            my_struc.set_option('list_test', [-1, 1, 2])
        # end

        my_struc.set_option('list_test', [1, 2, 6])
        with self.assertRaises(ValueError):
            my_struc.set_option('list_test', [1, 2, 7])
        # end

    def test_inst_over_bound(self):
        """Should raise a value error if apples is set above its bounds
        """

        with self.assertRaises(ValueError):
            Struc(self.name, def_opts=self.def_opts, apples=11)
        # end

        my_struc = Struc(self.name, def_opts=self.def_opts)

        with self.assertRaises(ValueError):
            my_struc.set_option('apples', 11)
        # end

    def test_inst_at_bounds(self):
        """Structure should accept a value *at* the upper and lower bound
        for ints and floats
        """

        Struc(self.name, def_opts=self.def_opts, apples=10)
        Struc(self.name, def_opts=self.def_opts, apples=0)
        Struc(self.name, def_opts=self.def_opts, canaries=-0.3)
        Struc(self.name, def_opts=self.def_opts, canaries=0.25)

    def test_inst_under_bound(self):
        """Structure should raise value error if apples is below its bounds
        """

        with self.assertRaises(ValueError):
            Struc(self.name, def_opts=self.def_opts, apples=-1)
        # end

    def test_no_bounds(self):
        """Tests the bounds of options which are unbounded.
        Should be able to be set to very large or small values

        """
        my_struc = Struc(self.name, def_opts=self.def_opts, ducks=-1E21)
        my_struc = Struc(self.name, def_opts=self.def_opts, ducks=1E21)

    def test_get_option(self):
        """get_option should return the default value if in a vanilla
        instantiation
        """
        my_struc = Struc(self.name, def_opts=self.def_opts)

        self.assertEqual(my_struc.get_option('apples'),
                         self.def_opts['apples'][1])

    def test_bad_get_option(self):
        """get_option should raise a KeyError if an invalid name is given
        """

        my_struc = Struc(self.name, def_opts=self.def_opts)

        with self.assertRaises(KeyError):
            my_struc.get_option('potato')

    def test_get_inform(self):
        """get_inform should return a given string for error code 0
        """

        my_struc = Struc(self.name, def_opts=self.def_opts)

        self.assertEqual(my_struc.get_inform(0),
                         "Test Structure Object completed successfully:")

    def test_bad_get_inform(self):
        """get_inform  should raise an error if the index is not an int
        or is out of bounds
        """

        my_struc = Struc(self.name, def_opts=self.def_opts)

        with self.assertRaises(TypeError):
            my_struc.get_inform('one')
        # end

        with self.assertRaises(KeyError):
            my_struc.get_inform(3)
        # end

    def test_get_warn(self):
        """get_warn should return a given string for error code 0
        """

        my_struc = Struc(self.name, def_opts=self.def_opts)

        self.assertEqual(my_struc.get_warn(0), "Test Structure Object warning:")

    def test_bad_get_warn(self):
        """get_warn  should raise an error if the index is not an int or is out
        of bounds
        """

        my_struc = Struc(self.name, def_opts=self.def_opts)

        with self.assertRaises(TypeError):
            my_struc.get_warn('one')
        # end

        with self.assertRaises(KeyError):
            my_struc.get_warn(2)
        # end
    # end


class TestContainer(unittest.TestCase):
    """

    Test of the container class

    """

    def setUp(self):
        self.my_container = Container(name="Test Container")

        self.a = [1, 2, 3, 4]
        self.b = [5, 6, 7, 7]
        self.c = [7, 8, 9]
        self.d = "Non-homogeneous-input"

    def test_set_get_object(self):
        """

        Tests setting an object in the container and getting it back

        """
        my_cont = self.my_container

        my_cont[0] = self.a

        self.assertListEqual(my_cont[0], self.a)

    def test_bad_set_object(self):
        """

        Tests setting an object to an invalid index

        """
        my_cont = self.my_container

        with self.assertRaises(TypeError):
            my_cont[0.0] = self.a
        # end

        with self.assertRaises(TypeError):
            my_cont['zero'] = self.a
        # end

    def test_bad_get_object(self):
        """

        Tests getting an invalid index

        """
        my_cont = self.my_container

        my_cont[0] = self.a

        with self.assertRaises(TypeError):
            tmp = my_cont[0.0]
        # end

        with self.assertRaises(TypeError):
            tmp = my_cont['zero']
        # end

        with self.assertRaises(KeyError):
            tmp = my_cont[1]
        # end

    def test_del_object(self):
        """

        Tests that an object was deleted.

        """

        my_cont = self.my_container

        my_cont[0] = self.a
        my_cont[1] = self.b
        my_cont[2] = self.c

        del my_cont[1]

        self.assertListEqual(my_cont[0], self.a)
        self.assertListEqual(my_cont[2], self.c)

        with self.assertRaises(KeyError):
            tmp = my_cont[1]
        # end

    def test_bad_del_object(self):
        """

        Tests deleting an invalid object

        """

        my_cont = self.my_container

        my_cont[0] = self.a

        with self.assertRaises(TypeError):
            del my_cont[0.0]
        # end

        with self.assertRaises(TypeError):
            del my_cont['zero']
        # end

        with self.assertRaises(KeyError):
            del my_cont[1]
        # end

    def test_append_to_null(self):
        """

        Tests appending to an empty container

        """

        my_cont = self.my_container

        my_cont.append(self.a)

        self.assertListEqual(self.a, my_cont[0])

    def test_append_to_pop(self):
        """

        Tests appending to a populated container

        """

        my_cont = self.my_container

        my_cont.append(self.a)
        my_cont.append(self.b)
        my_cont.append(self.c)

        self.assertListEqual(my_cont[0], self.a)
        self.assertListEqual(my_cont[1], self.b)
        self.assertListEqual(my_cont[2], self.c)

    def test_append_to_holy_container(self):
        """

        Tests appending to a container where an index has been deleted.
        Should append after the last index

        """
        my_cont = self.my_container

        my_cont.append(self.a)
        my_cont.append(self.b)
        my_cont.append(self.c)

        self.assertListEqual(my_cont[0], self.a)
        self.assertListEqual(my_cont[1], self.b)
        self.assertListEqual(my_cont[2], self.c)

        del my_cont[1]

        self.assertListEqual(my_cont[0], self.a)
        self.assertListEqual(my_cont[2], self.c)

        with self.assertRaises(KeyError):
            tmp = my_cont[1]

    def test_get_len(self):
        """

        Tests the len function, ensures it updates after a delete

        """
        my_cont = self.my_container

        my_cont.append(self.a)
        my_cont.append(self.b)
        my_cont.append(self.c)

        self.assertEqual(len(my_cont), 3)

        del my_cont[1]

        self.assertEqual(len(my_cont), 2)

    def test_iterable(self):
        """

        Tests the iterable generation

        """

        my_cont = self.my_container

        my_cont.append(self.a)
        my_cont.append(self.b)
        my_cont.append(self.c)
        my_cont.append(self.c)

        del my_cont[2]

        good_list = [self.a, self.b, self.c]
        k = 0
        for key in my_cont:
            self.assertListEqual(my_cont[key], good_list[k])
            k += 1
        # end
# end

if __name__ == '__main__':
    unittest.main(verbosity=4)
# end
