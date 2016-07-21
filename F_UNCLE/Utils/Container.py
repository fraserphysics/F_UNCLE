#/usr/bin/pyton
"""

container.pt

Contains the container abstract class definition

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
else:
    from .Struc import Struc
#end

# =========================
# Main Code
# =========================

class Container(Struc):
    """An abstract iterable container object

    Attributes:
        _contents(dict): An integer keyed list. Do not access this list directly
            use the iterable functions
    
    .. note:: The container does not fill in the gaps when an object is delete, i. e.
       if the container contained indices 1,2 and 3 and index 2 was deleted,
       the object would then contain index 1 and 3. If an object were then ap-
       pended to the list it would have index 4.

    """

    def __init__(self, name, def_opts=None, informs=None, warns=None,
                 *args, **kwargs):
        """Instantiates the integer indexed container

        Args:
            name(str): Name of the container
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Keyword Args:
            def_opts(dict): Dictionary of the default options for the container
            informs(dict): Dictionary of the default informs for the container
            warns(dict): Dictionary of the warnings for the container

        Return:
            None
        """

        Struc.__init__(self, name, def_opts=def_opts, informs=informs,
                       warns=warns, *args, **kwargs)

        self._contents = {}

    def clear(self):
        """Deletes all the container contents

        """

        self._contents = {}

    def append(self, value):
        """Appends the data to the end of contents

        """

        maxi = -1
        for i in self._contents.keys():
            if i > maxi:
                maxi = i
            #end
        #end

        self.__setitem__(maxi + 1, value)

    def __getitem__(self, i):
        """ Returns the data in the container at index i

        Args:
            i(int): Index

        Return:
            Container contents at index i
        """

        if not isinstance(i, int):
            raise TypeError("{:} Container keys must be integers".\
                            format(self.get_inform(1)))
        elif not i in self._contents:
            raise KeyError("{:} Container does not contain key {:d}".\
                           format(self.get_inform(1), i))
        else:
            return self._contents[i]
        #end


    def __setitem__(self, i, value):
        """Sets the data in the container at index i

        Args:
            i(int): Index
            value: Value to give index i in container

        Return:
            None
        """
        if not isinstance(i, int):
            raise TypeError("{:} Container keys must be integers".\
                            format(self.get_inform(1)))
        else:
            value = self._on_setitem(i, value)
            self._contents[i] = value
        #end

    def _on_setitem(self, i, value):
        """Overloaded method to perform instance specific checks

        Occurs before adding an item to contents

        Args:
            i(int): Index
            value: Value to give index i in container

        Return:
            None
        """

        return value

    def __delitem__(self, i):
        """Deletes the data in the container at index i

        Args:
            i(int): Index

        Return:
            None
        """
        if not isinstance(i, int):
            raise TypeError("{:} Container keys must be integers".\
                            format(self.get_inform(1)))
        elif not i in self._contents:
            raise KeyError("{:} Container does not contain key {:d}".\
                           format(self.get_inform(1), i))
        else:
            del self._contents[i]
        #end

    def __iter__(self):
        """ Returns an iterable for the contents of the container

        Return:
            (iterable): An iterable of the container
        """

        return iter(self._contents.keys())

    def __len__(self):
        """Returns the length of the container object

        Return:
            (int): The length of the container
        """

        return len(self._contents.keys())
    #end
#end


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
        #end

        with self.assertRaises(TypeError):
            my_cont['zero'] = self.a
        #end


    def test_bad_get_object(self):
        """

        Tests getting an invalid index

        """
        my_cont = self.my_container

        my_cont[0] = self.a

        with self.assertRaises(TypeError):
            tmp = my_cont[0.0]
        #end

        with self.assertRaises(TypeError):
            tmp = my_cont['zero']
        #end

        with self.assertRaises(KeyError):
            tmp = my_cont[1]
        #end


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
        #end

    def test_bad_del_object(self):
        """

        Tests deleting an invalid object

        """

        my_cont = self.my_container

        my_cont[0] = self.a

        with self.assertRaises(TypeError):
            del my_cont[0.0]
        #end

        with self.assertRaises(TypeError):
            del my_cont['zero']
        #end

        with self.assertRaises(KeyError):
            del my_cont[1]
        #end


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

        Tests appending to a container where an index has been deleted. Should append after the
        last index

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
        #end
#end

if __name__ == '__main__':
    unittest.main(verbosity=4)
#end
