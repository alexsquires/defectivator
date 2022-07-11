import unittest
from unittest.mock import Mock
from tools import extend_list_to_zero, charge_identity, get_prim_to_bulk_map
from numpy.testing import assert_equal

from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor

tst_prim = AseAtomsAdaptor.get_structure(bulk("LiF", "rocksalt", 4.1, cubic=False))
tst_conv = AseAtomsAdaptor.get_structure(bulk("LiF", "rocksalt", 4.1, cubic=True))

tst_conv = tst_conv.get_reduced_structure()
tst_prim = tst_prim.get_reduced_structure()

class testTools(unittest.TestCase):

    def test_extend_list_to_zero(self):
        """
        Test extend list to zero function
        """
        test_list = [1, 2, 3]
        test_list_extended = extend_list_to_zero(test_list)
        assert_equal(test_list_extended, [0, 1, 2, 3])

        test_list_2 = [-1, -2, -3]
        test_list_extended_2 = extend_list_to_zero(test_list_2)
        assert_equal(test_list_extended_2, [-3, -2, -1, 0])

    def test_charge_identity(self):
        """
        Test charge identity function
        """
        assert charge_identity("Li", 5) == "cation"
        assert charge_identity("O", 5) == "anion"
        assert charge_identity("Se", 5) == "both"

    def test_get_prim_to_bulk_map(self):
        """
        Test get_prim_to_bulk_map function
        """

        prim_map = get_prim_to_bulk_map(tst_conv, tst_prim)
        assert_equal(prim_map, [[1, 1, -1],[1, -1, 1],[-1, 1, 1]])

if __name__ == "__main__":
    unittest.main()