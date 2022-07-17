import unittest
from unittest.mock import Mock
from tools import extend_list_to_zero, charge_identity, generate_interstitial_template, get_charges, get_prim_to_bulk_map, group_ions
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

        prim_map = get_prim_to_bulk_map(tst_prim, tst_conv)
        assert_equal(prim_map, [[1, 1, -1], [1, -1, 1], [-1, 1, 1]])

    def test_get_charges(self):

        Li_charges = get_charges("Li", 5)
        O_charges = get_charges("O", 5)
        Se_charges = get_charges("Se", 5)
        assert_equal(Li_charges,[0, 1])
        assert_equal(O_charges,[-2, -1, 0])
        assert_equal(Se_charges, [-2, -1, 0, 1, 2, 3, 4, 5, 6])

    def test_group_ions(self):

        species = ["Li", "O", "Se"]
        assert_equal(["Li", "Se"], group_ions(species, "cation", 5))
        assert_equal(["O", "Se"], group_ions(species, "anion", 5))

    def test_generate_interstitial_templates(self):

        # assert len(generate_interstitial_template(tst_prim, "infit")) == 2
        assert len(generate_interstitial_template(tst_prim, "voronoi")) == 4

    def test_map_prim_defect_to_supercell(self):
        

if __name__ == "__main__":
    unittest.main()
