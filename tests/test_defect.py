import unittest
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor

from defectivator.defect import Defect

tst_structure = AseAtomsAdaptor.get_structure(bulk("LiF", "rocksalt", 4.1, cubic=True))


class TestDefect(unittest.TestCase):
    def setUp(self) -> None:
        structure = tst_structure
        charges = [1, 2, 3]
        degeneracy = 3
        name = "foo"
        center = False

        self.defect = Defect(
            structure=structure,
            charges=charges,
            degeneracy=degeneracy,
            name=name,
            center=center,
        )

    def test_Defect(self):
        assert self.defect.name == "foo"
        assert self.defect.charges == [1, 2, 3]
        assert self.defect.degeneracy == 3
        assert self.defect.center == False
        assert len(self.defect.structure) == 8

    def test_charge_decorate_structures(self):
        structures = self.defect.charge_decorate_structures()
        assert len(structures) == 3
        for structure in structures:
            assert (
                structure.charge == 1 or structure.charge == 2 or structure.charge == 3
            )


if __name__ == "__main__":
    unittest.main()
