from dataclasses import dataclass
from pymatgen.core import Structure
from copy import deepcopy
import numpy as np


@dataclass
class Defect:
    """Class which describes a Defect in a structure"""

    structure: Structure
    charges: list
    degeneracy: int
    name: str
    center: bool = True

    def __post_init__(self):
        if self.center:
            defect_site = [
                i.frac_coords
                for i in self.structure
                if i.properties["site_type"] == "defect"
            ][0]
            self.structure.translate_sites(
                [i for i in range(len(self.structure))], -1 * np.array(defect_site)
            )
            self.structure.translate_sites(
                [i for i in range(len(self.structure))], [0.5, 0.5, 0.5]
            )
        if "X" in self.structure.symbol_set:
            self.structure.remove_species(["X0+"])

    def charge_decorate_structures(self) -> list[Structure]:
        """
        decorate the structure of the defect with all defined charge states
        such that a calculation can be generated using pymatgen with no manual
        electron counting required.

        returns:
            structures (list): list charged structures
        """
        structures = []
        for charge in self.charges:
            structure = deepcopy(self.structure)
            structure.set_charge(charge)
            structures.append(structure)
        return structures
