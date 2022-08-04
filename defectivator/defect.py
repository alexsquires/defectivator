from dataclasses import dataclass
from pymatgen.core import Structure
from copy import deepcopy
import numpy as np


@dataclass
class Defect:
    """Class which describes a defect in a periodic structure
    
    args:
        structure (pmg.Structure): structure object that describes the defect
        charges (list[int]): list of integers descibing the charge states the 
            defect can adopt
        degeneracy (int): integer describing the site degeneracy of the defined 
            defect
        name (str): a unique label that can be used to distinguish the defect
        center (bool): whether to shift the origin of the cell such that the 
            defect is at fraction coordinates [0.5, 0.5, 0.5]
    """

    structure: "pymatgen.core.Structure"
    charges: list[int]
    degeneracy: int
    name: str
    center: bool = True

    def __post_init__(self):
        if self.center:
            # shift the defect to fraction coordinates [0.5, 0.5, 0.5]
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
            # remove any dummy atoms from the structure
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
