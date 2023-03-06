from dataclasses import dataclass
from defectivator.tools import distort
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor as AAA
from copy import deepcopy
import numpy as np


@dataclass
class Defect:
    """Class which describes a defect in a periodic structure
    
    args:
        structure (pmg.Structure): structure object that describes the defect
        charges (list[int]): list of integers describing the charge states the 
            defect can adopt
        degeneracy (int): integer describing the site degeneracy of the defined 
            defect
        name (str): a unique label that can be used to distinguish the defect
        center (bool): whether to shift the origin of the cell such that the 
            defect is at fraction coordinates [0.5, 0.5, 0.5]
    """

    structure: "pymatgen.core.Structure"
    defect_coordinates: list[float]
    charges: list[int]
    abs_delta_e: list[int]
    degeneracy: int
    name: str
    center_defect: bool = True

    def __post_init__(self):
        if self.center_defect:
            # shift the defect to fraction coordinates [0.5, 0.5, 0.5]
            self._center()
            self.centered = True
        else:
            self.centered = False
        if "X" in self.structure.symbol_set:
            # remove any dummy atoms from the structure
            self.structure.remove_species(["X0+"])

        if "v" not in self.name:
            self.defect_index = [
                i
                for i, j in enumerate(self.structure)
                if j.properties["site_type"] == "defect"
            ][0]
        else:
            self.defect_index = None

    def _center(self) -> None:
        self.structure.translate_sites(
            [i for i in range(len(self.structure))], -1 * np.array(self.defect_coords)
        )
        self.structure.translate_sites(
            [i for i in range(len(self.structure))], [0.5, 0.5, 0.5]
        )

    def get_distortions(
        self, min_distortion: float = 0.6, max_distortion: float = 1.4
    ) -> list[Structure]:
        """snb style distortions

        Returns:
            list[Structure]: _description_
        """
        all_structures =[]
        for charge, delta_e in zip(self.charges, self.abs_delta_e):

            print(charge)
            if abs(delta_e) > 4:
                num_neighbours = abs(8 - delta_e)
            else:
                num_neighbours = abs(delta_e)

            structures = []

            if num_neighbours == 0:
                structure = self.structure.copy()
                structure.set_charge(charge)
                structures.append(structure)
            else:
                for distortion_factor in np.arange(
                    min_distortion, max_distortion + 0.1, 0.1
                ):
                    
                    structure = distort(
                        structure=self.structure.copy(),
                        num_nearest_neighbours=num_neighbours,
                        distortion_factor=distortion_factor,
                        defect_index = self.defect_index,
                        defect_frac_coords=self.defect_coordinates
                    )
                    structure.set_charge(charge)
                    structures.append(structure.get_sorted_structure())
            all_structures.extend(structures)
        return all_structures

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
