from pymatgen.core.structure import Structure
from bsym.interface.pymatgen import unique_structure_substitutions
from dataclasses import dataclass
from defect import Defect
from point_defects import DefectSet
from tools import map_prim_defect_to_supercell, get_charges


@dataclass
class DefectComplexMaker:
    defect_set: DefectSet
    reference_defect_label: str
    associated_defect_label: str
    n_associated_defects: int

    def __post_init__(self):
        point_defects = (
            self.defect_set.vacancies
            + self.defect_set.antisites
            + self.defect_set.interstitials
        )
        self.reference_defect = [
            d for d in point_defects if d.name == self.reference_defect_label
        ][0]

        if self.associated_defect_label.split("_")[0] == "v":
            self._associated_species = self.associated_defect_label.split("_")[1]
            self.defect_complexes = self._generate_vacancy_complexes()

        elif self.associated_defect_label.split("_")[1] == "i":
            None

        else:
            self._associated_species_native = self.associated_defect_label.split("_")[1]
            self._associated_species_substituent = self.associated_defect_label.split(
                "_"
            )[0]
            self.defect_complexes = self._generate_antisite_complexes()

    def _generate_vacancy_complexes(self):

        all_complexes = []
        structure = self.reference_defect.structure
        n_native = structure.composition[self._associated_species]
        substitutions = unique_structure_substitutions(
            structure,
            self._associated_species,
            {
                "X": int(self.n_associated_defects),
                self._associated_species: int(n_native - self.n_associated_defects),
            }
        )
            
        for sub in substitutions:
            sub.remove_species(["X"])
            all_complexes.append(sub)
        return all_complexes

    def _generate_antisite_complexes(self):

        all_complexes = []
        structure = self.reference_defect.structure
        n_native = structure.composition[self._associated_species_native]
        substitutions = unique_structure_substitutions(
            structure,
            self._associated_species_native,
            {
                self._associated_species_substituent: int(self.n_associated_defects),
                self._associated_species_native: int(n_native - self.n_associated_defects),
            }
        )
        for sub in substitutions:
            all_complexes.append(sub)
        return all_complexes