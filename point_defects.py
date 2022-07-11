from pymatgen.core.structure import Structure
from pymatgen.analysis.defects.utils import (
    StructureMotifInterstitial,
    TopographyAnalyzer,
)
from pymatgen.io.vasp import Kpoints, Incar
from pymatgen.io.vasp.sets import DictSet
from bsym.interface.pymatgen import unique_structure_substitutions
from tools import (
    get_charges,
    charge_identity,
    map_prim_defect_to_supercell,
    extend_list_to_zero,
    generate_interstitial_template,
)
from dataclasses import dataclass
from itertools import permutations
import numpy as np
from defect import Defect


@dataclass
class DefectSet:
    host_structure: Structure
    charge_tol: float = 5
    interstitial_scheme: str = "voronoi"

    def __post_init__(self):
        self.primitive_structure = self.host_structure.get_primitive_structure()
        self._get_cations_and_anions()
        self.vacancies = self._generate_all_vacancies()
        self.antisites = self._generate_all_antisites()
        if self.interstitial_scheme != None:
            self.interstitial_template = generate_interstitial_template(
                self.primitive_structure
            )
            self.interstitials = self._populate_interstitial_sites()
        else:
            self.interstitials = []

    def _get_antisite_charges(self, site, sub) -> list[int]:
        site_charges = get_charges(site, self.charge_tol) * -1
        sub_charges = get_charges(sub, self.charge_tol)

        if sub in self.cations and site in self.cations:
            charges = list(
                np.arange(
                    max(sub_charges) + min(site_charges),
                    max(sub_charges) + 1,
                    1,
                    dtype=int,
                )
            )
            if 0 not in charges:
                charges = extend_list_to_zero(charges)
            return sorted(charges)
        elif sub in self.anions and site in self.anions:
            charges = np.arange(
                min(sub_charges) + min(site_charges), min(sub_charges) + 1, 1, dtype=int
            )
            if 0 not in charges:
                charges = extend_list_to_zero(charges)
            return sorted(charges)

    def _generate_all_vacancies(self) -> list[Defect]:
        """
        For each atom present in the host structure, generate all symmetrically
        distinct vacancies, and return a list of `Defect` objects defining their
        relactive charges and site degeneracies (site degenceracy is reported
        as the site degeneracy of the defect in the **primitive** cell)
        """
        vacancies = []
        for k, v in self.primitive_structure.composition.items():
            # charges are multiplied by -1 to retreive relative charge of _vacancy_
            charges = get_charges(str(k), self.charge_tol) * -1
            v = unique_structure_substitutions(
                self.primitive_structure, str(k), {"X": 1, str(k): int(v - 1)}
            )
            for i, s in enumerate(v):
                defect_site = [i.frac_coords for i in s if i.species_string == "X0+"][0]
                degeneracy = s.number_of_equivalent_configurations
                s = map_prim_defect_to_supercell(
                    s, defect_site, host=k, host_structure=self.host_structure, primitive_structure=self.primitive_structure
                )
                vacancies.append(Defect(s, charges, degeneracy, f"v_{k}_{i+1}"))
        return vacancies

    def _get_cations_and_anions(self):
        atoms = self.primitive_structure.symbol_set
        cations = [
            a
            for a in atoms
            if charge_identity(a, self.charge_tol) == "cation"
            or charge_identity(a, self.charge_tol) == "both"
        ]
        anions = [
            a
            for a in atoms
            if charge_identity(a, self.charge_tol) == "anion"
            or charge_identity(a, self.charge_tol) == "both"
        ]
        self.cations = cations
        self.anions = anions

    def _get_antisites(self, antisite_elements: list[str]):
        composition = self.primitive_structure.composition
        antisites = []
        for native, substituent in permutations(antisite_elements, 2):
            live_antisites = unique_structure_substitutions(
                self.primitive_structure,
                native,
                {"X": int(1), native: int(composition[native] - 1)},
            )
            antisite_charges = self._get_antisite_charges(native, substituent)
            for i, antisite in enumerate(live_antisites):
                defect_site = [
                    i.frac_coords for i in antisite if i.species_string == "X0+"
                ][0]
                degeneracy = antisite.number_of_equivalent_configurations
                antisite.replace_species({"X0+": substituent})
                antisite = map_prim_defect_to_supercell(
                    antisite,
                    defect_site,
                    host=native,
                    host_structure=self.host_structure,
                    primitive_structure=self.primitive_structure
                )
                antisites.append(
                    Defect(
                        antisite,
                        antisite_charges,
                        degeneracy,
                        f"{native}_{substituent}_{i+1}",
                    )
                )
        return antisites

    def _generate_all_antisites(self):
        """
        Generate antisites
        """
        cation_antites = self._get_antisites(self.cations)
        anion_antites = self._get_antisites(self.anions)
        return cation_antites + anion_antites

    def _populate_interstitial_sites(self):
        """
        Populate interstitial sites
        """
        interstitials = []
        for atom in self.primitive_structure.symbol_set:
            charges = get_charges(str(atom), self.charge_tol)
            live_interstitals = unique_structure_substitutions(
                self.interstitial_template,
                "X",
                {"Fr": 1, "X": int(self.interstitial_template.composition["X"] - 1)},
            )
            for i, interstitial in enumerate(live_interstitals):
                defect_site = [
                    i.frac_coords for i in interstitial if i.species_string == "Fr"
                ][0]
                interstitial.replace_species({"Fr": atom})
                interstitial = map_prim_defect_to_supercell(
                    interstitial,
                    defect_site,
                    host=None,
                    host_structure=self.host_structure,
                    primitive_structure=self.primitive_structure
                )
                interstitials.append(
                    Defect(
                        interstitial,
                        charges,
                        interstitial.number_of_equivalent_configurations,
                        f"{atom}_i_{i+1}",
                    )
                )
        return interstitials

    def make_defect_calcs(self, dict_set, calc_type):
        for defect in self.vacancies + self.interstitials + self.antisites:
            all_charge_states = defect.charge_decorate_structures()
            if calc_type == "gam":
                for charge_state in all_charge_states:
                    calculation = DictSet(
                        charge_state,
                        dict_set,
                        use_structure_charge=True,
                        user_kpoints_settings=Kpoints(),
                    )
                    nupdown = {"NUPDOWN": calculation.incar["NELECT"] % 2}
                    calculation.write_input(
                        f"{defect.name}/{defect.name}_{charge_state.charge}"
                    )
                    incar = Incar.from_file(
                        f"{defect.name}/{defect.name}_{charge_state.charge}/INCAR"
                    )
                    incar |= nupdown
                    incar.write_file(
                        f"{defect.name}/{defect.name}_{charge_state.charge}/INCAR"
                    )
