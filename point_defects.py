from pymatgen.io.vasp import Kpoints, Incar
from pymatgen.io.vasp.sets import DictSet
from bsym.interface.pymatgen import unique_structure_substitutions
from tools import (
    get_charges,
    map_prim_defect_to_supercell,
    extend_list_to_zero,
    generate_interstitial_template,
    group_ions,
)
from dataclasses import dataclass
from itertools import permutations, product
import numpy as np
from defect import Defect


@dataclass
class DefectSet:
    """class for generating point defects in a structure

    args:
        host_structure (pymatgen.core.Structure): the structure to generate
            point defects within
        extrinsic_species (list[str] or None): any species to include as dopants
        charge_tol (float): charge_tolerance cutoff to feed to get_charges
            (see defectivator.tools.get_charges)
        interstitial_scheme: scheme for searching/generating interstitials
            (see defectivator.tools.generate_interstitial_template)
        center_defects (bool): whether or not the Defect objects generated are
            shifted such that the defect sits at the center of the structure.
    """

    host_structure: "pymatgen.core.Structure"
    extrinsic_species: list = None
    charge_tol: float = 5
    interstitial_scheme: str = "voronoi"
    center_defects: bool = True

    def __post_init__(self):
        self.primitive_structure = self.host_structure.get_primitive_structure()
        self._get_cations_and_anions()
        self.vacancies = self._generate_all_vacancies()
        self.antisites = self._generate_all_antisites()
        if self.extrinsic_species != None:
            self.substitutions = self._generate_all_dopant_substiutions()
        if self.interstitial_scheme != None:
            # find interstitials only if an interstitial scheme is specified
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

        returns:
             vacancies (list[Defect]): list of instrinsic vacancy defects
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
                    s,
                    defect_site,
                    host=k,
                    primitive_structure=self.primitive_structure,
                )
                vacancies.append(
                    Defect(
                        s,
                        charges,
                        degeneracy,
                        f"v_{k}_{i+1}",
                        center=self.center_defects,
                    )
                )
        return vacancies

    def _get_cations_and_anions(self):
        """taking all the intrisic atoms and extrinsic dopants (if any)
        define them all as anions or cations, and add these as atributies
        of the defect set.
        """
        atoms = self.primitive_structure.symbol_set
        cations = group_ions(atoms, "cation", 5)
        anions = group_ions(atoms, "anion", 5)
        self.anions = anions
        self.cations = cations

        # if there are any extrinsic species, also determine whether
        # these are anions or cations
        if self.extrinsic_species != None:
            dopants = self.extrinsic_species
            dopant_cations = group_ions(dopants, "cation", 5)
            dopant_anions = group_ions(dopants, "anion", 5)
            self.dopant_cations = dopant_cations
            self.dopant_anions = dopant_anions
        else:
            self.dopant_cations = []
            self.dopant_anions = []

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
                    primitive_structure=self.primitive_structure,
                )
                antisites.append(
                    Defect(
                        antisite,
                        antisite_charges,
                        degeneracy,
                        f"{native}_{substituent}_{i+1}",
                        center=self.center_defects,
                    )
                )
        return antisites

    def _generate_all_dopant_substiutions(self):

        composition = self.primitive_structure.composition
        all_substitutions = []
        if self.dopant_cations != []:
            substitutions = list(product(self.dopant_cations, self.cations))

        for substituent, native in substitutions:
            live_substitutions = unique_structure_substitutions(
                self.primitive_structure,
                native,
                {"X": int(1), native: int(composition[native] - 1)},
            )
            substitution_charges = self._get_antisite_charges(native, substituent)
            for i, sub in enumerate(live_substitutions):
                defect_site = [
                    i.frac_coords for i in sub if i.species_string == "X0+"
                ][0]
                degeneracy = sub.number_of_equivalent_configurations
                sub.replace_species({"X0+": substituent})
                substitution = map_prim_defect_to_supercell(
                    sub,
                    defect_site,
                    host=native,
                    primitive_structure=self.primitive_structure,
                )
                all_substitutions.append(
                    Defect(
                        substitution,
                        substitution_charges,
                        degeneracy,
                        f"{native}_{substituent}_{i+1}",
                        center=self.center_defects,
                    )
                )
        return all_substitutions

    def _generate_all_antisites(self):
        """
        Generate antisites

        returns:
            list[Defect]: a list of defects representing all
            the intrinsic antisites
        """
        cation_antites = self._get_antisites(self.cations)
        anion_antites = self._get_antisites(self.anions)
        return cation_antites + anion_antites

    def _populate_interstitial_sites(self):
        """
        Populate interstitial sites

        returns:
            list[Defect]: a list of defects representing all
            the interstitals in the host structure.
        """
        interstitials = []
        atoms = list(
            self.anions + self.cations + self.dopant_anions + self.dopant_cations
        )

        for atom in atoms:
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
                    primitive_structure=self.primitive_structure,
                )
                interstitials.append(
                    Defect(
                        interstitial,
                        charges,
                        interstitial.number_of_equivalent_configurations,
                        f"{atom}_i_{i+1}",
                        center=self.center_defects,
                    )
                )
        return interstitials

    def make_defect_calcs(self, dict_set, calc_type):
        for defect in (
            self.vacancies + self.interstitials + self.antisites + self.substitutions
        ):
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
