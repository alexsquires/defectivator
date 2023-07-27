from pymatgen.io.vasp import Kpoints, Incar
from pymatgen.io.vasp.sets import DictSet
from pymatgen.core import Element
from bsym.interface.pymatgen import unique_structure_substitutions
from defectivator.tools import (
    get_charges,
    map_prim_defect_to_supercell,
    extend_list_to_zero,
    generate_interstitial_template,
    group_ions,
    classify_defect,
    find_interstitial,
    get_highest_charge
)
from dataclasses import dataclass
from itertools import permutations, product
import numpy as np
from defectivator.defect import Defect
from typing import Optional, Union
from copy import deepcopy
from pymatgen.analysis.structure_matcher import StructureMatcher


@dataclass
class PointDefectSet:
    """class for generating point defects in a structure. The generated `Defect`
    objects will assume a fully static lattice. Further approaches should be
    applied for ground states defect structure searching.

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
    center_defects: bool = False
    bulk_oxidation_states: Optional[dict] = None

    def __post_init__(self):
        self.host_structure = self.host_structure.get_reduced_structure()
        self.primitive_structure = self.host_structure.get_primitive_structure()
        if self.bulk_oxidation_states is None:
            self.bulk_oxidation_states = (
                self.host_structure.composition.oxi_state_guesses()[0]
            )

        self._get_cations_and_anions()
        self.vacancies = self._generate_all_vacancies()
        self.su = self._generate_all_substitutions()
        if self.extrinsic_species is not None:
            self.substitutions = self._generate_all_dopant_substiutions()
        if self.interstitial_scheme is not None:
            # find interstitials only if an interstitial scheme is specified
            self.interstitial_template = generate_interstitial_template(
                self.primitive_structure
            )
            self.interstitial_template.merge_sites(1, "delete")
            self.interstitials = self._populate_interstitial_sites()
        else:
            self.interstitials = []

    def _get_substitution_charges(self, site, sub) -> list[int]:
        site_charges = get_charges(site, self.charge_tol) * -1
        sub_charges = get_charges(sub, self.charge_tol)

        if sub in self.cations +self.dopant_cations and site in self.cations:
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
        elif sub in self.anions + self.dopant_anions and site in self.anions:
            charges = np.arange(
                min(sub_charges) + min(site_charges), min(sub_charges) + 1, 1, dtype=int
            )
            if 0 not in charges:
                charges = extend_list_to_zero(charges)
            return sorted(charges)

    def _get_n_elect(self, defect_name: str):
        oxidation_states = deepcopy(self.bulk_oxidation_states)
        oxidation_states.update({"v": 0})

        for species in self.extrinsic_species:
            oxidation_states.update({species: get_highest_charge(species)})

        defect = classify_defect(defect_name)

        if defect[0] == "vacancy":
            site_species = defect[1]
            subs_species = "v"

        elif defect[0] == "interstitial":
            subs_species = defect[1]
            site_species = "v"

        elif defect[0] == "substitution":
            subs_species = defect[1]
            site_species = defect[2]

        num_electrons = oxidation_states[subs_species] - oxidation_states[site_species]

        return int(-num_electrons)

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
                    host_cell=self.host_structure,
                )
                name = f"v_{k}_{i+1}"
                vacancies.append(
                    Defect(
                        structure=s,
                        defect_coordinates=defect_site,
                        charges=charges,
                        abs_delta_e=[
                            abs(self._get_n_elect(name) + charge) for charge in charges
                        ],
                        degeneracy=degeneracy,
                        name=name,
                        center_defect=self.center_defects,
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
        if self.extrinsic_species is not None:
            dopants = self.extrinsic_species
            dopant_cations = group_ions(dopants, "cation", 5)
            dopant_anions = group_ions(dopants, "anion", 5)
            self.dopant_cations = dopant_cations
            self.dopant_anions = dopant_anions
        else:
            self.dopant_cations = []
            self.dopant_anions = []

    def _get_substitutions(self, substitution_elements: list[str]):
        """
        get all possible cation and anion substitutions in the material.
        If a species can exhibit both catio and anion-like behviour, substitutions
        will be generated for both ion types.
        """
        composition = self.primitive_structure.composition
        substitutions = []
        for native, substituent in permutations(substitution_elements, 2):
            live_substitutions = unique_structure_substitutions(
                self.primitive_structure,
                native,
                {"X": int(1), native: int(composition[native] - 1)},
            )
            substitution_charges = self._get_substitution_charges(native, substituent)
            for i, substitution in enumerate(live_substitutions):
                defect_site = [
                    i.frac_coords for i in substitution if i.species_string == "X0+"
                ][0]
                degeneracy = substitution.number_of_equivalent_configurations
                substitution.replace_species({"X0+": substituent})
                substitution = map_prim_defect_to_supercell(
                    substitution,
                    defect_site,
                    host=native,
                    host_cell=self.host_structure,
                )
                name = f"{native}_{substituent}_{i+1}"
                substitutions.append(
                    Defect(
                        structure=substitution,
                        defect_coordinates=defect_site,
                        charges=substitution_charges,
                        abs_delta_e=[
                            abs(self._get_n_elect(name) + charge)
                            for charge in substitution_charges
                        ],
                        degeneracy=degeneracy,
                        name=f"{native}_{substituent}_{i+1}",
                        center_defect=self.center_defects,
                    )
                )
        return substitutions

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
            substitution_charges = self._get_substitution_charges(native, substituent)
            for i, sub in enumerate(live_substitutions):
                defect_site = [i.frac_coords for i in sub if i.species_string == "X0+"][
                    0
                ]
                degeneracy = sub.number_of_equivalent_configurations
                sub.replace_species({"X0+": substituent})
                substitution = map_prim_defect_to_supercell(
                    sub,
                    defect_site,
                    host=native,
                    host_cell=self.host_structure,
                )

                name = f"{substituent}_{native}_{i+1}"
                all_substitutions.append(
                    Defect(
                        structure=substitution,
                        defect_coordinates=defect_site,
                        charges=substitution_charges,
                        abs_delta_e = [
                            abs(self._get_n_elect(name) + charge)
                            for charge in substitution_charges
                        ],
                        degeneracy=degeneracy,
                        name=name,
                        center_defect=self.center_defects,
                    )
                )
        return all_substitutions

    def _generate_all_substitutions(self) -> list:
        """
        Generate substitutions

        returns:
            list[Defect]: a list of defects representing all
            the intrinsic substitutions
        """
        cation_antites = self._get_substitutions(self.cations)
        anion_antites = self._get_substitutions(self.anions)
        return cation_antites + anion_antites

    def _populate_interstitial_sites(self) -> list[Defect]:
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
                    host_cell=self.host_structure,
                )
                name = f"{atom}_i_{i+1}"
                interstitials.append(
                    Defect(
                        structure=interstitial,
                        defect_coordinates=defect_site,
                        charges=charges,
                        degeneracy=interstitial.number_of_equivalent_configurations,
                        abs_delta_e=[
                            abs(self._get_n_elect(name) + charge) for charge in charges
                        ],
                        name=name,
                        center_defect=self.center_defects,
                    )
                )
        return interstitials

    def cluster_interstitials_by_relaxation(self):
        """ """

        interstitals_dict = {}
        species = [k for k in self.bulk_oxidation_states.keys()]
        for specie in species:
            interstitals = [
                interstitial
                for interstitial in self.interstitials
                if interstitial.name.split("_")[0] == specie
            ]
            interstitals_dict.update({specie: interstitals})

        all_interstitials = []
        for k, v in interstitals_dict.items():
            interstital_structures = []
            for defect in v:
                structure = defect.structure.copy().relax(
                    relax_cell=False, verbose=False
                )
                interstital_structures.append(structure)

            sm = StructureMatcher()
            groups = sm.group_structures(interstital_structures)
            interstital_structures = [group[0] for group in groups]

            for i, structure in enumerate(interstital_structures):
                interstitials = Defect(
                    structure=structure,
                    charges=defect.charges,
                    degeneracy=defect.degeneracy,
                    name=f"{k}_i_{i}",
                    abs_delta_e=defect.abs_delta_e,
                    defect_coordinates=find_interstitial(
                        structure, self.host_structure, k
                    ),
                    center_defect=False,
                )
                all_interstitials.append(interstitials)

        self.interstitials = all_interstitials

    def make_defect_calcs(
        self, dict_set: "pymatgen.io.vasp.set.DictSet", calc_type: str
    ) -> None:
        """
        taking all defects in the point DefectSet, write vasp calculations
        given a pymatgen.io.vasp.sets.DictSet object, and wether or not the calculation
        should be gamma only (useful for a down-sampled first-pass on defect energies)

        ##TODO: currently only accepts `gam`: `std` and `ncl` should also be options

        args:
            dict_set (pymatgen.io.vasp.sets.DictSet):
            calc_type(str):
        """
        for defect in (
            self.vacancies + self.interstitials + self.substitutions + self.substitutions
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
