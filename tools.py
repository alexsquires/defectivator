import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.defects.utils import (
    StructureMotifInterstitial,
    TopographyAnalyzer,
)
from copy import deepcopy


def extend_list_to_zero(l: list) -> np.array:
    """
    find the nearest integer to 0 in the list, and include all numbers between
    the nearest integer and 0 in the list.

    args:
        l (list): list of numbers
    returns:
        np.array: array of numbers, extended to 0
    """
    furthest_int = max([i for i in l if i != 0], key=lambda x: abs(x))
    if furthest_int < 0:
        return np.arange(furthest_int, 0 + 1, 1)
    else:
        return np.arange(0, furthest_int + 1, 1)


def get_charges(atom: str, charge_tol: float = 5) -> np.array:
    """
    for a given atom, parse the oxidation states of that atom present in the
    ICSD and return a range of charges from the highest to lowest possible
    oxidation states. The charges are only considered if they represent greater
    than `charge_tol`% of all the oxidation states of the atoms in the ICSD

    ICSD data taken from: `doi.org/10.1021/acs.jpclett.0c02072`

    args:
        atom (str): atom to get oxidation states for
        charge_tol (float): tolerance for oxidation states in percentage
    returns:
        np.array: array of reasonable oxidation states for atom

    """
    ox_states = pd.read_csv("charges.csv", delim_whitespace=True, index_col="Element")
    perc = ox_states.loc[atom] / ox_states.loc[atom].sum() * 100
    charges = [int(k) for k, v in perc.items() if v > charge_tol]
    if 0 not in charges:
        charges.append(0)
    return np.arange(min(charges), max(charges) + 1, 1, dtype=int)


def charge_identity(atom: str, charge_tol: float) -> str:
    """
    define whether we consider an atom to be an anion, cation or amphoteric

    args:
        atom (str): atom to check
        charge_tol (float): tolerance for oxidation states in percentage
    returns:
        str: "anion", "cation" or "amphoteric"
    """
    charges = get_charges(atom, charge_tol)
    if all(charges >= 0):
        return "cation"
    elif all(charges <= 0):
        return "anion"
    else:
        return "both"


def get_prim_to_bulk_map(
    host_structure: Structure, primitive_structure: Structure
) -> "np.Array[np.Array]":
    """
    Generate a transformation matrix between the of the primitive structure
    and the bulk structure
    """
    lattice = host_structure.get_reduced_structure().lattice
    prim_lattice = primitive_structure.get_reduced_structure().lattice
    mapping = prim_lattice.find_mapping(lattice)
    return mapping[-1]


def map_prim_defect_to_supercell(
    structure: Structure,
    defect_site: list[float],
    host: str,
    host_structure: Structure,
    primitive_structure: Structure,
) -> Structure:
    """ """
    site_type = []
    for site in structure:
        if list(site.frac_coords) == list(defect_site):
            site_type.append("defect")
        else:
            site_type.append("native")
    structure.add_site_property("site_type", site_type)
    mapping = get_prim_to_bulk_map(primitive_structure = primitive_structure, host_structure = host_structure)
    structure.make_supercell(mapping)
    defects = [
        i for i, j in enumerate(structure) if j.properties["site_type"] == "defect"
    ]
    if host != None:
        for i in defects[1:]:
            structure[i].species = host
            structure[i].properties["site_type"] = "native"
    else:
        structure.remove_sites(defects[1:])
    return structure


def generate_interstitial_template(structure, interstitial_scheme="voronoi"):
    """
    Generate interstitials
    """
    structure = deepcopy(structure)
    if interstitial_scheme == "infit":
        interstitial_generator = StructureMotifInterstitial(structure, "Fr")
        interstitials = interstitial_generator.enumerate_defectsites()
        for interstitial in interstitials:
            structure.append(interstitial.species, interstitial.frac_coords)
    structure.replace_species({"Fr": "X"})
    if interstitial_scheme == "voronoi":
        interstitial_generator = TopographyAnalyzer(structure, structure.symbol_set, [])
        interstitial_generator.cluster_nodes(0.8)
        interstitial_generator.remove_collisions()
        structure = interstitial_generator.get_structure_with_nodes()
    return structure
