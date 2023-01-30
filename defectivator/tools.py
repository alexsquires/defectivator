import numpy as np
import pandas as pd
from defectivator.interstitials import InterstitialMap
from copy import deepcopy
import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, 'data', path)

data = get_data('charges.csv')

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
    ox_states = pd.read_csv(data, delim_whitespace=True, index_col="Element")
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
    primitive_structure: "pymatgen.core.Structure",
    host_structure: "pymatgen.core.Structure",
) -> "np.Array[np.Array]":
    """
    Generate a transformation matrix between the of the primitive structure
    and the host structure.

    args:
        primitive_structure: primitive_structure for the defect structure to
            relate to
        host_structure: the defect supercell to be related back to the primitive
            structure
    """
    lattice = host_structure.get_reduced_structure().lattice
    prim_lattice = primitive_structure.get_reduced_structure().lattice
    mapping = prim_lattice.find_mapping(lattice)
    return mapping[-1]


def define_site_types(
    structure: "pymatgen.core.Structure", defect_position: list[float]
) -> None:
    """
    given a structure, and the position of a defect site in that structure,
    label all the sites in the structure as either "native" or "defect"

    args:
        structure: structure to label sites in
        defect_position: position of the defect site in the structure
    """
    site_type = []
    for site in structure:
        if list(site.frac_coords) == list(defect_position):
            site_type.append("defect")
        else:
            site_type.append("native")
    structure.add_site_property("site_type", site_type)


def map_prim_defect_to_supercell(
    structure: "pymatgen.core.Structure",
    defect_position: list[float],
    host: str,
    host_cell: "pymatgen.core.structure",
) -> "pymatgen.core.structure":
    """given a primitive cell continaing a defect, `structure`
    find the transformation matrix between that and a supecell,
    and generate the supercell containing a single point defect

    Args:
        structure (pymatgen.core.Structure): primitive cell with
            a defect
        defect position (list[float]): fractional coordinates of
            the point defect
        host: (Optional[str]): the host species for a defect,
            None for interstitials
        host_cell (pymatgen.core.Structure): supercell for defect
            calculation

    Returns:
        pmg.core.Structure: supercell containing a point defect.
    """

    # get the mapping between the primitive structure and the supercell, labelling
    # the sites in the supercell as either "native" or "defect"
    define_site_types(structure, defect_position)
    mapping = get_prim_to_bulk_map(
        primitive_structure=structure, host_structure=host_cell
    )

    # make a supercell of the structure between the bulk and primitive cell
    # and collect the site indicies of the defects in the supercell 
    structure.make_supercell(mapping)
    defects = [
        i for i, j in enumerate(structure) if j.properties["site_type"] == "defect"
    ]

    # get the indicies of the defect images in the supercell, if there is a host
    # species that should be on that site, then replace the defect site with the
    # host site
    if host != None:
        for i in defects[1:]:
            structure[i].species = host
            structure[i].properties["site_type"] = "native"
    
    # if their is no host species, i.e. an interstitial defect, then simply
    # remove the defect images from it
    else:
        structure.remove_sites(defects[1:])
    return structure


def generate_interstitial_template(
    structure: "pymatgen.core.Structure"
):
    """Return a structure populated with interstitial sites

    Args:
        structure (pymatgen.core.Structure): structure to find interstitials in

    Returns:
        "pymatgen.core.Structure": structure populated with interstitials
    """
    structure = deepcopy(structure)
    map = InterstitialMap(structure)
    return map.structure


def group_ions(species, atom_type, charge_tol=5):
    """given a list of atoms, find all the cations
    or anions depending on the atom_type arg

    args:
        species (list[str]): list of atoms to pick out all the anions or cations
        atoms_type (str): whether to search for ions classed as "anion" or "cation"
        charge_tol (float): charge_tolerance to pass to charge_indentity

    returns:
        ions (list[str]): list of atoms which have been defined as either anions
            or cations.
    """
    ions = [
        a
        for a in species
        if charge_identity(a, charge_tol) == atom_type
        or charge_identity(a, charge_tol) == "both"
    ]
    return ions
