import numpy as np
import pandas as pd
from defectivator.interstitials import InterstitialMap
from copy import deepcopy
from pymatgen.util.coord import pbc_diff
from pymatgen.io.ase import AseAtomsAdaptor as AAA
import os
from typing import Optional
from pymatgen.core import Structure

# from hiphive.structure_generation.rattle import (
#     generate_mc_rattled_structures,
# )
from pymatgen.io.ase import AseAtomsAdaptor as AAA

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data(path):
    return os.path.join(_ROOT, "data", path)


data = get_data("charges.csv")


def classify_defect(defect_name: str) -> "str":
    """
    Args:
        defect_name (str): _description_

    Returns:
        str: _description_
    """
    species, site = defect_name.split("_")[:2]

    if species == "v":
        return "vacancy", site

    elif site == "i":
        return "interstitial", species

    else:
        assert len(species) in [1, 2] and len(site) in [1, 2]
        return "anitsite", species, site


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


def distort(
    structure: "pymatgen.core.Structure",
    num_nearest_neighbours: int,
    distortion_factor: float,
    defect_index: Optional[int] = None,
    defect_frac_coords: Optional[list[float]] = None,
) -> Structure:
    """
    rewritten from the excellent ShakeNBreak!
    """

    if defect_index == None:
        structure.append("Fr", defect_frac_coords)
        defect_index = [i for i, j in enumerate(structure) if j.species_string == "Fr"][
            0
        ]

    distances = [
        [structure.get_distance(defect_index, j), defect_index, j]
        for j in range(len(structure))
        if j != defect_index
    ]
    distances = sorted(  # Sort the distances shortest->longest
        distances, key=lambda tup: tup[0]
    )
    near_neighbors = distances[0 : num_nearest_neighbours + 1]
    distorted = [(i[0] * distortion_factor, i[1], i[2]) for i in near_neighbors]

    input_atoms = AAA.get_atoms(structure)
    for i in distorted:
        input_atoms.set_distance(i[1], i[2], i[0], fix=0, mic=True)

    distorted_structure = AAA.get_structure(input_atoms)
    distorted_structure.remove_species(["Fr"])
    return distorted_structure


def rattle(
    structure: Structure,
    stdev: float = 0.15,
    d_min: float = 1.5,
    n_iter: int = 1,
    active_atoms: Optional[list] = None,
    width: float = 0.1,
    max_attempts: int = 10000,
    max_disp: float = 2.0,
    seed: int = 42,
) -> Structure:
    """
    Given a pymatgen Structure object, apply random displacements to all atomic
    positions, with the displacement distances randomly drawn from a Gaussian
    distribution of standard deviation `stdev`.
    Args:
        structure (:obj:`~pymatgen.core.structure.Structure`):
            Structure as a pymatgen object
        stdev (:obj:`float`):
            Standard deviation (in Angstroms) of the Gaussian distribution from
            which atomic displacement distances are drawn.
            (Default: 0.25)
        d_min (:obj:`float`):
            Minimum interatomic distance (in Angstroms). Monte Carlo rattle
            moves that put atoms at distances less than this will be heavily
            penalised.
            (Default: 2.25)
        n_iter (:obj:`int`):
            Number of Monte Carlo cycles to perform.
            (Default: 1)
        active_atoms (:obj:`list`, optional):
            List of which atomic indices should undergo Monte Carlo rattling.
            (Default: None)
        nbr_cutoff (:obj:`float`):
            The radial cutoff distance (in Angstroms) used to construct the
            list of atomic neighbours for checking interatomic distances.
            (Default: 5)
        width (:obj:`float`):
            Width of the Monte Carlo rattling error function, in Angstroms.
            (Default: 0.1)
        max_disp (:obj:`float`):
            Maximum atomic displacement (in Angstroms) during Monte Carlo
            rattling. Rarely occurs and is used primarily as a safety net.
            (Default: 2.0)
        max_attempts (:obj:`int`):
            Maximum Monte Carlo rattle move attempts allowed for a single atom;
            if this limit is reached an `Exception` is raised.
            (Default: 5000)
        seed (:obj:`int`):
            Seed for NumPy random state from which random rattle displacements
            are generated. (Default: 42)
    Returns:
        :obj:`Structure`:
            Rattled pymatgen Structure object
    """
    ase_struct = AAA.get_atoms(structure)

    rattled_ase_struct = generate_mc_rattled_structures(
        ase_struct,
        n_configs=1,
        rattle_std=stdev,
        d_min=d_min,
        n_iter=n_iter,
        active_atoms=active_atoms,
        width=width,
        max_attempts=max_attempts,
        max_disp=max_disp,
        seed=seed,
    )[0]
    rattled_structure = AAA.get_structure(rattled_ase_struct)
    rattled_structure.set_charge(structure.charge)
    return rattled_structure.get_sorted_structure()


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


def get_prim_map(
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
    prim_lattice = (
        primitive_structure.get_reduced_structure().get_primitive_structure().lattice
    )
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


def generate_interstitial_template(structure: "pymatgen.core.Structure"):
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


def find_interstitial(
    defect: "pymatgen.core.Structure", host: "pymatgen.core.Structure", species: str
) -> list[float]:
    """
    Args:
        defect (pymatgen.core.Structure): _description_
        host (pymatgen.core.Structure): _description_
        species (str): _description_

    Returns:
        list[float]: _description_
    """
    temp_structure = host.copy()
    for site in defect:
        if site.species_string == species:
            temp_structure.append(
                site.species_string, site.frac_coords, coords_are_cartesian=False
            )

    distance_matrix = np.linalg.norm(
        pbc_diff(
            np.array([s.coords for s in host])[:, None],
            np.array([s.coords for s in defect]),
        ),
        axis=2,
    )
    site_matches = distance_matrix.argmin(axis=1)
    defect_index = list(
        set(np.arange(defect.composition[species])) - set(site_matches)
    )[0]
    return defect[int(defect_index)].frac_coords


def get_defect_coords(
    bulk: "pymatgen.core.Structure", defect: "pymatgen.core.Structure", defect_name: str
) -> list[float]:
    """_summary_

    Args:
        bulk (pymatgen.core.Structure): _description_
        defect (pymatgen.core.Structure): _description_
        defect_name (str): _description_

    Returns:
        List[float]: _description_
    """

    bulk_prim = bulk.copy().get_primitive_structure()
    matrix = get_prim_to_bulk_map(bulk_prim, defect)
    bulk_prim.make_supercell(matrix)

    defect_type = classify_defect(defect_name)
    if defect_type[0] == "interstitial":
        return find_interstitial(defect, bulk_prim, defect_type[1])

    # TODO vacancies and substitutions
