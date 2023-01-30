from pymatgen.core import Structure
from pymatgen.core.sites import Site
from pymatgen.transformations.advanced_transformations import (
    MonteCarloRattleTransformation as MCRT,
)
import numpy as np
from random import uniform
from copy import deepcopy
from typing import Union
from defectivator.defect import Defect

from dataclasses import dataclass

@dataclass
class generate_random_defects:
    """_summary_
    """
    defect: Defect
    n_structures: int
    reconstruction_radius: float # would be good to also accept a range
    rattle_std: float = 0 # want to code this up so that if rattle_std == 0, then no rattle takes place 

    ### disappointing hard code ###
    if defect.centered == False:
        defect._center()
    ### TODO: test with pbcs ###

    ####  I think probably, this should just be a generator.
    




def get_sites_in_r_of_point(structure: Structure, radius: float, point: np.array) -> list[Site]:
    """get sites in structure within +/- r of the center of
    the cell

    Args:
        structure (Structure): the structure to find the sites within
        radius (float): radius of sphere to search within

    Returns:
        list[Site]: sites within +/- r of center
    """
    a, b, c = structure.lattice.abc
    structure.add_site_property("index", range(len(structure)))
    sites = structure.get_sites_in_sphere([a * 0.5, b * 0.5, c * 0.5], radius)
    return sites


def get_frac_radii(structure: Structure, radius: float) -> np.ndarray:
    """convert radius in cartesian space into fractional space

    Args:
        structure (Structure): structure to get fractional radius in
        radius (float): radius in cartesian space

    Returns:
        np.ndarray[float]: radius in fractional space in each of x, y and z
    """
    return radius / np.array(structure.lattice.abc)


def get_min_max_near_center(structure: Structure, radius: float) -> tuple[float]:
    """given a radius in a structure, find the minimum and maximum values of
    fractional coordinates corresponding to that radius

    Args:
        structure (Structure): Structure to find limits in
        radius (float): radius in cartesian space.

    Returns:
        tuple(float, float): the minimum and maximum bounds on the sphere within
        `radius` of the cell center.
    """
    frac_radii = get_frac_radii(structure=structure, radius=radius)
    global_min = 0.5 - frac_radii
    global_max = 0.5 + frac_radii
    return global_min, global_max


def pick_random_coords(min: float, max: float) -> tuple[float]:
    """get random set of coordinates in fractional space between `min` and `max`

    Args:
        min (float): minimum value to find coordinates between
        max (float): maximum value to find coordinates between

    Returns:
        tuple[float]: a,b,c fractional coordinates between min and max
    """
    a = uniform(min[0], max[0])
    b = uniform(min[1], max[1])
    c = uniform(min[2], max[2])
    return a, b, c


def satisfies_min_sep(structure: Structure, min_sep: float) -> bool:
    """check whether any atoms in a structure are too close
    together based on a tolerance, `min_sep`.

    Args:
        structure (Structure): structure to assess
        min_sep (float): distance tolerance

    Returns:
        bool: True if min_sep satisfied, else False.
    """
    dm = [i for i in structure.distance_matrix.flatten() if i != 0]
    if any([d < min_sep for d in dm]):
        return False
    else:
        return True


def place_atoms(
    sites: list[Site],
    structure: Structure,
    min: float,
    max: float,
    min_sep: float,
    max_iters: int = 10000,
) -> Union[Structure, None]:
    """given a structure, replace atoms within min-max of center of the cell
    rejecting results that return atoms that are closer together than a specified
    minimum separation

    Args:
        sites (list[Site]): list of sites to remove and place
        structure (Structure): structure to do the replacing in
        min (float): min distance from r in fractional space
        max (float): max distance from r in fractional space
        min_sep (float): the minimum separation between atoms allowed for the
           random search
        max_iters (int, optional): the maximum number of trial placements to
          find result that satisfies `min sep`. Defaults to 10000.

    Returns:
        Union[Structure, None]: structure with atoms arranged.
    """
    for n in range(max_iters):
        live_structure = deepcopy(structure)
        live_structure.remove_sites([site.properties["index"] for site in sites])
        for site in sites:
            coords = pick_random_coords(min=min, max=max)
            live_structure.append(site.species, coords=coords)
        if satisfies_min_sep(structure=live_structure, min_sep=min_sep):
            return live_structure
    print(f"No structure found within max iter {max_iters}")
    return None


def jumble(
    structure: Structure,
    radius: float,
    min_sep: float = 1.23,
    max_iters: int = 100000,
    rattle: bool = True,
) -> Structure:
    """take a structure, and return randomly placed atoms within `radius` of the
    center of the cell

    Args:
        structure (Structure): structure to jumble
        radius (float): radius of sphere to jumble
        min_sep (float, optional): minimal separation between atoms allowed in the
          final structure. Defaults to 1.23.
        max_iters (int, optional): maximum attempts the algorithm will make to
          jumble the atoms in cell. Defaults to 10000.
        rattle (bool, optional): whether to apply a pseudorandom perturbation
          to all atomic positions in the structure after replacing atoms

    Returns:
        Structure: structure with rearranged atoms
    """
    sites = get_sites_in_r_of_center(structure, radius)
    min, max = get_min_max_near_center(structure, radius)
    jumbled_structure = place_atoms(
        sites=sites,
        structure=structure,
        min=min,
        max=max,
        min_sep=min_sep,
        max_iters=max_iters,
    )
    if rattle == True:
        mcrt = MCRT(rattle_std=0.2, min_distance=1.2)
        mcrt.apply_transformation(jumbled_structure)
    return jumbled_structure