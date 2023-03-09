from dataclasses import dataclass
from scipy.spatial import Voronoi
from collections import defaultdict
import itertools
import numpy as np
from pymatgen.core import Lattice
from pymatgen.util.coord import pbc_diff
from scipy.spatial import ConvexHull


def get_facets(qhull_data):
    """
    Get the simplex facets for the Convex hull.
    Args:
        qhull_data (np.ndarray): The data from which to construct the convex
            hull as a Nxd array (N being number of data points and d being the
            dimension)
        joggle (boolean): Whether to joggle the input to avoid precision
            errors.
    Returns:
        List of simplices of the Convex Hull.
    """
    return ConvexHull(qhull_data, qhull_options="QJ i").simplices


def get_mapping(poly, vnodes, tol):
    """
    Helper function to check if a vornoi poly is a periodic image
    of one of the existing voronoi polys.
    """
    for v in vnodes:
        if v.is_image(poly, tol):
            return v
    return None


def calculate_vol(coords):
    """
    Calculate volume given a set of coords.
    :param coords: List of coords.
    :return: Volume
    """
    if len(coords) == 4:
        coords_affine = np.ones((4, 4))
        coords_affine[:, 0:3] = np.array(coords)
        return abs(np.linalg.det(coords_affine)) / 6

    simplices = get_facets(coords)
    center = np.average(coords, axis=0)
    vol = 0
    for s in simplices:
        c = list(coords[i] for i in s)
        c.append(center)
        vol += calculate_vol(c)
    return vol


@dataclass
class VoronoiPolyhedron:
    """
    Convenience container for a voronoi point in PBC and its associated polyhedron.
    """

    lattice: Lattice
    frac_coords: list[float]
    polyhedron_indicies: int
    polyhedron_coords: list[list[float]]
    name: str

    def is_image(self, poly, tol):
        """
        :param poly: VoronoiPolyhedron
        :param tol: Coordinate tolerance.
        :return: Whether a poly is an image of the current one.
        """
        frac_diff = pbc_diff(poly.frac_coords, self.frac_coords)
        if not np.allclose(frac_diff, [0, 0, 0], atol=tol):
            return False
        to_frac = self.lattice.get_fractional_coords
        for c1 in self.polyhedron_coords:
            found = False
            for c2 in poly.polyhedron_coords:
                d = pbc_diff(to_frac(c1), to_frac(c2))
                if not np.allclose(d, [0, 0, 0], atol=tol):
                    found = True
                    break
            if not found:
                return False
        return True

    @property
    def coordination(self):
        """
        :return: Coordination number
        """
        return len(self.polyhedron_indices)

    @property
    def volume(self):
        """
        :return: Volume
        """
        return calculate_vol(self.polyhedron_coords)


@dataclass
class InterstitialMap:
    structure: "pymatgen.core.Structure"
    tol: float = 0.0001
    check_volume: bool = True

    def __post_init__(self):
        structure = self.structure
        lattice = structure.lattice

        # Divide the sites into framework and non-framework sites.
        framework = [site for site in self.structure]

        # We construct a supercell series of coords. This is because the
        # Voronoi polyhedra can extend beyond the standard unit cell. Using a
        # range of -2, -1, 0, 1 should be fine.
        all_halo_coords = []
        halo_range = [-2, -1, 0, 1]
        for shift in itertools.product(halo_range, halo_range, halo_range):
            for site in framework:
                shifted = site.frac_coords + shift
                all_halo_coords.append(lattice.get_cartesian_coords(shifted))

        # Perform the voronoi tessellation.
        voro = Voronoi(all_halo_coords)

        # Store a mapping of each voronoi node to a set of points.
        node_points_map = defaultdict(set)
        for pts, vs in voro.ridge_dict.items():
            for v in vs:
                node_points_map[v].update(pts)

        # Vnodes store all the valid voronoi polyhedra. Cation vnodes store
        # the voronoi polyhedra that are already occupied by existing cations.
        vnodes = []

        # Filter all the voronoi polyhedra so that we only consider those
        # which are within the unit cell.
        for i, vertex in enumerate(voro.vertices):
            if i == 0:
                continue
            fcoord = lattice.get_fractional_coords(vertex)
            poly = VoronoiPolyhedron(
                lattice, fcoord, node_points_map[i], all_halo_coords, i
            )
            if np.all([-self.tol <= c < 1 + self.tol for c in fcoord]):
                if len(vnodes) == 0:
                    vnodes.append(poly)
                else:
                    ref = get_mapping(poly, vnodes, self.tol)
                    if ref is None:
                        vnodes.append(poly)

        self.coords = all_halo_coords
        self.vnodes = vnodes
        self.framework = framework

        for node in vnodes:
            structure.append("X", node.frac_coords)
        self.interstitial_map = structure
