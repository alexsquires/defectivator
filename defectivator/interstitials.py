from scipy.spatial import Voronoi
from collections import defaultdict
import itertools
import numpy as np
from pymatgen.analysis.phase_diagram import get_facets

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

    simplices = get_facets(coords, joggle=True)
    center = np.average(coords, axis=0)
    vol = 0
    for s in simplices:
        c = list(coords[i] for i in s)
        c.append(center)
        vol += calculate_vol(c)
    return vol

class VoronoiPolyhedron:
    """
    Convenience container for a voronoi point in PBC and its associated polyhedron.
    """

    def __init__(self, lattice: Lattice, frac_coords, polyhedron_indices, all_coords, name=None):
        """
        :param lattice:
        :param frac_coords:
        :param polyhedron_indices:
        :param all_coords:
        :param name:
        """
        self.lattice = lattice
        self.frac_coords = frac_coords
        self.polyhedron_indices = polyhedron_indices
        self.polyhedron_coords = np.array(all_coords)[list(polyhedron_indices), :]
        self.name = name

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

    def __str__(self):
        return f"Voronoi polyhedron {self.name}"

class InterstitialMap:
    """
    This is a generalized module to perform topological analyses of a crystal
    structure using Voronoi tessellations. It can be used for finding potential
    interstitial sites. Applications including using these sites for
    inserting additional atoms or for analyzing diffusion pathways.
    Note that you typically want to do some preliminary postprocessing after
    the initial construction. The initial construction will create a lot of
    points, especially for determining potential insertion sites. Some helper
    methods are available to perform aggregation and elimination of nodes. A
    typical use is something like::
        a = TopographyAnalyzer(structure, ["O"], ["P"])
        a.cluster_nodes()
        a.remove_collisions()
    """

    def __init__(
        self,
        structure,
        framework_ions,
        cations,
        tol=0.0001,
        max_cell_range=1,
        check_volume=True,
        constrained_c_frac=0.5,
        thickness=0.5,
    ):
        """
        Init.
        Args:
            structure (Structure): An initial structure.
            framework_ions ([str]): A list of ions to be considered as a
                framework. Typically, this would be all anion species. E.g.,
                ["O", "S"].
            cations ([str]): A list of ions to be considered as non-migrating
                cations. E.g., if you are looking at Li3PS4 as a Li
                conductor, Li is a mobile species. Your cations should be [
                "P"]. The cations are used to exclude polyhedra from
                diffusion analysis since those polyhedra are already occupied.
            tol (float): A tolerance distance for the analysis, used to
                determine if something are actually periodic boundary images of
                each other. Default is usually fine.
            max_cell_range (int): This is the range of periodic images to
                construct the Voronoi tessellation. A value of 1 means that we
                include all points from (x +- 1, y +- 1, z+- 1) in the
                voronoi construction. This is because the Voronoi poly
                extends beyond the standard unit cell because of PBC.
                Typically, the default value of 1 works fine for most
                structures and is fast. But for really small unit
                cells with high symmetry, you may need to increase this to 2
                or higher.
            check_volume (bool): Set False when ValueError always happen after
                tuning tolerance.
            constrained_c_frac (float): Constraint the region where users want
                to do Topology analysis the default value is 0.5, which is the
                fractional coordinate of the cell
            thickness (float): Along with constrained_c_frac, limit the
                thickness of the regions where we want to explore. Default is
                0.5, which is mapping all the site of the unit cell.
        """
        self.structure = structure
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
        cation_vnodes = []

        def get_mapping(poly):
            """
            Helper function to check if a vornoi poly is a periodic image
            of one of the existing voronoi polys.
            """
            for v in vnodes:
                if v.is_image(poly, tol):
                    return v
            return None

        # Filter all the voronoi polyhedra so that we only consider those
        # which are within the unit cell.
        for i, vertex in enumerate(voro.vertices):
            if i == 0:
                continue
            fcoord = lattice.get_fractional_coords(vertex)
            poly = VoronoiPolyhedron(lattice, fcoord, node_points_map[i], all_halo_coords, i)
            if np.all([-tol <= c < 1 + tol for c in fcoord]):
                if len(vnodes) == 0:
                    vnodes.append(poly)
                else:
                    ref = get_mapping(poly)
                    if ref is None:
                        vnodes.append(poly)

        self.coords = all_halo_coords
        self.vnodes = vnodes
        self.framework = framework
        if check_volume:
            self.check_volume()

    def check_volume(self):
        """
        Basic check for volume of all voronoi poly sum to unit cell volume
        Note that this does not apply after poly combination.
        """
        vol = sum(v.volume for v in self.vnodes)
        if abs(vol - self.structure.volume) > 1e-8:
            raise ValueError(
                "Sum of voronoi volumes is not equal to original volume of "
                "structure! This may lead to inaccurate results. You need to "
                "tweak the tolerance and max_cell_range until you get a "
                "correct mapping."
            )
