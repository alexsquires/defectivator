from pymatgen.core import Structure, Site
from defectivator.tools import embiggen, get_defect_coords, get_prim_to_bulk_map
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
import numpy as np

# Load the two structures
smaller_structure = Structure.from_file("tests/POSCAR_rel_int_test.vasp")
larger_structure = Structure.from_file("tests/POSCAR_rel_bulk_test.vasp")
primitive_structure = larger_structure.get_primitive_structure()

matrix = get_prim_to_bulk_map(smaller_structure, larger_structure)
smaller_structure.make_supercell(matrix)

# temp = larger_structure()
# smaller_structure.to(filename = "vis.cif")

print(np.matmul([0.5, 0.5, 0.5], matrix))