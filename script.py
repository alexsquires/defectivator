from fileinput import filename
import yaml
from pymatgen.core import Structure
from point_defects import DefectSet
from defect_complexes import DefectComplexMaker
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import make_supercell

# define some example structure
bulk_supercell = bulk("LiF", "rocksalt", 4.1)
bulk_supercell = make_supercell(bulk_supercell, [[-2, 2, 2], [2, -2, 2], [2, 2, -2]])
bulk_supercell = AseAtomsAdaptor.get_structure(bulk_supercell)

# generate the host defect
ds = DefectSet(bulk_supercell, charge_tol=5, interstitial_scheme="voronoi")
dc = DefectComplexMaker(ds, "v_Li_1", "v_F", 1)

# how many complexes?
print(len(dc.defect_complexes))

# dump complexes to cifs
dc.defect_complexes[0]
for i,j in enumerate(dc.defect_complexes):
    j.to(filename=f"LiF_{i}.cif")