from fileinput import filename
from defectivator.point_defects import PointDefectSet
from defectivator.defect_complexes import DefectComplexMaker
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import make_supercell
from defectivator.random_defect_structure import generate_random_defects

# define some example structure
bulk_supercell = bulk("LiF", "rocksalt", 4.1)
bulk_supercell = make_supercell(bulk_supercell, [[-2, 2, 2], [2, -2, 2], [2, 2, -2]])
bulk_supercell = AseAtomsAdaptor.get_structure(bulk_supercell)

# generate the host defect
ds = PointDefectSet(
    bulk_supercell,
    # extrinsic_species=["Fe"],
    charge_tol=5,
    interstitial_scheme=None
    )

structure = ds.vacancies[0]
for i in generate_random_defects(structure, 10, 2):
    i.to(filename = "vis.cif")


# for v in ds.interstitials:
#     v.structure.to(filename = "vis.cif")

