from fileinput import filename
from defectivator.point_defects import DefectSet
from defectivator.defect_complexes import DefectComplexMaker
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import make_supercell

# define some example structure
bulk_supercell = bulk("LiF", "rocksalt", 4.1)
bulk_supercell = make_supercell(bulk_supercell, [[-2, 2, 2], [2, -2, 2], [2, 2, -2]])
bulk_supercell = AseAtomsAdaptor.get_structure(bulk_supercell)

# generate the host defect
ds = DefectSet(
    bulk_supercell,
    extrinsic_species=["Fe"],
    charge_tol=5,
    interstitial_scheme= None
    )

for v in ds.vacancies:
    print(v.structure.composition)
