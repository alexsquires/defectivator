from dataclasses import dataclass
from pymatgen.core import Structure, Site
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import PeriodicSite
from monty.json import MSONable

class Path:
    def __init__(self, iindex, eindex):
        self.iindex = iindex
        self.eindex = eindex

    def __hash__(self):
        return hash(frozenset([self.iindex, self.eindex]))

    def __eq__(self, other):
        if isinstance(other, Path):
            return frozenset([self.iindex, self.eindex]) == frozenset([other.iindex, other.eindex])
        return False
    
@dataclass
class PathFinder(MSONable):
    structure: Structure
    migrating_specie: str
    max_path_length: float = None
    symprec: float = 0.1

    """
    Determines symmetrically distinct paths between existing sites.
    """

    def __post_init__(self):
        a = SpacegroupAnalyzer(self.structure, symprec=self.symprec)
        self.symm_structure = a.get_symmetrized_structure()

    def get_paths(self):
        """
        Returns:
            list of dict: All distinct migration paths with additional symmetry information.
        """
        if self.max_path_length is None:
            raise ValueError("max_path_length cannot be None.")

        paths = set()
        for sites in self.symm_structure.equivalent_sites:
            if sites[0].species_string == self.migrating_specie:
                isite = sites[0]
                for nn in self.symm_structure.get_neighbors(isite, r=round(self.max_path_length, 3) + 0.01):
                    if nn.species_string == self.migrating_specie:
                        esite = nn
                        hop = self.get_hop(isite, esite)
                        paths.add(Path(hop['iindex'], hop['eindex']))

        # Convert paths back to list of dictionaries for compatibility with the rest of the code
        paths = [dict(iindex=path.iindex, eindex=path.eindex) for path in paths]

        return paths

    def get_hop(self, isite, esite):
        """
        Calculate the symmetry information for a migration path.
        """
        hop = {
            'isite': isite,
            'esite': esite,
            'msite': PeriodicSite(esite.specie, (isite.frac_coords + esite.frac_coords) / 2, esite.lattice)
        }
        sg = self.symm_structure.spacegroup
        for i, sites in enumerate(self.symm_structure.equivalent_sites):
            if sg.are_symmetrically_equivalent([isite], [sites[0]]):
                hop['iindex'] = i
            if sg.are_symmetrically_equivalent([esite], [sites[0]]):
                hop['eindex'] = i

        if 'iindex' not in hop:
            for i, sites in enumerate(self.symm_structure.equivalent_sites):
                for itr_site in sites:
                    if sg.are_symmetrically_equivalent([isite], [itr_site]):
                        hop['iindex'] = i
                        break
                else:
                    continue
                break

        if 'eindex' not in hop:
            for i, sites in enumerate(self.symm_structure.equivalent_sites):
                for itr_site in sites:
                    if sg.are_symmetrically_equivalent([esite], [itr_site]):
                        hop['eindex'] = i
                        break
                else:
                    continue
                break

        if 'iindex' not in hop:
            raise RuntimeError(f"No symmetrically equivalent site was found for {isite}")
        if 'eindex' not in hop:
            raise RuntimeError(f"No symmetrically equivalent site was found for {esite}")