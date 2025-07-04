from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter
from tqdm.contrib.concurrent import process_map
from scipy.spatial.distance import cosine as cos_distance
import numpy as np
from functools import partial
from rdkit.Chem.Scaffolds import MurckoScaffold

def cos_similarity(ref_counts, gen_counts):
    """
    Computes cosine similarity between
     dictionaries of form {name: count}. Non-present
     elements are considered zero:

     sim = <r, g> / ||r|| / ||g||
    """
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    return 1 - cos_distance(ref_vec, gen_vec)

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def fragmenter(mol):
    """
    fragment mol using BRICS and return smiles list
    """
    try:
        fgs = AllChem.FragmentOnBRICSBonds(get_mol(mol))
        fgs_smi = Chem.MolToSmiles(fgs).split(".")
        return fgs_smi
    except:
        pass


def compute_fragments(mol_list, max_workers=10, chunksize=100):
    """
    fragment list of mols using BRICS and return smiles list
    """
    fragments = Counter()
    results = process_map(fragmenter, mol_list, max_workers=max_workers, chunksize=chunksize)
    for res in results:
        if res:
            fragments.update(res)
    return fragments



def compute_scaffolds(mol_list, max_workers=10, min_rings=2):
    """
    Extracts a scafold from a molecule in a form of a canonic SMILES
    """
    scaffolds = Counter()
    results = process_map(partial(compute_scaffold, min_rings=min_rings), mol_list, max_workers=max_workers, chunksize=100)
    for res in results:
        if res:
            scaffolds.update([res])
    return scaffolds


def compute_scaffold(mol, min_rings=2):
    mol = get_mol(mol)
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except (ValueError, RuntimeError):
        return None
    n_rings = scaffold.GetRingInfo().NumRings()
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == '' or n_rings < min_rings:
        return None
    return scaffold_smiles
