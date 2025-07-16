import os
import numpy as np
from tqdm.contrib.concurrent import process_map
from copy import deepcopy
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit.Chem.QED import qed
from posebusters import PoseBusters
from rdkit.Chem.FilterCatalog import *
from rdkit.Contrib.SA_Score import sascorer
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from src.utils.ConfSeq_3_2 import replace_angle_brackets_with_line
from src.utils.misc import save_pickle, load_pickle
import timeout_decorator


def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    logp = get_logp(mol)
    rule_4 = (logp >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


def get_rdkit_rmsd_single_mol(mol, n_conf=100, random_seed=42, num_workers=20):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the `n_conf` rdkit conformers
    """
    try:
        return get_rdkit_rmsd_single_mol_timeout(mol, n_conf, random_seed, num_workers)
    except:
        return [], []
    
@timeout_decorator.timeout(600)
def get_rdkit_rmsd_single_mol_timeout(mol, n_conf=100, random_seed=42, num_workers=20):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the `n_conf` rdkit conformers
    """
    mol = deepcopy(mol)
    rmsd_list = []
    # predict 3d
    try:
        Chem.SanitizeMol(mol)
        mol3d = Chem.AddHs(mol)
        confIds = AllChem.EmbedMultipleConfs(mol3d, n_conf, randomSeed=random_seed, numThreads=num_workers)
        for confId in confIds:
            AllChem.MMFFOptimizeMolecule(mol3d, confId=confId)
            rmsd = Chem.rdMolAlign.GetBestRMS(mol, mol3d, refId=confId)
            rmsd_list.append(rmsd)
        rmsd_list = np.array(rmsd_list)
        return rmsd_list
    except:
        return []
    
def get_rdkit_rmsd(mols, save_path):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the `n_conf` rdkit conformers
    """
    if os.path.exists(save_path):
        results = load_pickle(save_path)
    else:
        results = process_map(get_rdkit_rmsd_single_mol, mols, max_workers=20, chunksize=10)
        save_pickle(results, save_path)

    min_rmsd = []
    for rmsd in results:
        if np.size(rmsd) > 0:
            min_rmsd.append(np.min(rmsd))
    min_rmsd = np.array(min_rmsd)    
    df = pd.DataFrame({'RMSD': [np.mean(min_rmsd)]})
    return df


def get_logp(mol):
    return Crippen.MolLogP(mol)


def safe_calculate(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except:
        return np.nan

def get_druglike_properties(mol):
    """
    Calculate drug-likeness metrics for a single molecule and return a dictionary containing SMILES and various metrics.
    """
    # 1. First convert mol to SMILES (preserving stereochemical information)
    try:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        smiles = None

    # 2. Various calculation function mappings
    calculations = {
        'QED':      lambda m: qed(m),
        'SAS':      lambda m: sascorer.calculateScore(m),
        'logP':     lambda m: get_logp(m),
        'Lipinski': lambda m: obey_lipinski(m),
        'TPSA':     lambda m: Descriptors.TPSA(m),
    }

    # 3. Safe calculation for each item
    results = {'SMILES': smiles}
    for name, func in calculations.items():
        results[name] = safe_calculate(func, mol)

    return results


def compute_druglike_properties(mols, save_path=None, max_workers=10, disable_tqdm=False):
    """
    Calculate drug-likeness metrics for a batch of molecules in parallel, returning a DataFrame containing SMILES and all metrics.
    
    Parameters:
      - mols: Iterable list of RDKit Mol objects
      - save_path: If not None, save results as CSV
      - max_workers: Number of parallel processes
      - disable_tqdm: Whether to disable progress bar
    
    Returns:
      - pandas.DataFrame, each row corresponds to a molecule, columns include SMILES, QED, SAS, logP, Lipinski, TPSA
    """
    # Parallel mapping calculation
    props_list = process_map(
        get_druglike_properties,
        mols,
        max_workers=max_workers,
        disable=disable_tqdm
    )

    # Construct DataFrame, first column is SMILES
    df = pd.DataFrame(props_list, columns=['SMILES', 'QED', 'SAS', 'logP', 'Lipinski', 'TPSA'])

    # Optionally save
    if save_path is not None:
        df.to_csv(save_path, index=False)

    return df



def cal_mean_druglike_properties(df):
    """
    Calculate the mean of drug-like properties.
    Parameters:
    df (DataFrame): Input DataFrame containing drug-like properties.
    Returns:
    DataFrame: A DataFrame summarizing the mean of the properties.
    """

    # Define the properties to evaluate
    properties = ['QED', 'SAS', 'logP', 'TPSA']
    results = {}

    # Calculate mean for each property
    for prop in properties:
        results[prop] = df[prop].mean()

    # Calculate the proportion of Lipinski rule matches
    results['Lipinski'] = (df['Lipinski'] == 5).sum() / len(df)

    # Create a summary DataFrame with raw numerical results
    result_df = pd.DataFrame({
        'QED': [results['QED']],
        'SAS': [results['SAS']],
        'logP': [results['logP']],
        'TPSA': [results['TPSA']],
        'Lipinski': [results['Lipinski']]
    })

    return result_df

def my_bust(mol):
    try:
        return my_bust_timeout(mol)
    except:
        return None

@timeout_decorator.timeout(120)
def my_bust_timeout(mol):
    buster = PoseBusters(config='mol')
    df = buster.bust(mol, full_report=True).reset_index()
    if df is not None:
        return df


def compute_posebusters_parallel(input, save_path=None, max_workers=10, chunksize=20, disable_tqdm=False):
    if isinstance(input, str):
        sdf_path = input
        mols = [mol for mol in Chem.SDMolSupplier(sdf_path) if mol is not None]
    else:
        mols = input
    results = process_map(my_bust, mols, max_workers=max_workers, chunksize=chunksize, disable=disable_tqdm)
    df = pd.DataFrame()
    for result in results:
        if result is not None:
            df = pd.concat([df, result])
    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df


def compute_posebusters(sdf_path, save_path=None):
    """
    Compute PoseBusters validation metrics for molecules in an SDF file.

    Parameters:
    - sdf_path: Path to the SDF file containing molecules.

    Returns:
    - df: A DataFrame containing PoseBusters validation results.
    """
    buster = PoseBusters(config='mol')
    df = buster.bust(sdf_path, None, None, full_report=True).reset_index()
    if save_path is not None:
        df.to_csv(save_path, index=False)

    return df


def get_posebusters_summary(df, num_samples=10000):
    df_check = df[['mol_pred_loaded', 
                'sanitization', 
                'inchi_convertible',
                'all_atoms_connected',
                'bond_lengths',
                'bond_angles',
                'internal_steric_clash',
                'aromatic_ring_flatness',
                'double_bond_flatness',
                'internal_energy',
                'passes_valence_checks',
                'passes_kekulization']]
    
    df_check = df_check.astype(bool)
    # Calculate the mean of each column
    result_dict = {col: df_check[col].sum()/ num_samples for col in df_check.columns}
    result_df = pd.DataFrame(result_dict, index=[0])

    valid = df_check[df_check.all(axis=1)]
    result_df['PB_valid'] = valid.shape[0] / num_samples
    
    return result_df


def compute_basic_metrics_confseq(gen_smiles, train_smiles, num_samples=10000):
    # validity
    valid = []
    for smi in gen_smiles:
        in_smiles = replace_angle_brackets_with_line(smi)
        in_smiles = in_smiles.replace('^ |','')
        in_smiles = in_smiles.replace(' !','')
        in_smiles = in_smiles.replace('/ -','/').replace('\\ -','\\')
        smiles = ''.join(in_smiles.split())
        try:
            mol = Chem.MolFromSmiles(smiles)
            Chem.SanitizeMol(mol)
            # if check_validity(mol):
            #     valid.append(Chem.MolToSmiles(mol))
            valid.append(Chem.MolToSmiles(mol))
        except:
            pass

    validity = len(valid) / num_samples if num_samples > 0 else 0.0

    # uniqueness
    uniqueness = len(set(valid)) / len(valid) if len(valid) > 0 else 0.0

    # validity * uniqueness
    validity_plus_uniqueness = validity * uniqueness

    # novelty
    if train_smiles is None:
        novelty = np.nan
    else:
        novelty = len(set(valid) - set(train_smiles)) / len(set(valid)) if len(valid) > 0 else 0.0

    # Create a DataFrame for the metrics
    df = pd.DataFrame({
        'Validity': [validity],
        'Uniqueness': [uniqueness],
        'Validity * Uniqueness': [validity_plus_uniqueness],
        'Novelty': [novelty],
    })

    return df