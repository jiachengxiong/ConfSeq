import os
import numpy as np
import torch
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from copy import deepcopy
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit.Chem.QED import qed
from posebusters import PoseBusters
from easydict import EasyDict
from src.utils.docking import QVinaDockingTask
# from utils.datasets import get_dataset
from rdkit.Chem.FilterCatalog import *
from multiprocessing import Pool
from functools import partial
from collections import Counter
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem.TorsionFingerprints import GetTFDBetweenConformers
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from fcd_torch import FCD
import logging
from pathlib import Path
from typing import List, Optional, Any

from src.utils.ConfSeq_3_2 import replace_angle_brackets_with_line
from src.utils.similarity import (get_shape_similarity_matrix, 
                                  get_tanimoto_similarity_matrix,
                                  get_shape_similarity_matrix_shaep,
                                  get_tanimoto_similarity_matrix_rdkfingerprint,
                                  cal_intdiv,
                                  cal_sumbottleneck,
                                  cal_num_circles,
                                  cal_snn)
from src.utils.evaluation import check_validity
from src.utils.moses import *
from src.utils.cal_geometry import get_sub_geometry_metric
from src.utils.datasets_config import geom_with_h_1
from src.utils.misc import save_pickle, load_pickle
import timeout_decorator


def is_pains(mol):
    params_pain = FilterCatalogParams()
    params_pain.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    catalog_pain = FilterCatalog(params_pain)
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    entry = catalog_pain.GetFirstMatch(mol)
    if entry is None:
        return False
    else:
        return True


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
    

def get_basic(mol):
    n_atoms = len(mol.GetAtoms())
    n_bonds = len(mol.GetBonds())
    n_rings = len(Chem.GetSymmSSSR(mol))
    weight = Descriptors.ExactMolWt(mol)
    return n_atoms, n_bonds, n_rings, weight


def get_rdkit_rmsd_tfd_single_mol(mol, n_conf=100, random_seed=42, num_workers=20):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the `n_conf` rdkit conformers
    """
    try:
        return get_rdkit_rmsd_tfd_single_mol_timeout(mol, n_conf, random_seed, num_workers)
    except:
        return [], []
    
@timeout_decorator.timeout(600)
def get_rdkit_rmsd_tfd_single_mol_timeout(mol, n_conf=100, random_seed=42, num_workers=20):
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
        mol3d = Chem.RemoveHs(mol3d)
        new_confid = mol3d.AddConformer(mol.GetConformer(), assignId=True)
        tfd_list = GetTFDBetweenConformers(mol3d, confIds1=[new_confid], confIds2=[i for i in range(new_confid)])

        tfd_list = np.array(tfd_list)
        rmsd_list = np.array(rmsd_list)
        return tfd_list, rmsd_list
    except:
        return [], []
    
def get_rdkit_rmsd_tfd(mols, save_path):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the `n_conf` rdkit conformers
    """
    if os.path.exists(save_path):
        results = load_pickle(save_path)
    else:
        results = process_map(get_rdkit_rmsd_tfd_single_mol, mols, max_workers=20, chunksize=10)
        save_pickle(results, save_path)

    min_rmsd, min_tfd = [], []
    for rmsd, tfd in results:
        if np.size(rmsd) > 0:
            min_rmsd.append(np.min(rmsd))
        if np.size(tfd) > 0:
            min_tfd.append(np.min(tfd))
    
    df = pd.DataFrame({'RMSD': [np.mean(min_rmsd)], 'TFD': [np.mean(min_tfd)]})
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
    计算单个分子的药物相似性指标，并返回一个包含 SMILES 和各项指标的字典。
    """
    # 1. 先把 mol 转成 SMILES（保留立体化学信息）
    try:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        smiles = None

    # 2. 各项计算函数映射
    calculations = {
        'QED':      lambda m: qed(m),
        'SAS':      lambda m: sascorer.calculateScore(m),
        'logP':     lambda m: get_logp(m),
        'Lipinski': lambda m: obey_lipinski(m),
        'TPSA':     lambda m: Descriptors.TPSA(m),
    }

    # 3. 逐项安全计算
    results = {'SMILES': smiles}
    for name, func in calculations.items():
        results[name] = safe_calculate(func, mol)

    return results


def compute_druglike_properties(mols, save_path=None, max_workers=10, disable_tqdm=False):
    """
    并行计算一批分子的药物相似性指标，返回包含 SMILES 和所有指标的 DataFrame。
    
    参数:
      - mols: 可迭代的 RDKit Mol 对象列表
      - save_path: 若不为 None，则将结果保存为 CSV
      - max_workers: 并行进程数
      - disable_tqdm: 是否屏蔽进度条
    
    返回:
      - pandas.DataFrame，每行对应一个分子，列包括 SMILES、QED、SAS、logP、Lipinski、TPSA
    """
    # 并行映射计算
    props_list = process_map(
        get_druglike_properties,
        mols,
        max_workers=max_workers,
        disable=disable_tqdm
    )

    # 构造 DataFrame，第一列就是 SMILES
    df = pd.DataFrame(props_list, columns=['SMILES', 'QED', 'SAS', 'logP', 'Lipinski', 'TPSA'])

    # 可选地保存
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


def get_molecule_force_field(mol, conf_id=None, force_field='mmff', **kwargs):
    """
    Get a force field for a molecule.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    conf_id : int, optional
        ID of the conformer to associate with the force field.
    force_field : str, optional
        Force Field name.
    kwargs : dict, optional
        Keyword arguments for force field constructor.
    """
    if force_field == 'uff':
        ff = AllChem.UFFGetMoleculeForceField(
            mol, confId=conf_id, **kwargs)
    elif force_field.startswith('mmff'):
        AllChem.MMFFSanitizeMolecule(mol)
        mmff_props = AllChem.MMFFGetMoleculeProperties(
            mol, mmffVariant=force_field)
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, mmff_props, confId=conf_id, **kwargs)
    else:
        raise ValueError("Invalid force_field {}".format(force_field))
    return ff


def get_conformer_energies(mol, force_field='mmff'):
    """
    Calculate conformer energies.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    force_field : str, optional
        Force Field name.
    Returns
    -------
    energies : array_like
        Minimized conformer energies.
    """
    energies = []
    for conf in mol.GetConformers():
        ff = get_molecule_force_field(mol, conf_id=conf.GetId(), force_field=force_field)
        energy = ff.CalcEnergy()
        energies.append(energy)
    energies = np.asarray(energies, dtype=float)
    return energies

def cal_fcd(ref_smiles, gen_smiles):
    try:
        fcd = FCD(device='cuda:0', n_jobs=10)
        fcd_value = fcd(ref=ref_smiles, gen=gen_smiles)
    except:
        fcd_value = np.nan
    return fcd_value

def cal_snn_metric(ref_smiles, gen_smiles):
    try:
        matrix = get_tanimoto_similarity_matrix(ref_smiles, gen_smiles)
        snn_value = cal_snn(matrix)
    except:
        snn_value = np.nan
    return snn_value

def cal_frag_metric(ref_smiles, gen_smiles):
    try:
        ref_frag = compute_fragments(ref_smiles)
        gen_frag = compute_fragments(gen_smiles)
        frag_value = cos_similarity(ref_frag, gen_frag)
    except:
        frag_value = np.nan
    return frag_value

def cal_scaf_metric(ref_smiles, gen_smiles):
    try:
        ref_scaf = compute_scaffolds(ref_smiles)
        gen_scaf = compute_scaffolds(gen_smiles)
        scaf_value = cos_similarity(ref_scaf, gen_scaf)
    except:
        scaf_value = np.nan
    return scaf_value


# class SimilarityWithMe:
#     def __init__(self, mol) -> None:
#         self.mol = deepcopy(mol)
#         self.mol = Chem.MolFromSmiles(Chem.MolToSmiles(self.mol))
#         self.fp= Chem.RDKFingerprint(self.mol)
#
#     def get_sim(self, mol):
#         mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # automately sanitize
#         fg_query = Chem.RDKFingerprint(mol)
#         sims = DataStructs.TanimotoSimilarity(self.fp, fg_query)
#         return sims
#
# class SimilarityWithTrain:
#     def __init__(self, base_dir='.') -> None:
#         self.cfg_dataset = EasyDict({
#             'name': 'pl',
#             'path': os.path.join(base_dir,  'data/crossdocked_pocket10'),
#             'split': os.path.join(base_dir, 'data/crossdocked_pocket10_split.pt'),
#             'fingerprint': os.path.join(base_dir, 'data/crossdocked_pocket10_fingerprint.pt'),
#             'smiles': os.path.join(base_dir, 'data/crossdocked_pocket10_smiles.pt'),
#         })
#         self.train_smiles = None
#         self.train_fingers = None
#
#     def _get_train_mols(self):
#         file_not_exists = (not os.path.exists(self.cfg_dataset.fingerprint)) or (not os.path.exists(self.cfg_dataset.smiles))
#         if file_not_exists:
#             _, subsets = get_dataset(config = self.cfg_dataset)
#             train_set = subsets['train']
#             self.train_smiles = []
#             self.train_fingers = []
#             for data in tqdm(train_set):  # calculate fingerprint and smiles of train data
#                 data.ligand_context_pos = data.ligand_pos
#                 data.ligand_context_element = data.ligand_element
#                 data.ligand_context_bond_index = data.ligand_bond_index
#                 data.ligand_context_bond_type = data.ligand_bond_type
#                 mol = reconstruct_from_generated_with_edges(data, sanitize=True)
#                 mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # automately sanitize
#                 smiles = Chem.MolToSmiles(mol)
#                 fg = Chem.RDKFingerprint(mol)
#                 self.train_fingers.append(fg)
#                 self.train_smiles.append(smiles)
#             self.train_smiles = np.array(self.train_smiles)
#             # self.train_fingers = np.array(self.train_fingers)
#             torch.save(self.train_smiles, self.cfg_dataset.smiles)
#             torch.save(self.train_fingers, self.cfg_dataset.fingerprint)
#         else:
#             self.train_smiles = torch.load(self.cfg_dataset.smiles)
#             self.train_fingers = torch.load(self.cfg_dataset.fingerprint)
#             self.train_smiles = np.array(self.train_smiles)
#             # self.train_fingers = np.array(self.train_fingers)
#
#     def _get_uni_mols(self):
#         self.train_uni_smiles, self.index_in_train = np.unique(self.train_smiles, return_index=True)
#         self.train_uni_fingers = [self.train_fingers[idx] for idx in self.index_in_train]
#
#     def get_similarity(self, mol):
#         if self.train_fingers is None:
#             self._get_train_mols()
#             self._get_uni_mols()
#         mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # automately sanitize
#         fp_mol = Chem.RDKFingerprint(mol)
#         # sim_func = DataStructs.TanimotoSimilarity
#         # with Pool(32) as pool:
#         #     sims = pool.map(partial(sim_func, bv1=fp_mol), self.train_uni_fingers)
#         sims = [DataStructs.TanimotoSimilarity(fp, fp_mol) for fp in self.train_uni_fingers]
#         return np.array(sims)
#
#
#     def get_top_sims(self, mol, top=3):
#         similarities = self.get_similarity(mol)
#         idx_sort = np.argsort(similarities)[::-1]
#         top_scores = similarities[idx_sort[:top]]
#         top_smiles = self.train_uni_smiles[idx_sort[:top]]
#         return top_scores, top_smiles


def compute_basic_metrics_confseq(gen_smiles, train_smiles, test_smiles, num_samples=10000):
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

    # FCD, SNN, Frag, Scaf
    if test_smiles is None:
        fcd_value = np.nan
        snn_value = np.nan
        frag_value = np.nan
        scaf_value = np.nan
    else:
        print('Computing FCD...')
        fcd_value = cal_fcd(ref_smiles=test_smiles, gen_smiles=valid)
        print('Computing SNN...')
        snn_value = cal_snn_metric(test_smiles, valid)
        print('Computing Frag...')
        frag_value = cal_frag_metric(test_smiles, valid)
        print('Computing Scaf...')
        scaf_value = cal_scaf_metric(test_smiles, valid)

    # Create a DataFrame for the metrics
    df = pd.DataFrame({
        'Validity': [validity],
        'Uniqueness': [uniqueness],
        'Validity * Uniqueness': [validity_plus_uniqueness],
        'Novelty': [novelty],
        'FCD': [fcd_value],
        'SNN': [snn_value],
        'Frag': [frag_value],
        'Scaf': [scaf_value]
    })

    return df


def compute_basic_metrics_baseline(mols, train_smiles, test_smiles, num_samples=10000):
    # validity
    valid = []
    for mol in mols:
        try:
            Chem.SanitizeMol(mol)
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            Chem.SanitizeMol(largest_mol)
            # if check_validity(largest_mol):
            #     valid.append(Chem.MolToSmiles(largest_mol))
            valid.append(Chem.MolToSmiles(largest_mol))
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

    # FCD, SNN, Frag, Scaf
    if test_smiles is None:
        fcd_value = np.nan
        snn_value = np.nan
        frag_value = np.nan
        scaf_value = np.nan
    else:
        print('Computing FCD...')
        fcd_value = cal_fcd(ref_smiles=test_smiles, gen_smiles=valid)
        print(f'FCD: {fcd_value}')
        print('Computing SNN...(very slow, may take 10 min or more)')
        snn_value = cal_snn_metric(test_smiles, valid)  # this step is very slow
        print(f'SNN: {snn_value}')
        print('Computing Frag...')
        frag_value = cal_frag_metric(test_smiles, valid)
        print(f'Frag: {frag_value}')
        print('Computing Scaf...')
        scaf_value = cal_scaf_metric(test_smiles, valid)
        print(f'Scaf: {scaf_value}')

    # Create a DataFrame for the metrics
    df = pd.DataFrame({
        'Validity': [validity],
        'Uniqueness': [uniqueness],
        'Validity * Uniqueness': [validity_plus_uniqueness],
        'Novelty': [novelty],
        'FCD': [fcd_value],
        'SNN': [snn_value],
        'Frag': [frag_value],
        'Scaf': [scaf_value]
    })

    return df


def flatten_similarity_data(
    shape_list: List[np.ndarray],
    graph_list: List[np.ndarray],
    score_list: Optional[List[List[float]]] = None
) -> pd.DataFrame:
    """
    将成对的相似度数组扁平化为 DataFrame。

    Parameters
    ----------
    shape_list : List[np.ndarray]
        每组中 shape 相似度数组列表。
    graph_list : List[np.ndarray]
        每组中 graph 相似度数组列表。
    score_list : Optional[List[List[float]]]
        每组中分数列表，若为 None 则不包含 'score' 列。

    Returns
    -------
    pd.DataFrame
        包含列 ['group_id','mol_id','shape_similarity','graph_similarity']
        以及可选的 'score' 列。
    """
    records = []
    for group_id, (s_arr, g_arr) in enumerate(zip(shape_list, graph_list)):
        s_flat = np.asarray(s_arr).ravel()
        g_flat = np.asarray(g_arr).ravel()
        if score_list is not None:
            score_flat = np.asarray(score_list[group_id]).ravel()
            for mol_id, (s, g, sc) in enumerate(zip(s_flat, g_flat, score_flat)):
                records.append({
                    'group_id':       group_id,
                    'mol_id':         mol_id,
                    'shape_similarity':  s,
                    'graph_similarity':  g,
                    'score': sc
                })
        else:
            for mol_id, (s, g) in enumerate(zip(s_flat, g_flat)):
                records.append({
                    'group_id':         group_id,
                    'mol_id':           mol_id,
                    'shape_similarity':    s,
                    'graph_similarity':    g
                })

    cols = ['group_id', 'mol_id', 'shape_similarity', 'graph_similarity']
    if score_list is not None:
        cols.append('score')
    return pd.DataFrame.from_records(records, columns=cols)


def compute_similarity_dataframe(
    ref_mols: List[Any],
    gen_data: List[List[Any]],
    method: str = 'shaep',
    save_path: Optional[str] = None,
    has_scores: bool = True
) -> pd.DataFrame:
    """
    针对参考分子和生成分子批次，计算 shape 与 Tanimoto 相似度，
    并以 DataFrame 形式返回（可选保存为 CSV）。

    Parameters
    ----------
    ref_mols : List[Mol]
        参考分子列表。
    gen_data : List[List[Mol]]
        生成分子按批次组织的列表。
    method : str, default 'shaep'
        相似度计算方法，可选 'shaep' 或 'rdkit'。
    save_path : Optional[str]
        若提供，则保存结果为 CSV 文件。
    has_scores : bool, default True
        是否从分子属性中提取 'score' 字段。

    Returns
    -------
    pd.DataFrame
        包含相似度及可选分数的 DataFrame。
    """
    shape_list, graph_list = [], []
    for idx, batch in enumerate(tqdm(gen_data, desc='计算相似度')):
        try:
            if method == 'shaep':
                s = get_shape_similarity_matrix_shaep([ref_mols[idx]], batch).flatten()
                g = get_tanimoto_similarity_matrix([ref_mols[idx]], batch).flatten()
            elif method == 'rdkit':
                s = get_shape_similarity_matrix([ref_mols[idx]], batch).flatten()
                g = get_tanimoto_similarity_matrix([ref_mols[idx]], batch).flatten()
            else:
                raise ValueError(f'未知 method: {method}')
        except Exception as e:
            logging.warning(f'第 {idx} 组计算失败，使用空数组替代：{e}')
            s, g = np.array([]), np.array([])

        shape_list.append(s)
        graph_list.append(g)

    scores = None
    if has_scores:
        scores = [
            [float(mol.GetProp('score')) for mol in batch]
            for batch in gen_data
        ]

    df = flatten_similarity_data(shape_list, graph_list, scores)

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df



def compute_similarity_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 group_id 分组后，计算 shape_similarity 和 graph_similarity 的组内均值，
    并对这些组均值分别计算整体均值和标准差，最终以“均值±标准差”字符串形式返回。

    参数：
    - df: DataFrame，必须包含以下三列
        • group_id          ：分组标识
        • shape_similarity  ：形状相似度
        • graph_similarity  ：图谱（Tanimoto）相似度

    返回：
    - summary: DataFrame，索引为 ['shape', 'graph']，
        列为 ['mean±std']，数值均为格式化后的字符串形式。
    """
    # 1. 按组聚合，得到每组的均值
    df_group = (
        df
        .groupby('group_id', as_index=False)
        .agg(
            shape_mean=('shape_similarity', 'mean'),
            graph_mean=('graph_similarity', 'mean')
        )
    )

    # 2. 分别计算“组均值”的整体均值 & 标准差
    shape_mean_of_means = df_group['shape_mean'].mean()
    shape_std_of_means  = df_group['shape_mean'].std(ddof=0)
    graph_mean_of_means = df_group['graph_mean'].mean()
    graph_std_of_means  = df_group['graph_mean'].std(ddof=0)

    # 3. 格式化为 “均值±标准差” 的字符串，保留三位小数
    summary = pd.DataFrame({
        'Avg_shape': [
            f'{shape_mean_of_means:.3f} ± {shape_std_of_means:.3f}'
        ],
        'Avg_graph': [
            f'{graph_mean_of_means:.3f} ± {graph_std_of_means:.3f}'
        ]
    })

    return summary



def compute_diversity_metrics(mols, logger=None, threshold=0.75):
    logger.info('Computing tanimoto similarity matrix...')
    similarity_matrix = get_tanimoto_similarity_matrix(mols, mols)

    # intdiv 
    logger.info('Computing IntDiv...')
    intdiv = cal_intdiv(similarity_matrix, p=1)

    # sumbottleneck
    logger.info('Computing SumBottleneck...')
    sumbottleneck = cal_sumbottleneck(similarity_matrix)

    # num_circles
    logger.info('Computing NumCircles...')
    num_circles = cal_num_circles(len(mols), similarity_matrix, threshold=threshold)

    data = {
        'IntDiv': intdiv,
        'SumBottleneck': sumbottleneck,
        'NumCircles': num_circles
    }

    df = pd.DataFrame(data, index=[0])
    
    return df


def compute_geometry_metrics(gen_mols):
    root_path = 'data/geom/geom_sdf/'
    dataset_info = geom_with_h_1
    if os.path.exists(os.path.join(root_path, 'target_geometry_stat.pkl')):
        test_mols = None
    else:
        test_mols = [mol for mol in Chem.SDMolSupplier(os.path.join(root_path, 'test.sdf')) if mol is not None]
    sub_geo_mmd_metric_fn = get_sub_geometry_metric(test_mols, dataset_info, root_path)
    sub_geo_mmd_res = sub_geo_mmd_metric_fn(gen_mols)

    return sub_geo_mmd_res