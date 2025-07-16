import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import logging
from pathlib import Path
from typing import List, Optional, Any
from posebusters import PoseBusters
import timeout_decorator
from tqdm.contrib.concurrent import process_map
from src.utils.ConfSeq_3_2 import replace_angle_brackets_with_line
from src.utils.similarity import (get_tanimoto_similarity_matrix,
                                  get_shape_similarity_matrix_shaep,
                                  )

@timeout_decorator.timeout(120)
def my_bust_timeout(mol):
    buster = PoseBusters(config='mol')
    df = buster.bust(mol, full_report=True).reset_index()
    if df is not None:
        return df
    

def my_bust(mol):
    try:
        return my_bust_timeout(mol)
    except:
        return None


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


def flatten_similarity_data(
    shape_list: List[np.ndarray],
    graph_list: List[np.ndarray],
    score_list: Optional[List[List[float]]] = None
) -> pd.DataFrame:
    """
    Flattens pairwise similarity arrays into a DataFrame.

    Parameters
    ----------
    shape_list : List[np.ndarray]
        List of shape similarity arrays for each group.
    graph_list : List[np.ndarray]
        List of graph similarity arrays for each group.
    score_list : Optional[List[List[float]]]
        List of scores for each group. If None, the 'score' column is not included.

    Returns
    -------
    pd.DataFrame
        Contains columns ['group_id', 'mol_id', 'shape_similarity', 'graph_similarity']
        and an optional 'score' column.
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
    save_path: Optional[str] = None,
    has_scores: bool = True
) -> pd.DataFrame:
    """
    Computes shape and Tanimoto similarity for batches of reference and generated molecules,
    and returns them as a DataFrame (optionally saved to CSV).

    Parameters
    ----------
    ref_mols : List[Mol]
        List of reference molecules.
    gen_data : List[List[Mol]]
        List of generated molecules organized in batches.
    method : str, default 'shaep'
        Similarity calculation method, options are 'shaep' or 'rdkit'.
    save_path : Optional[str]
        If provided, saves the results to a CSV file.
    has_scores : bool, default True
        Whether to extract the 'score' field from molecule properties.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing similarity and optional scores.
    """
    shape_list, graph_list = [], []
    for idx, batch in enumerate(tqdm(gen_data, desc='Calculating similarity')):
        try:
            s = get_shape_similarity_matrix_shaep([ref_mols[idx]], batch).flatten()
            g = get_tanimoto_similarity_matrix([ref_mols[idx]], batch).flatten()
        except Exception as e:
            logging.warning(f'Group  {idx}  calculation failed, using empty array instead:{e}')
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
    After grouping by group_id, calculate the intra-group mean of shape_similarity and graph_similarity,
    and then calculate the overall mean and standard deviation for these group means, 
    finally returning the result as a "mean±std" string.

    Parameters:
    - df: DataFrame, must contain the following three columns
        • group_id          : Group identifier
        • shape_similarity  : Shape similarity
        • graph_similarity  : Graph (Tanimoto) similarity

    Returns:
    - summary: DataFrame, with index ['shape', 'graph'],
        and column ['mean±std'], where the values are formatted strings.
    """
    # 1. Aggregate by group to get the mean of each group
    df_group = (
        df
        .groupby('group_id', as_index=False)
        .agg(
            shape_mean=('shape_similarity', 'mean'),
            graph_mean=('graph_similarity', 'mean')
        )
    )

    # 2. Calculate the overall mean & standard deviation of the "group means"
    shape_mean_of_means = df_group['shape_mean'].mean()
    graph_mean_of_means = df_group['graph_mean'].mean()

    # 3. Format as a "mean±std" string, keeping three decimal places
    summary = pd.DataFrame({
        'Avg_shape': [
            shape_mean_of_means
        ],
        'Avg_graph': [
            graph_mean_of_means
        ]
    })

    return summary
