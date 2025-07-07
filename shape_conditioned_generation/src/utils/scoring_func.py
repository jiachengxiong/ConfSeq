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
            s = get_shape_similarity_matrix_shaep([ref_mols[idx]], batch).flatten()
            g = get_tanimoto_similarity_matrix([ref_mols[idx]], batch).flatten()
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
