from __future__ import annotations

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
import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from rdkit.Chem import MolToSmiles
from rdkit.Chem.Draw import MolToACS1996SVG
from rdkit.Chem import rdDepictor
from tqdm.auto import tqdm

import os, shutil, tempfile, logging
from pathlib import Path
from copy import deepcopy
from typing import List, Any, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
# 若已封装了 MolToACS1996SVG，请改成对应函数名
from rdkit.Chem.rdmolfiles import SDMolSupplier, SDWriter
from typing import Tuple, Dict
from Bio.PDB import PDBParser, MMCIFIO
from io import StringIO

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


import logging
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor, DataStructs

logger = logging.getLogger(__name__)


def _calc_shaep_and_get_aligned_gen_mol(
    ref_mol: Chem.Mol,
    gen_mol: Chem.Mol,
    work_dir: Path,
    shaep_bin: Optional[str] = None,
) -> Tuple[float, Optional[Chem.Mol]]:
    """
    使用 SHAEP 对 ref_mol 与 gen_mol 做两两比对。

    SHAEP 会在 overlay.sdf 中写入两个分子：
      - 第一个：参考分子 (ref)
      - 第二个：叠合后的查询分子 (aligned gen)

    Parameters
    ----------
    ref_mol, gen_mol
        带 3D 坐标的 RDKit Mol 对象。
    work_dir
        临时工作目录，用于存放中间文件。
    shaep_bin
        SHAEP 可执行文件路径。若为 None，则使用环境变量
        SHAEP_BIN 或默认的 "./software/shaep"。

    Returns
    -------
    shape_similarity : float
        形状相似度；失败时为 np.nan。
    aligned_gen_mol : Optional[Chem.Mol]
        对齐后的生成分子；失败时为 None。
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    shaep_bin = shaep_bin or os.getenv("SHAEP_BIN", "./software/shaep")

    ref_path = work_dir / "ref.mol"
    gen_path = work_dir / "gen.mol"
    sim_csv = work_dir / "similarity.csv"
    overlay_sdf = work_dir / "overlay.sdf"

    Chem.MolToMolFile(ref_mol, str(ref_path))
    Chem.MolToMolFile(gen_mol, str(gen_path))

    cmd = f'{shaep_bin} -q "{ref_path}" "{gen_path}" "{sim_csv}" --onlyshape -s "{overlay_sdf}"'
    ret = os.system(cmd)
    if ret != 0:
        logger.warning("SHAEP run failed with return code %s. Command: %s", ret, cmd)
        return float("nan"), None

    # 解析 shape 相似度
    try:
        df = pd.read_csv(sim_csv, delimiter="\t")
        # 很多 SHAEP 版本第一列是 "molecule" 列，存 shape 相似度
        shape_sim = float(df["molecule"].iloc[0])
    except Exception as e:
        logger.warning("Failed to parse SHAEP similarity from %s: %s", sim_csv, e)
        shape_sim = float("nan")

    # 从 overlay.sdf 中解析对齐后的生成分子 (索引为 1)
    try:
        if overlay_sdf.exists():
            suppl = Chem.SDMolSupplier(str(overlay_sdf), removeHs=False)
            mols = [m for m in suppl if m is not None]
            if len(mols) >= 2:
                # 注意：这里修正为取第二个分子（对齐后的查询分子）
                aligned_gen = mols[1]
                return shape_sim, aligned_gen
            else:
                logger.warning(
                    "Overlay SDF %s does not contain two molecules, got %d.",
                    overlay_sdf,
                    len(mols),
                )
    except Exception as e:
        logger.warning("Failed to read aligned molecule from %s: %s", overlay_sdf, e)

    return shape_sim, None


def _tanimoto(
    m1: Chem.Mol,
    m2: Chem.Mol,
    radius: int = 3,
    n_bits: int = 2048,
) -> float:
    """
    计算一对分子的 Morgan 指纹 Tanimoto 相似度。

    Parameters
    ----------
    m1, m2
        RDKit Mol 对象（通常只需要 2D 拓扑）。
    radius
        Morgan 指纹半径。
    n_bits
        指纹 bit 向量长度。

    Returns
    -------
    float
        Tanimoto 相似度 [0, 1]。
    """
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, radius=radius, nBits=n_bits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, radius=radius, nBits=n_bits)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def convert_mol_to_molfile(mol: Chem.Mol, output_path: Path, mol_name: str) -> bool:
    """
    将 RDKit Mol 对象写出为 V2000 Molfile。

    Parameters
    ----------
    mol
        RDKit Mol 对象，要求已经带 3D 坐标。
    output_path
        输出 .mol 文件路径。
    mol_name
        Molfile 第一行的分子名称。

    Returns
    -------
    bool
        成功返回 True，失败返回 False。
    """
    try:
        mol.SetProp("_Name", mol_name)
        mol_block = Chem.MolToMolBlock(mol)
        if not mol_block:
            logger.warning("RDKit failed to generate Mol block for '%s'.", mol_name)
            return False
        output_path.write_text(mol_block, encoding="utf-8")
        return True
    except Exception as e:
        logger.warning(
            "Failed to write molfile for '%s' to %s: %s",
            mol_name,
            output_path,
            e,
        )
        return False


def _draw_mol_to_svg(
    mol: Chem.Mol,
    svg_path: Path,
    mol_size: Tuple[int, int] = (300, 300),
) -> bool:
    """
    绘制 2D SVG（用于参考分子和生成分子预览）。
    """
    try:
        mol_2d = deepcopy(mol)
        rdDepictor.Compute2DCoords(mol_2d)
        w, h = mol_size
        drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
        drawer.DrawMolecule(mol_2d)
        drawer.FinishDrawing()
        svg_path.write_text(drawer.GetDrawingText(), encoding="utf-8")
        return True
    except Exception as e:
        logger.warning("Failed to draw SVG to %s: %s", svg_path, e)
        return False


def compute_similarity_and_artifacts(
    ref_mols: List[Chem.Mol],
    gen_data: List[List[Chem.Mol]],
    img_dir: Path,
    mol_dir: Path,
    shaep_bin: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[Any, Path]]]:
    """
    针对一批参考分子和对应的生成分子，计算 shape / graph 相似度，并输出 2D / 3D 工件。

    Parameters
    ----------
    ref_mols
        参考分子列表，长度为 G。
    gen_data
        生成分子按参考分子分组的列表：gen_data[g] 是第 g 个参考分子的生成分子列表。
    img_dir
        用于存放 SVG 的目录。
    mol_dir
        用于存放 3D molfile 的目录。
    shaep_bin
        SHAEP 可执行文件路径（可选）。不指定时由 _calc_shaep_and_get_aligned_gen_mol
        使用默认逻辑（环境变量 SHAEP_BIN 或 ./software/shaep）。

    Returns
    -------
    df : pd.DataFrame
        每一行对应一对 (ref, gen) 的相似度与 SMILES 信息。
    file_paths : Dict[str, Dict[Any, Path]]
        已写出的文件路径索引，结构示例：
        {
            "gen_svg": { "gid_mid": Path, ... },
            "ref_svg": { gid: Path, ... },
            "ref_molfile": { gid: Path, ... },
            "aligned_gen_molfile": { "gid_mid": Path, ... },
        }
    """
    img_dir = Path(img_dir)
    mol_dir = Path(mol_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    mol_dir.mkdir(parents=True, exist_ok=True)

    ref_smiles_list = [Chem.MolToSmiles(m) for m in ref_mols]
    records: List[Dict[str, Any]] = []
    file_paths: Dict[str, Dict[Any, Path]] = {
        "gen_svg": {},
        "ref_svg": {},
        "ref_molfile": {},
        "aligned_gen_molfile": {},
    }

    for gid, (ref_mol, batch, ref_smiles) in enumerate(
        zip(ref_mols, gen_data, ref_smiles_list)
    ):
        # 1) 参考分子 SVG
        ref_svg_path = img_dir / f"ref_{gid}.svg"
        if _draw_mol_to_svg(ref_mol, ref_svg_path):
            file_paths["ref_svg"][gid] = ref_svg_path

        # 2) 参考分子 3D Molfile
        ref_molfile_path = mol_dir / f"ref_{gid}.mol"
        if convert_mol_to_molfile(ref_mol, ref_molfile_path, f"Reference_{gid}"):
            file_paths["ref_molfile"][gid] = ref_molfile_path

        # 遍历该参考分子的所有生成分子
        for mid, gen_mol in enumerate(batch):
            if gen_mol is None:
                continue

            gen_smiles = Chem.MolToSmiles(gen_mol)
            unique_key = f"{gid}_{mid}"

            # 3) 生成分子 SVG
            gen_svg_path = img_dir / f"{unique_key}_gen.svg"
            if _draw_mol_to_svg(gen_mol, gen_svg_path):
                file_paths["gen_svg"][unique_key] = gen_svg_path

            # 4) 计算相似度 + 对齐后 3D Molfile
            shape_sim, graph_sim = float("nan"), float("nan")

            with tempfile.TemporaryDirectory(prefix=f"shaep_{unique_key}_") as tmpd:
                tmp_dir = Path(tmpd)
                # SHAEP shape
                try:
                    shape_sim, aligned_gen_mol = _calc_shaep_and_get_aligned_gen_mol(
                        ref_mol=ref_mol,
                        gen_mol=gen_mol,
                        work_dir=tmp_dir,
                        shaep_bin=shaep_bin,
                    )
                except Exception as e:
                    logger.warning(
                        "SHAEP-based shape similarity failed for pair (%s): %s",
                        unique_key,
                        e,
                    )
                    shape_sim = float("nan")
                    aligned_gen_mol = None

                # Tanimoto graph
                try:
                    graph_sim = _tanimoto(ref_mol, gen_mol)
                except Exception as e:
                    logger.warning(
                        "Tanimoto similarity failed for pair (%s): %s",
                        unique_key,
                        e,
                    )
                    graph_sim = float("nan")

                # 写出对齐后的生成分子
                if aligned_gen_mol is not None:
                    aligned_path = mol_dir / f"gen_{unique_key}_aligned.mol"
                    if convert_mol_to_molfile(
                        aligned_gen_mol, aligned_path, f"Aligned_Gen_{unique_key}"
                    ):
                        file_paths["aligned_gen_molfile"][unique_key] = aligned_path

            # 5) 记录到 DataFrame
            score_val = (
                float(np.exp(float(gen_mol.GetProp("score"))))
                if gen_mol.HasProp("score")
                else float("nan")
            )

            records.append(
                {
                    "ref_id": gid,
                    "gen_mol_id": mid,
                    "ref_smiles": ref_smiles,
                    "gen_smiles": gen_smiles,
                    "shape_similarity": shape_sim,
                    "graph_similarity": graph_sim,
                    "score": score_val,
                }
            )

    df = pd.DataFrame.from_records(records)
    cols_order = [
        "ref_id",
        "gen_mol_id",
        "ref_smiles",
        "gen_smiles",
        "shape_similarity",
        "graph_similarity",
        "score",
    ]
    df = df[[c for c in cols_order if c in df.columns]]

    return df, file_paths
