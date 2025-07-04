import numpy as np
import copy
import pdb
from rdkit import Chem, DataStructs
from src.utils.shaep_utils import *
from src.utils.espsim import GetEspSim, GetShapeSim
from rdkit.Chem import rdShapeAlign
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import pdb
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from .moses import get_mol
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator
import networkx as nx
#from utils.openeye_utils import ROCS_shape_overlap

def tanimoto_sim(mol, ref):
    fp1 = Chem.RDKFingerprint(ref)
    fp2 = Chem.RDKFingerprint(mol)
    return DataStructs.TanimotoSimilarity(fp1,fp2)

def tanimoto_sim_pairwise(mols):
    sims = np.ones((len(mols), len(mols)))

    for i, m1 in enumerate(mols):
        for j, m2 in enumerate(mols[i+1:]):
            sims[i, i+j+1] = tanimoto_sim(m1, m2)
            sims[i+j+1, i] = sims[i, i+j+1]
    return sims

def batched_number_of_rings(mols):
    n = []
    for m in mols:
        n.append(Chem.rdMolDescriptors.CalcNumRings(m))
    return np.array(n)

def calculate_shaep_shape_sim(mols, ref):
    aligned_mols = []
    aligned_simROCS = []
    for i, mol in enumerate(mols):
        try:
            mol, rocs = ESP_shape_align(ref, mol)
        except Exception as e:
            print(e)
            mol = None
            rocs = -1
        aligned_mols.append(mol)
        aligned_simROCS.append(rocs)
    return aligned_mols, aligned_simROCS

def calculate_espsim_shape_sim(mols, ref):
    #ref_crippen = rdMolDescriptors._CalcCrippenContribs(ref)
    aligned_simEsps = []
    aligned_simShapes = []
    for i, mol in enumerate(mols):
        #mol_crippen = rdMolDescriptors._CalcCrippenContribs(mol)
        if mol is None:
            simEsp, simShape = -1, -1
        else:
            simEsp = GetEspSim(ref, mol, prbCid = 0, refCid = 0, partialCharges = 'ml', nocheck=True)
            simShape = GetShapeSim(ref, mol, prbCid = 0, refCid = 0)
        
        aligned_simEsps.append(simEsp)
        aligned_simShapes.append(simShape)
    return aligned_simEsps, aligned_simShapes

def read_similarity(path='./tmp/shaep/similarity.txt', query_num=1000, lib_num=970):
    with open(path, 'r') as f:
        lines = f.readlines()
    # 提取矩阵
    data = lines[1].strip().split('\t')[6+query_num:]
    # 相似度数值化
    data = [float(i) if i != 'nan' else 0 for i in data ]
    # 矩阵
    data = np.array(data)
    data.resize(lib_num, query_num)  #(lib_num, query_num)
    return data

def get_shape_similarity_matrix_shaep(query_mol_list, lib_mol_list):
    # mkdir
    os.system('rm -rf ./tmp/shaep')
    os.makedirs('./tmp/shaep', exist_ok=True)

    # save mols
    writer = Chem.SDWriter('./tmp/shaep/query.sdf')
    for mol in query_mol_list:
        writer.write(mol)
    writer.close()
    writer = Chem.SDWriter('./tmp/shaep/lib.sdf')
    for mol in lib_mol_list:
        writer.write(mol)
    writer.close()

    # calculate shape similarity
    os.system('./software/shaep -q ./tmp/shaep/query.sdf  ./tmp/shaep/lib.sdf  ./tmp/shaep/similarity.txt --onlyshape')

    # get similarity matrix
    shape_matrix = read_similarity(path='./tmp/shaep/similarity.txt', query_num=len(query_mol_list), lib_num=len(lib_mol_list))

    # remove files
    os.system('rm -rf ./tmp/shaep')

    return shape_matrix #(lib_num, query_num)

def get_pairwise_shape_similarity(mol1, mol2, RemoveHs=True, useColors=False):
    mol1_, mol2_ = copy.deepcopy(mol1), copy.deepcopy(mol2)
    if RemoveHs:
        return rdShapeAlign.AlignMol(Chem.RemoveHs(mol1_), Chem.RemoveHs(mol2_), useColors=useColors)[0]
    else:
        return rdShapeAlign.AlignMol(mol1_, mol2_, useColors=useColors)[0]


def get_shape_similarity_matrix(query_mols_list, lib_mols_list, RemoveHs=True, useColors=False):
    shape_similarity_matrix = np.zeros((len(lib_mols_list), len(query_mols_list)))

    for i, query_mol in enumerate(query_mols_list):
        for j, lib_mol in enumerate(lib_mols_list):
            shape_similarity_matrix[j, i] = get_pairwise_shape_similarity(query_mol, lib_mol, RemoveHs=RemoveHs, useColors=useColors)

    return shape_similarity_matrix


def get_pairwise_tanimoto_similarity(mol1, mol2):
    fpg = GetMorganGenerator()
    return DataStructs.TanimotoSimilarity(fpg.GetFingerprint(mol1), fpg.GetFingerprint(mol2))


def get_tanimoto_similarity_matrix(query_mol_list, lib_mol_list):
    morgan_generator = GetMorganGenerator()
    query_mol_list = [get_mol(mol) for mol in query_mol_list]
    lib_mol_list = [get_mol(mol) for mol in lib_mol_list]
    query_fps = [morgan_generator.GetFingerprint(mol) for mol in query_mol_list]
    lib_fps = [morgan_generator.GetFingerprint(mol) for mol in lib_mol_list]
    tanimoto_matrix = np.zeros((len(lib_fps), len(query_fps)))
    for i, lib_fp in enumerate(lib_fps):
        similarities = DataStructs.BulkTanimotoSimilarity(lib_fp, query_fps)
        tanimoto_matrix[i, :] = similarities
    
    return tanimoto_matrix


def get_tanimoto_similarity_matrix_rdkfingerprint(query_mol_list, lib_mol_list):
    fpg = GetRDKitFPGenerator()
    for mol in query_mol_list:
        Chem.SanitizeMol(mol)
    for mol in lib_mol_list:
        Chem.SanitizeMol(mol)
    query_fps = [fpg.GetFingerprint(mol) for mol in query_mol_list]
    lib_fps = [fpg.GetFingerprint(mol) for mol in lib_mol_list]
    tanimoto_matrix = np.zeros((len(lib_fps), len(query_fps)))

    for i, lib_fp in enumerate(lib_fps):
        similarities = DataStructs.BulkTanimotoSimilarity(lib_fp, query_fps)
        tanimoto_matrix[i, :] = similarities
    
    return tanimoto_matrix


def get_max_shape_align_mol(lib_mol_list, shape_matrix):
    '''
    get max align mol

    Args:
    lib_mol_list: list of library molecules
    similarity_matrix: similarity matrix (len(lib_mol_list), len(query_mol_list))

    Returns:
    max_align_mol: list of max align molecules
    idx: index of max align molecules
    '''
    index = np.argmax(shape_matrix, axis=0)
    max_align_mol = []
    for i, num in enumerate(index):
        max_align_mol.append(lib_mol_list[num])
    return max_align_mol, index


def get_min_graph_align_mol(lib_mol_list, tanimoto_matrix):
    '''
    get min align mol

    Args:
    lib_mol_list: list of library molecules
    tanimoto_matrix: tanimoto similarity matrix (len(lib_mol_list), len(query_mol

    Returns:
    min_align_mol: list of min align molecules
    '''
    index = np.argmin(tanimoto_matrix, axis=0)
    min_align_mol = []
    for i, num in enumerate(index):
        min_align_mol.append(lib_mol_list[num])
    return min_align_mol

import numpy as np

def average_agg_tanimoto_from_matrix(S, agg='mean', p=1):
    """
    Mimic the logic of average_agg_tanimoto, given a precomputed Tanimoto matrix S.
    
    Parameters
    ----------
    S : np.ndarray
        shape = (n_stock, n_gen)，S[j, i] = Tanimoto(stock_j, gen_i).
    agg : {'mean', 'max'}
        Aggregation method for each gen molecule across all stock molecules.
    p : float
        Power for p-mean. Default=1 means arithmetic mean (or plain max).
    
    Returns
    -------
    float
        The final aggregated Tanimoto value (averaged over all gen molecules).
    """
    n_stock, n_gen = S.shape
    
    # 对所有生成分子做聚合结果
    aggregator = np.zeros(n_gen, dtype=np.float64)

    if agg == 'max':
        # 对每个 gen 列取最大值（若 p!=1，则先 S^p 再取 max，再 ^(1/p)）
        if p != 1:
            S_pow = S ** p
            # 每一列取最大值
            col_max = np.max(S_pow, axis=0)      # shape = (n_gen,)
            aggregator = col_max ** (1/p)        # (max_{j} s_{j,i}^p)^(1/p)
        else:
            aggregator = np.max(S, axis=0)       # (max_{j} s_{j,i})
    
    elif agg == 'mean':
        # 对每个 gen 列取 (mean of S^p)^(1/p)
        S_pow = S ** p if p != 1 else S
        col_mean = np.mean(S_pow, axis=0)       # shape = (n_gen,)
        if p != 1:
            aggregator = col_mean ** (1/p)
        else:
            aggregator = col_mean
    
    else:
        raise ValueError("agg must be 'max' or 'mean'")

    # 最后对所有 gen 分子再做一次平均
    return aggregator.mean()


def cal_intdiv(similarity_matrix, p=1):
    return 1 - average_agg_tanimoto_from_matrix(similarity_matrix, agg='mean', p=p).mean()


def cal_snn(similarity_matrix, p=1):
    return average_agg_tanimoto_from_matrix(similarity_matrix, agg='mean', p=p)


def cal_sumbottleneck(similarity_matrix):
    """
    Calculate the sum of the bottleneck distances for a similarity matrix.

    Parameters:
    ----------
    similarity_matrix : 2D list or np.ndarray
        The similarity matrix.

    Returns:
    ----------
    float
        The sum of bottleneck values, where bottleneck represents the minimum similarity for each point.
    """
    mat_copy = similarity_matrix.copy()

    np.fill_diagonal(mat_copy, np.inf)
    min_vals = mat_copy.min(axis=0)

    return np.sum(min_vals)


def cal_num_circles(len_mols, similarity_matrix, threshold=0.7):
    """
    Use NetworkX's greedy algorithm to compute a Maximal Independent Set (MIS) and return its size.
    This method is suitable for large-scale graphs or when graph structure complexity causes recursion depth issues.

    Parameters:
    - len_mols: Number of molecules (or molecule ID list length)
    - similarity_matrix: Tanimoto similarity matrix between molecules (2D numpy array)
    - threshold: Similarity threshold

    Returns:
    - Size of the approximate "Maximal Independent Set" (MIS)
    """
    # 1. Build the graph
    G = nx.Graph()
    G.add_nodes_from(range(len_mols))

    # 2. Add edges between nodes if similarity >= threshold
    for i in range(len_mols):
        for j in range(i + 1, len_mols):
            if 1 - similarity_matrix[i, j] <= threshold:
                G.add_edge(i, j)

    # 3. Use NetworkX's built-in function nx.maximal_independent_set(G) to get a greedy Maximal Independent Set
    #    This function returns one possible Maximal Independent Set (not necessarily the Maximum Independent Set)
    mis_nodes = nx.maximal_independent_set(G)

    # 4. Return the size of the set
    return len(mis_nodes)

