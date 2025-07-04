import numpy as np
import os
from rdkit import Chem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


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

