from rdkit import Chem
from rdkit.Chem import AllChem
from src.utils.ConfSeq_3_2 import (get_mol_from_ConfSeq_pair, 
                                   replace_angle_brackets_with_line, 
                                   remove_degree_in_molblock)


def convert_tdsmiles_to_mol(td_smiles):
    try:
        in_smiles = replace_angle_brackets_with_line(td_smiles)
        generated_mol = get_mol_from_ConfSeq_pair(in_smiles, td_smiles)
        generated_mol = Chem.MolFromMolBlock(remove_degree_in_molblock(Chem.MolToMolBlock(generated_mol)))
        if generated_mol is not None:
            return generated_mol
    except Exception as e:
        return f'Error: {e}'
    return None


def convert_smiles_to_mol(smiles):
    try:
        smiles = ''.join(smiles.split(' '))
        mol = Chem.MolFromSmiles(smiles)
           #添加氢原子
        mol_with_h = Chem.AddHs(mol)
        # 生成3D构象
        params = AllChem.ETKDGv3()
        params.randomSeed = 42  # 设置随机种子

        ret = AllChem.EmbedMolecule(mol_with_h, params)
        if ret == -1:
            params.enforceChirality = False
            ret = AllChem.EmbedMolecule(mol_with_h, params)
        if ret == -1:
            return None
        
        # 可选: 使用力场优化构象
        AllChem.MMFFOptimizeMolecule(mol_with_h)

        # 移除氢原子
        mol = Chem.RemoveHs(mol_with_h)
        if mol is not None:
            return mol
    except Exception as e:
            return f'Error: {e}'
    return None

