import sys
from tqdm import tqdm
sys.path.append('../')  # 替换为你实际的目录路径

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, rdchem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdMolTransforms
import random
import copy
import math
from indigo import *
indigo = Indigo()
from scipy.stats import mode

from rdkit import Chem
import random
from ConfSeq_3_2 import randomize_mol,get_ConfSeq_pair_from_mol,get_mol_from_ConfSeq_pair
import copy

import pickle
from tqdm.contrib.concurrent import process_map  



def rm_invalid_chirality(mol):
    mol = copy.deepcopy(mol)
    """
    找出分子中同时出现在三个环中的原子。
    
    参数:
        mol: RDKit 分子对象
    返回:
        List[int]: 同时出现在三个环中的原子的索引列表
    """
    # 获取分子的所有环（SSSR：最小集的简单环）
    rings = rdmolops.GetSymmSSSR(mol)

    # 创建一个字典，记录每个原子出现在多少个环中
    atom_in_rings_count = {}

    # 遍历所有环，统计每个原子出现的次数
    for ring in rings:
        for atom_idx in ring:
            if atom_idx not in atom_in_rings_count:
                atom_in_rings_count[atom_idx] = 0
            atom_in_rings_count[atom_idx] += 1

    # 找出那些同时出现在三个环中的原子
    atoms_in_3_rings = [atom for atom, count in atom_in_rings_count.items() if count == 3]

    for atom_idx in atoms_in_3_rings:
        atom = mol.GetAtomWithIdx(atom_idx)
        atom.SetChiralTag(rdchem.ChiralType.CHI_UNSPECIFIED)

    return mol



def aug_mol(mol_o,mode):

    mol = Chem.RemoveHs(mol_o)
    mol = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol))

    mol = rm_invalid_chirality(mol)
    
    mol = randomize_mol(mol)
    Chem.MolToSmiles(mol)
    ran_mol = Chem.RenumberAtoms(mol, eval(mol.GetProp('_smilesAtomOutputOrder'))) 
    if mode == 0:
        atom_num = ran_mol.GetNumAtoms()
        rootedatom = 0
        Chem.MolToSmiles(ran_mol,rootedAtAtom = rootedatom)    
        ran_mol = Chem.RenumberAtoms(ran_mol, eval(ran_mol.GetProp('_smilesAtomOutputOrder'))) 
        Chem.MolToSmiles(ran_mol,canonical = False)
        ran_mol = Chem.RenumberAtoms(ran_mol, eval(ran_mol.GetProp('_smilesAtomOutputOrder'))) 
    if mode == 1:
        atom_num = ran_mol.GetNumAtoms()
        rootedatom  = random.randint(0,atom_num-1)
        Chem.MolToSmiles(ran_mol,rootedAtAtom = rootedatom)    
        ran_mol = Chem.RenumberAtoms(ran_mol, eval(ran_mol.GetProp('_smilesAtomOutputOrder'))) 
        Chem.MolToSmiles(ran_mol,canonical = False)
        ran_mol = Chem.RenumberAtoms(ran_mol, eval(ran_mol.GetProp('_smilesAtomOutputOrder'))) 
    if mode == 2:
        #atom_num = ran_mol.GetNumAtoms()
        Chem.MolToSmiles(ran_mol,doRandom=True)    
        ran_mol = Chem.RenumberAtoms(ran_mol, eval(ran_mol.GetProp('_smilesAtomOutputOrder'))) 
        Chem.MolToSmiles(ran_mol,canonical = False)  
        ran_mol = Chem.RenumberAtoms(ran_mol, eval(ran_mol.GetProp('_smilesAtomOutputOrder'))) 
        
    return ran_mol



def run_aug_mol_get_ConfSeq_pair_0(data):
    try:
        data = copy.deepcopy(data)
        mol_o = data[0]
        o_smiles = data[1]
        ran_mol = aug_mol(mol_o,0)
        in_smiles,TD_smiles = get_ConfSeq_pair_from_mol(ran_mol)
        txt = o_smiles +'\t'+in_smiles +  '\t'+TD_smiles
    except:
        print('error')
        txt = ''
    return txt

def run_aug_mol_get_ConfSeq_pair_1(data):
    try:
        data = copy.deepcopy(data)
        mol_o = data[0]
        o_smiles = data[1]
        ran_mol = aug_mol(mol_o,1)
        in_smiles,TD_smiles = get_ConfSeq_pair_from_mol(ran_mol)
        txt = o_smiles +'\t'+in_smiles +  '\t'+TD_smiles
    except:
        print('error')
        txt = ''
    return txt

def run_aug_mol_get_ConfSeq_pair_2(data):
    try:
        data = copy.deepcopy(data)
        mol_o = data[0]
        o_smiles = data[1]
        ran_mol = aug_mol(mol_o,2)
        in_smiles,TD_smiles = get_ConfSeq_pair_from_mol(ran_mol)
        txt = o_smiles +'\t'+in_smiles +  '\t'+TD_smiles
    except:
        print('error')
        txt = ''
    return txt


def get_conf(para):
    try:
        in_smiles,TD_smiles = para

        conf = get_mol_from_ConfSeq_pair(in_smiles,TD_smiles,is_op = True)
        conf = Chem.MolFromMolBlock(remove_degree_in_molblock(Chem.MolToMolBlock(conf)))
    except:
        conf = ''
        #print('error')
    return conf

def remove_degree_in_molblock(content):
    lines = content.split('\n')
    atom_num = int(lines[3][:3].strip(' '))
    bond_num = int(lines[3][3:6].strip(' '))
            
    for atom_idx in range(0,atom_num):
        lines[4+atom_idx] = lines[4+atom_idx][:48] + '  0' + lines[4+atom_idx-1][51:]
    
    content = '\n'.join(lines)
    return content


# 打开.pkl文件
with open('./raw_data/train_data_39k.pkl', 'rb') as file:
    # 加载文件中的对象
    datas = pickle.load(file)
random.seed(0)
random.shuffle(datas)

filtered_datas = []
for data in tqdm(datas):
    mol_o = list(data)[4][1] 
    if '.' not in Chem.MolToSmiles(mol_o):
        filtered_datas.append(data)

filtered_datas_r = []
for data in filtered_datas:  
    filtered_datas_r.append((copy.deepcopy(data.rdmol),copy.deepcopy(data.smiles)))  #不这样好像没法作为并行的输入

results_t0 = process_map(run_aug_mol_get_ConfSeq_pair_0, tqdm(filtered_datas_r*40), max_workers = 40)

with open('./processed_data/train_data_39k_ConfSeq_aug_0.txt','w+') as f:
    for i in results_t0:
        i = i.replace('<180>','<-180>')  #不需要180度
        f.write(i)
        f.write('\n')

results_t1 = process_map(run_aug_mol_get_ConfSeq_pair_1, tqdm(filtered_datas_r*40), max_workers = 40)

with open('./processed_data/train_data_39k_ConfSeq_aug_1.txt','w+') as f:
    for i in results_t1:
        i = i.replace('<180>','<-180>')  #不需要180度
        f.write(i)
        f.write('\n')


results_t2 = process_map(run_aug_mol_get_ConfSeq_pair_2, tqdm(filtered_datas_r*80), max_workers = 40)

with open('./processed_data/train_data_39k_ConfSeq_aug_2.txt','w+') as f:
    for i in results_t2:
        i = i.replace('<180>','<-180>')  #不需要180度
        f.write(i)
        f.write('\n')


# 打开.pkl文件
with open('./raw_data/val_data_5k.pkl', 'rb') as file:
    # 加载文件中的对象
    datas = pickle.load(file)

random.seed(0)
random.shuffle(datas)

filtered_datas = []
for data in tqdm(datas):
    mol_o = list(data)[4][1] 
    if '.' not in Chem.MolToSmiles(mol_o):
        filtered_datas.append(data)

filtered_datas_r = []
for data in filtered_datas:  
    filtered_datas_r.append((copy.deepcopy(data.rdmol),copy.deepcopy(data.smiles)))  #不这样好像没法作为并行的输入

results_t4 = process_map(run_aug_mol_get_ConfSeq_pair_0, tqdm(filtered_datas_r), max_workers = 40)

with open('./processed_data/val_data_5k_ConfSeq.txt','w+') as f:
    for i in results_t4:
        i = i.replace('<180>','<-180>')
        f.write(i)
        f.write('\n')