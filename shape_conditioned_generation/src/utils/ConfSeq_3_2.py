import random
from rdkit import Chem
from rdkit.Chem import rdmolops
import copy
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, rdchem
import random
import math
from indigo import *
indigo = Indigo()
from scipy.stats import mode
from SmilesPE.pretokenizer import atomwise_tokenizer
import re
from rdkit.Chem import rdmolops
import timeout_decorator


def rm_invalid_chirality(mol):
    mol = copy.deepcopy(mol)
    """
    Find atoms that appear in three rings simultaneously in a molecule.
    
    Parameters:
        mol: RDKit molecule object
    Returns:
        List[int]: List of indices of atoms that appear in three rings simultaneously
    """
    # Get all rings in the molecule (SSSR: Smallest Set of Smallest Rings)
    rings = rdmolops.GetSymmSSSR(mol)

    # Create a dictionary to record how many rings each atom appears in
    atom_in_rings_count = {}

    # Traverse all rings and count occurrences of each atom
    for ring in rings:
        for atom_idx in ring:
            if atom_idx not in atom_in_rings_count:
                atom_in_rings_count[atom_idx] = 0
            atom_in_rings_count[atom_idx] += 1

    # Find atoms that appear in exactly three rings
    atoms_in_3_rings = [atom for atom, count in atom_in_rings_count.items() if count == 3]

    for atom_idx in atoms_in_3_rings:
        atom = mol.GetAtomWithIdx(atom_idx)
        atom.SetChiralTag(rdchem.ChiralType.CHI_UNSPECIFIED)

    return mol

def randomize_mol(mol):
    """
    Generate randomized SMILES representation.
    """
    atom_indices = list(range(mol.GetNumAtoms()))
    random.shuffle(atom_indices)  # Randomly shuffle atom order
    randomized_mol = rdmolops.RenumberAtoms(mol, atom_indices)
    return randomized_mol


from rdkit import Chem

def pick_neighbor(mol,neighbors):
    new_neighbors = []
    for i in neighbors:
        if len(mol.GetAtomWithIdx(i).GetNeighbors()) == 1:
            new_neighbors.append(i + 1000000)
        else:
            new_neighbors.append(i)
    #print(new_neighbors)
    min_ = min(new_neighbors)
    if min_ >= 1000000:
        min_ = min_ -1000000
    else:
        pass
        
    return min_
    
def get_st_dihedrals(mol):  # Can be changed to provide options
    
    # List to store single bonds and their neighbors
    dihedrals = []

    # Iterate through bonds
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.SINGLE:
            # Get the indices of the atoms at each end of the bond
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
        
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            
            # Get the neighbors of each atom
            neighbors1 = [n.GetIdx() for n in mol.GetAtomWithIdx(idx1).GetNeighbors() if n.GetIdx() != idx2]
            neighbors2 = [n.GetIdx() for n in mol.GetAtomWithIdx(idx2).GetNeighbors() if n.GetIdx() != idx1]
            

            if len(set(neighbors1+neighbors2)) == len(neighbors1) + len(neighbors2):  #3-membered ring is rigid

                if len(neighbors1) > 0 and len(neighbors2) > 0:

                        i, j, k, l = pick_neighbor(mol,neighbors1),idx1, idx2,pick_neighbor(mol,neighbors2)
                        dihedral_angle = rdMolTransforms.GetDihedralDeg(mol.GetConformer(), i, j, k, l)

                        dihedrals.append((i, j, k, l, dihedral_angle))
    
    return dihedrals


# def get_bond_token_atom_pairs(smiles_BE):

#     mol_ = indigo.loadMolecule(smiles_BE)
#     mol_block = mol_.molfile()
    
#     num_atom = int(mol_block.split('\n')[3][:3].strip())
#     num_bond = int(mol_block.split('\n')[3][3:6].strip())
    
#     bond_lines = mol_block.split('\n')[4+num_atom:4+num_atom+num_bond]
    
#     atom_pairs = [(int(i[:3].strip())-1,int(i[3:6].strip())-1) for i in bond_lines]
#     atom_pairs = [tuple(sorted(i)) for i in atom_pairs]
    
#     smiles_BE = list(smiles_BE)
    
#     bond_idx_token_idx_dic = {}
#     token_idx_bond_idx_dic = {}

#     count = 0
#     for i in range(len(smiles_BE)):
#         if smiles_BE[i] in ['-','=','#',':','/','\\'] and smiles_BE[i+1] != ']': #Error in [C2-]
#             bond_idx_token_idx_dic[count] = i
#             token_idx_bond_idx_dic[i] = count
#             count += 1

#     return bond_idx_token_idx_dic,token_idx_bond_idx_dic,atom_pairs



def get_bond_token_atom_pairs(smiles_BE):

    mol_ = indigo.loadMolecule(smiles_BE)
    mol_block = mol_.molfile()

    num_atom = int(mol_block.split('\n')[3][:3].strip())
    num_bond = int(mol_block.split('\n')[3][3:6].strip())

    bond_lines = mol_block.split('\n')[4+num_atom:4+num_atom+num_bond]

    atom_pairs = [(int(i[:3].strip())-1,int(i[3:6].strip())-1) for i in bond_lines]
    atom_pairs = [tuple(sorted(i)) for i in atom_pairs]

    smiles_BE = list(smiles_BE)

    bond_idx_token_idx_dic = {}
    token_idx_bond_idx_dic = {}

    count = 0
    in_parentheses = 0
    for i in range(len(smiles_BE)):
        if smiles_BE[i] == '[':
            in_parentheses += 1
        elif smiles_BE[i] == ']':
            in_parentheses -= 1        

        if smiles_BE[i] in ['-','=','#',':','/','\\'] and in_parentheses == 0:
            bond_idx_token_idx_dic[count] = i
            token_idx_bond_idx_dic[i] = count
            count += 1

    return bond_idx_token_idx_dic,token_idx_bond_idx_dic,atom_pairs


def replace_smiles_BE(smiles_BE,token_idx_bond_idx_dic,atom_pairs,atom_pair_dihedrals_dic):
    
    t_lis = []
    
    for i in range(len(smiles_BE)):
        if i in token_idx_bond_idx_dic:
            bond_idx = token_idx_bond_idx_dic[i]
            atom_pair = atom_pairs[bond_idx]
            if atom_pair in atom_pair_dihedrals_dic:
                #t_lis.append(smiles_BE[i]) ######
                # if smiles_BE[i] in ['/','\\']:
                #     t_lis.append('<'+smiles_BE[i]+str(int(atom_pair_dihedrals_dic[atom_pair])) + '>' )
                # else:
                #     t_lis.append('<'+ str(int(atom_pair_dihedrals_dic[atom_pair])) + '>' ) 
                t_lis.append('<'+ str(int(atom_pair_dihedrals_dic[atom_pair])) + '>' )      
            else:
                t_lis.append(smiles_BE[i])
        else:
            t_lis.append(smiles_BE[i])

    return t_lis



def condense_t_smiles_lis(smiles_BE_lis,t_smiles_lis,token_idx_bond_idx_dic):
    for k in range(len(t_smiles_lis)):

        if k in list(token_idx_bond_idx_dic.keys()):
            if t_smiles_lis[k] in ['-',':']:
                t_smiles_lis[k] = 'π'
                
                if smiles_BE_lis[k] != '/' and smiles_BE_lis[k] != '\\':
                    smiles_BE_lis[k] = 'π'
                
            else:
                pass
    
    t_smiles_lis = [i for i in t_smiles_lis if i != 'π']
    smiles_BE_lis = [i for i in smiles_BE_lis if i != 'π']
    return smiles_BE_lis,t_smiles_lis


def get_TD_smiles_from_mol(mol):
    mol = copy.deepcopy(mol)

    dihedrals = get_st_dihedrals(mol)
    atom_pair_dihedrals_dic = {}
    for dihedral in dihedrals:
        atom_pair_dihedrals_dic[(dihedral[1],dihedral[2])] = dihedral[4]

    smiles_BE = Chem.MolToSmiles(mol,canonical = False,allBondsExplicit = True)
    bond_idx_token_idx_dic,token_idx_bond_idx_dic,atom_pairs = get_bond_token_atom_pairs(smiles_BE)
    dihedrals = sorted(dihedrals, key=lambda x: atom_pairs.index((x[1], x[2])))    # The part above is the same as get sorted dihedrals
    t_smiles_lis = replace_smiles_BE(smiles_BE,token_idx_bond_idx_dic,atom_pairs,atom_pair_dihedrals_dic)
    i_smiles_lis,TD_smiles_lis = condense_t_smiles_lis(list(smiles_BE),t_smiles_lis,token_idx_bond_idx_dic)
    
    return ' '.join(i_smiles_lis), ' '.join(TD_smiles_lis),dihedrals



############################################################################




def get_p_chiral_dic(ran_mol):
    
    abs_chiral_dic = dict(Chem.FindMolChiralCenters(ran_mol, includeUnassigned=True))
    
    chiral_dic = {}
    
    for atom in ran_mol.GetAtoms():
        if str(atom.GetChiralTag()) != 'CHI_UNSPECIFIED' and atom.GetIdx() in abs_chiral_dic:
            chiral_dic[atom.GetIdx()] = atom.GetChiralTag()
            

            
    p_chiral_ran_mol = copy.deepcopy(ran_mol)

    for atom in p_chiral_ran_mol.GetAtoms():
        if atom.GetSymbol() == 'N':  # Find nitrogen atom
            hyb = atom.GetHybridization()  # Get hybridization type
            if hyb == Chem.rdchem.HybridizationType.SP3:  # Check if it's sp3 hybridization
                atom.SetFormalCharge(1)  # Set charge to +1    
                
    isotope_base = 1 
    for atom in p_chiral_ran_mol.GetAtoms():
        atom.SetIsotope(isotope_base)
        isotope_base += 1
        
    p_chiral_ran_mol = Chem.MolFromMolBlock(Chem.MolToMolBlock(p_chiral_ran_mol))
    p_chiral_ran_mol = Chem.MolFromSmiles(Chem.MolToSmiles(p_chiral_ran_mol,canonical = False)) #Strange, without this some chirality calculations will have bugs#
    Chem.SanitizeMol(p_chiral_ran_mol)
    p_chiral_ran_mol = rm_invalid_chirality(p_chiral_ran_mol)
    #print(Chem.MolToSmiles(p_chiral_ran_mol,canonical = False))
    
    p_chiral_dic = {}
    for atom in p_chiral_ran_mol.GetAtoms():

        if str(atom.GetChiralTag()) != 'CHI_UNSPECIFIED':
            #if (atom.GetIdx() not in chiral_dic) or (atom.GetChiralTag() != chiral_dic[atom.GetIdx()]):
            if atom.GetIdx() not in chiral_dic:
                #print(atom.GetIdx())
                #atom.GetChiralTag()
                #chiral_dic[atom.GetIdx()] #Don't understand why adding this makes it more correct
                #atom.GetChiralTag()
                p_chiral_dic[atom.GetIdx()] = atom.GetChiralTag()

        #2025.3.1 supplement, three atoms on N form a plane
        neibors = [i.GetIsotope() for i in atom.GetNeighbors()]
        if atom.GetSymbol() == "N" and len(set(neibors)) == 3 and str(atom.GetChiralTag()) == 'CHI_UNSPECIFIED' and atom.GetChiralTag() == Chem.rdchem.HybridizationType.SP3:
            if random.random() > 0.5:
                p_chiral_dic[atom.GetIdx()] = Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
            else:
                p_chiral_dic[atom.GetIdx()] = Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW     

    return p_chiral_dic


# new
def find_all_equ_atoms(mol):
    
    mol = copy.deepcopy(mol)
    
    ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
    
    target_atoms = []
    for atom in mol.GetAtoms():
        if atom.GetDegree() == 4:
            neighbors = atom.GetNeighbors()
            neighbor_ranks = [ranks[n.GetIdx()] for n in neighbors]
            neighbor_degrees = [n.GetDegree() for n in neighbors]

            # Count frequency of each level
            rank_count = {}
            for rank in neighbor_ranks:
                if rank in rank_count:
                    rank_count[rank] = rank_count[rank] + 1
                else:
                    rank_count[rank] = 1
            
            #print(rank_count)
            # Check if there are at least three chemically equivalent neighbor atoms with degree 1
            for rank, count in rank_count.items():
                if count >= 3 and all(neighbor_degrees[i] == 1 for i, r in enumerate(neighbor_ranks) if r == rank):   #Neighbor degree 1 should not be required here
                    target_atoms.append(atom.GetIdx())  # Collect indices of qualifying atoms
            
    return target_atoms


def find_part_equ_atoms(mol,dihedral_list,p_chiral_dic):     
    mol = copy.deepcopy(mol)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    
    target_atoms = []
    for i in p_chiral_dic.keys():
        keep = 0
        for dihedral in dihedral_list:
            if i == dihedral[1]:
                neibors = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(dihedral[1]).GetNeighbors()]
                if len(neibors) == 4:  #If not equal to 4, there's a hydrogen, can't be the same as another heavy atom
                    neibors.remove(dihedral[0])
                    neibors.remove(dihedral[2])
        
                    if ranks[neibors[0]] == ranks[neibors[1]]:
                        pass
                    else:
                        keep += 1     
                else:
                    keep += 1     
    
            
            elif i == dihedral[2]:
    
                neibors = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(dihedral[2]).GetNeighbors()]
                if len(neibors) == 4:  #If not equal to 4, there's a hydrogen, can't be the same as another heavy atom
                    neibors.remove(dihedral[1])
                    neibors.remove(dihedral[3])
        
                    if ranks[neibors[0]] == ranks[neibors[1]]:
                        pass
                    else:
                        keep += 1
                else:
                    keep += 1     
    
        if keep == 0:
            target_atoms.append(i)
    
    return target_atoms



def get_TD_smiles_in_smiles_iso(TD_smiles,in_smiles):
    new_TD_token_lis = []
    new_in_token_lis = []
    
    for TD_token, in_token in zip(TD_smiles.split(' '),in_smiles.split(' ')):
        if in_token in ['/','\\']:
            if TD_token == in_token:
                TD_token = in_token + ' ' + '-'
            else: 
                TD_token = in_token + ' ' + TD_token
            in_token = in_token + ' ' + '-'
    
        new_in_token_lis.append(in_token)
        new_TD_token_lis.append(TD_token)

    return ' '.join(new_TD_token_lis),' '.join(new_in_token_lis)



def add_chiral_to_TD_smiles_in_smiles(p_chiral_dic,dihedrals,TD_smiles,in_smiles):
    
    p_chiral_dic_bond = {}

    for i,j in p_chiral_dic.items():
        for d in dihedrals:
            atom1,atom2,atom3,atom4,angle = d
            if i == atom2:
                idx = dihedrals.index(d)
                p_chiral_dic_bond[idx] =('q',j)
                break
            elif i == atom3:
                idx = dihedrals.index(d)
                p_chiral_dic_bond[idx] =('h',j)
                break 
    
                
    count = 0
    

    new_TD_smiles_lis = []
    new_in_smiles_lis = []

    for token,in_token in zip(TD_smiles.split(' '),in_smiles.split(' ')):
        if '<' in token:
            if count in p_chiral_dic_bond.keys():
                p_chiral = p_chiral_dic_bond[count]

                if p_chiral[0] == 'q':
                    token = str(p_chiral[1]) + ' ' + token
                    in_token = '!' + ' ' + in_token
                elif p_chiral[0] == 'h':
                    token = token + ' ' + str(p_chiral[1])
                    in_token = in_token + ' ' + '!'

                token = token.replace('CHI_TETRAHEDRAL_CCW','}')
                token = token.replace('CHI_TETRAHEDRAL_CW','{')



            count += 1

        new_TD_smiles_lis.append(token)
        new_in_smiles_lis.append(in_token)
        
    return  ' '.join(new_TD_smiles_lis),' '.join(new_in_smiles_lis)

##########################2025_3_2 added Angle

def is_atom_in_ring(atom, mol):
    """Determine if atom is on a ring"""
    atom_idx = atom.GetIdx()
    # Get ring information of the molecule
    ring_info = mol.GetRingInfo()
    
    # Traverse all rings and check if atom belongs to a ring
    for ring in ring_info.AtomRings():
        if atom_idx in ring:
            return True
    return False


def get_angle(mol):
    """Calculate bond angle distribution of qualifying atoms in molecule"""
    angle_dict = {}
    
    # Get 3D coordinates
    conf = mol.GetConformer()
    
    for atom in mol.GetAtoms():
        # Exclude atoms on rings
        if is_atom_in_ring(atom, mol):
            continue
        
        # Get all neighbor atoms
        neighbors = atom.GetNeighbors()
        
        # Exclude oxygen atoms connected by double bonds
        valid_neighbors = []
        for neighbor in neighbors:
            if neighbor.GetAtomicNum() == 8 and any([bond.GetBondType() == Chem.rdchem.BondType.DOUBLE for bond in neighbor.GetBonds()]):
                continue  # Exclude oxygen atoms connected by double bonds
            valid_neighbors.append(neighbor)
        
        # Only consider atoms with 2 heavy atom neighbors
        heavy_neighbors = [n for n in valid_neighbors if n.GetAtomicNum() > 1]
        if len(heavy_neighbors) != 2:
            continue
        
        
        # Get atom indices
        atom_idx = atom.GetIdx()
        heavy_neighbors_idx = [n.GetIdx() for n in heavy_neighbors]
        
        # Calculate bond angle (only need to calculate once: neighbor1, center atom, neighbor2)
        neighbor1_idx, neighbor2_idx = heavy_neighbors_idx
        angle = rdMolTransforms.GetAngleDeg(conf, neighbor1_idx, atom_idx, neighbor2_idx)
        
        angle_dict[atom_idx] = angle
    
    return angle_dict

def get_atom_token_pos_lis(t_smiles):
    lis = t_smiles.split(' ')
    atom_lis = []
    inside_brackets = False  # Track whether we're between `[` and `]`

    for idx, token in enumerate(lis):
        # Check if entering inside `[`
        if '[' == token:
            inside_brackets = True  
        
        elif token == ']':
            inside_brackets = False  # Encountering `]` means end
            atom_lis.append(idx)

                # If current token is between `[` and `]`, skip
        if inside_brackets:
            pass  

        else:
            #print(token)
            # If token is lowercase and previous character is uppercase
            if len(token) == 'l'and idx > 0 and lis[idx - 1] == 'C':
                pass  # If lowercase and previous is uppercase, skip
            elif len(token) == 'r'and idx > 0 and lis[idx - 1] == 'B':
                pass  # If lowercase and previous is uppercase, skip
            elif len(token) == 'i'and idx > 0 and lis[idx - 1] == 'S':
                pass  # If lowercase and previous is uppercase, skip
            elif len(token) == 's'and idx > 0 and lis[idx - 1] == 'A':
                pass  # If lowercase and previous is uppercase, skip
            elif len(token) == 'e'and idx > 0 and lis[idx - 1] == 'S':
                pass  # If lowercase and previous is uppercase, skip
            # If token is a letter
            elif token.isalpha():
                atom_lis.append(idx)  # If token is a letter, add to atom_lis

    return atom_lis


@timeout_decorator.timeout(10)
def get_ConfSeq_pair_from_mol(ran_mol):
    ran_mol = copy.deepcopy(ran_mol)
    #####2025_3_2 added Angle
    angle_dict = get_angle(ran_mol)
    #####
    
    in_smiles, TD_smiles,dihedrals = get_TD_smiles_from_mol(ran_mol)
    p_chiral_dic = get_p_chiral_dic(ran_mol)
    equ_atoms = find_all_equ_atoms(ran_mol)
    

    for key in equ_atoms:
        if key in p_chiral_dic:
            del p_chiral_dic[key]
    equ_atoms = find_part_equ_atoms(ran_mol,dihedrals,p_chiral_dic)
    
    for key in equ_atoms:
        if key in p_chiral_dic:
            del p_chiral_dic[key]
    
    TD_smiles,in_smiles = get_TD_smiles_in_smiles_iso(TD_smiles,in_smiles)  # Add cis-trans isomer information
    TD_smiles,in_smiles = add_chiral_to_TD_smiles_in_smiles(p_chiral_dic,dihedrals,TD_smiles,in_smiles)  



    #####2025_3_2 added Angle
    atom_token_pos_lis= get_atom_token_pos_lis(TD_smiles)
    #print(TD_smiles,atom_token_pos_lis)

    in_smiles_lis = in_smiles.split(' ')
    TD_smiles_lis = TD_smiles.split(' ')
    
    for idx,angle in angle_dict.items():
        token_pos = atom_token_pos_lis[idx]
        #print(token_pos,angle)
        #print(in_smiles_lis[token_pos])
        in_smiles_lis[token_pos] = in_smiles_lis[token_pos] + ' ' + '^' + ' ' +'|'
        TD_smiles_lis[token_pos] = TD_smiles_lis[token_pos] + ' ' + '<{}>'.format(int(angle)) + ' ' + '|'

    in_smiles, TD_smiles = ' '.join(in_smiles_lis),' '.join(TD_smiles_lis)
    #####
    
    return in_smiles, TD_smiles
##########################


##2025_3_2 added Angle
'''
def get_ConfSeq_pair_from_mol(ran_mol):
    ran_mol = copy.deepcopy(ran_mol)
    in_smiles, TD_smiles,dihedrals = get_TD_smiles_from_mol(ran_mol)
    p_chiral_dic = get_p_chiral_dic(ran_mol)
    equ_atoms = find_all_equ_atoms(ran_mol)
    

    for key in equ_atoms:
        if key in p_chiral_dic:
            del p_chiral_dic[key]
    equ_atoms = find_part_equ_atoms(ran_mol,dihedrals,p_chiral_dic)
    
    for key in equ_atoms:
        if key in p_chiral_dic:
            del p_chiral_dic[key]
    
    TD_smiles,in_smiles = get_TD_smiles_in_smiles_iso(TD_smiles,in_smiles)  # Add cis-trans isomer information
    TD_smiles,in_smiles = add_chiral_to_TD_smiles_in_smiles(p_chiral_dic,dihedrals,TD_smiles,in_smiles)  
    
    return in_smiles, TD_smiles
'''


def get_p_chiral_mol_3d(smiles,p_chiral_dic,is_op = True):
    # Create molecule from SMILES string
    mol = Chem.MolFromSmiles(smiles)  # Using ethanol as example
    
    o_charge_dic = {}
    o_H_dic = {}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'N':  # Find nitrogen atom
            hyb = atom.GetHybridization()  # Get hybridization type
            if hyb == Chem.rdchem.HybridizationType.SP3 and atom.GetFormalCharge() == 0:  # Check if it's sp3 hybridization
                o_charge_dic[atom.GetIdx()] = atom.GetFormalCharge()
                o_H_dic[atom.GetIdx()] = atom.GetNumExplicitHs()
                atom.SetFormalCharge(1)  # Set charge to +1  
                atom.SetNumExplicitHs(1)
                #Explicit hydrogen 1

        if atom.GetSymbol() == 'S':  # Find nitrogen atom
            hyb = atom.GetHybridization()  # Get hybridization type
            #print(hyb)
            if hyb == Chem.rdchem.HybridizationType.SP3 and atom.GetFormalCharge() == 1:  # Check if it's sp3 hybridization
                o_charge_dic[atom.GetIdx()] = atom.GetFormalCharge()
                o_H_dic[atom.GetIdx()] = atom.GetNumExplicitHs()
                atom.SetFormalCharge(0)  # Set charge to +1  
                atom.SetNumExplicitHs(1)
                #Explicit hydrogen 1
           
    isotope_base = 1 
    for atom in mol.GetAtoms():
        atom.SetIsotope(isotope_base)
        isotope_base += 1
        
   
    for key,value in p_chiral_dic.items():
        mol.GetAtomWithIdx(key).SetChiralTag(value)

    #Add hydrogen atoms
    mol_with_h = Chem.AddHs(mol)
    # Generate 3D conformation
    params = AllChem.ETKDG()
    #params = AllChem.ETKDGv3()
    #params.randomSeed = 0  # Set random seed
    
    # Generate 3D conformation
    AllChem.EmbedMolecule(mol_with_h, params)

    # Optional: Use force field to optimize conformation
    if is_op == True:
        AllChem.UFFOptimizeMolecule(mol_with_h)
    else:
        pass
    # Remove hydrogen atoms
    mol = Chem.RemoveHs(mol_with_h)
    #AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    for k,v in o_charge_dic.items():
        atom = mol.GetAtomWithIdx(k)
        atom.SetFormalCharge(v)  # Set new charge

    for k,v in o_H_dic.items():
        atom = mol.GetAtomWithIdx(k)
        atom.SetNumExplicitHs(v)  # Set new charge        
        
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    
    mol.GetConformer()
    
    return mol

def complete_t_smiles(t_smiles_lis,smiles_BE_lis):
    for i in range(len(smiles_BE_lis)):
        if smiles_BE_lis[i] != t_smiles_lis[i]:
            if '>' not in t_smiles_lis[i]:
                t_smiles_lis = t_smiles_lis[:i] + ['-'] + t_smiles_lis[i:]   #May need to change here, sometimes not necessarily single bond
    return t_smiles_lis


def get_atom_pair_dihedrals_dic_from_TD_SMILES(in_smiles,TD_smiles,is_float):
    TD_smiles_lis = TD_smiles.split(' ')
#     smiles_lis = [i for i in TD_smiles_lis if '<' not in i]
#     smiles = ''.join(smiles_lis)
#     print(smiles)
    mol = Chem.MolFromSmiles(in_smiles.replace(' ',''))
    smiles_BE = Chem.MolToSmiles(mol,canonical = False,allBondsExplicit = True)
    #print(smiles_BE)
    smiles_BE_lis = list(smiles_BE)
    
    bond_idx_token_idx_dic,token_idx_bond_idx_dic,atom_pairs = get_bond_token_atom_pairs(smiles_BE)
    #print(TD_smiles_lis)
    #print(smiles_BE_lis)
    
    t_smiles = complete_t_smiles(TD_smiles_lis,smiles_BE_lis)
    
    atom_pair_dihedrals_dic = {} 
    for token_idx,bond_idx in token_idx_bond_idx_dic.items():
        if '<' in t_smiles[token_idx]:
            #print(atom_pairs[bond_idx],t_smiles[token_idx])
            if is_float:
                atom_pair_dihedrals_dic[atom_pairs[bond_idx]] = float(t_smiles[token_idx][1:-1].replace('/','').replace('\\',''))
            else:
                atom_pair_dihedrals_dic[atom_pairs[bond_idx]] = t_smiles[token_idx]

    return atom_pair_dihedrals_dic


def reset_dihedrals(mol,atom_pair_dihedrals_dic):
    mol = copy.deepcopy(mol)

    dihedrals = get_st_dihedrals(mol)
    
    new_dihedrals = []
    for dihedral in dihedrals:
        angle = atom_pair_dihedrals_dic[dihedral[1],dihedral[2]]

        new_dihedral = list(dihedral[:4]) + [angle]
        new_dihedrals.append(tuple(new_dihedral))    
    return new_dihedrals



def apply_dihedrals(mol, dihedral_list):
    mol = copy.deepcopy(mol)
    conf = mol.GetConformer()
    ssr = Chem.GetSymmSSSR(mol)  # Get ring system in molecule
    unapplied_dihedrals = []
    for dihedral in dihedral_list:
        

        atom1, atom2, atom3, atom4, angle = dihedral
        if  mol.GetBondBetweenAtoms(atom1, atom2) is not None and \
            mol.GetBondBetweenAtoms(atom2, atom3) is not None and \
            mol.GetBondBetweenAtoms(atom3, atom4) is not None:

            # Check if (atom2, atom3) bond belongs to ring
            in_ring = any(atom2 in ring and atom3 in ring for ring in ssr)
            if not in_ring:
                # If not in ring, set dihedral angle
                AllChem.SetDihedralRad(conf, atom1, atom2, atom3, atom4, math.radians(angle))
            else:
                unapplied_dihedrals.append(dihedral)
                
        else:
            unapplied_dihedrals.append(dihedral)


    return mol,unapplied_dihedrals




# def get_last_ring_bonds(in_smiles):
#     smiles_BE = Chem.MolToSmiles(Chem.MolFromSmiles(in_smiles),allBondsExplicit = True,canonical = False)
#     toks = atomwise_tokenizer(smiles_BE)
#     #print(toks)
#     bond_idx_token_idx_dic,token_idx_bond_idx_dic,atom_pairs = get_bond_token_atom_pairs(in_smiles)


#     ring_bond_lis = []
#     count = 0
#     for i in range(len(toks)-1):
#         if toks[i] in ['=','-','#','/','\\',':']:
#             if (toks[i + 1].isdigit() or toks[i + 1].replace('%','').isdigit()) and toks[i] != ':' :
#                 #print(toks[i],toks[i + 1])
#                 ring_bond_lis.append(count)
#             count += 1
    
#     last_ring_bonds = []
#     for i in ring_bond_lis:
#         last_ring_bonds.append(atom_pairs[i])
        
#     return last_ring_bonds


##################2025_3_1
def get_fully_shared_ring_bonds(mol):
    """
    Identify rings where all bonds belong to at least two rings, and extract bonds on these rings.

    Parameters:
        mol (rdkit.Chem.Mol): RDKit molecule object

    Returns:
        list of tuples: A list of bonds on these rings, in the format (atom1, atom2)
    """
    ring_info = mol.GetRingInfo()
    all_rings = ring_info.BondRings()  # Get bond indices of all rings
    bond_counts = {}  # Count how many rings each bond appears in

    # Count how many rings each bond appears in
    for ring in all_rings:
        for bond_idx in ring:
            bond_counts[bond_idx] = bond_counts.get(bond_idx, 0) + 1

    
    fully_shared_bonds = set()

    # Filter rings where all bonds belong to at least two rings
    for ring in all_rings:
        if all(bond_counts[bond_idx] >= 2 for bond_idx in ring):  # Ensure all bonds are in at least two rings
            fully_shared_bonds.update(ring)

    # Convert to (atom1, atom2) format
    shared_bond_list = []
    for bond_idx in fully_shared_bonds:
        bond = mol.GetBondWithIdx(bond_idx)
        atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        shared_bond_list.append(tuple(sorted((atom1, atom2))))

    return shared_bond_list

def get_last_ring_bonds(in_smiles):
    mol = Chem.MolFromSmiles(in_smiles)
    shared_bond_list = get_fully_shared_ring_bonds(mol)
    smiles_BE = Chem.MolToSmiles(mol,allBondsExplicit = True,canonical = False)
    toks = atomwise_tokenizer(smiles_BE)
    #print(toks)
    bond_idx_token_idx_dic,token_idx_bond_idx_dic,atom_pairs = get_bond_token_atom_pairs(in_smiles)


    ring_bond_lis = []
    count = 0
    for i in range(len(toks)-1):
        if toks[i] in ['=','-','#','/','\\',':']:
            if (toks[i + 1].isdigit() or toks[i + 1].replace('%','').isdigit()) and toks[i] != ':' :
                #print(toks[i],toks[i + 1])
                ring_bond_lis.append(count)
            count += 1
    
    last_ring_bonds = []
    for i in ring_bond_lis:
        if atom_pairs[i] not in shared_bond_list:
            last_ring_bonds.append(atom_pairs[i])
        
    return last_ring_bonds
##################


####2025_3_1
# def get_last_ring_bonds(in_smiles):
#     """
#     Get all non-aromatic ring bonds from an RDKit molecule object and group them by ring.

#     Parameters:
#         mol (rdkit.Chem.Mol): RDKit molecule object

#     Returns:
#         list: A list of bonds on each non-aromatic ring (represented as tuples of bond indices)
#     """
#     mol = Chem.MolFromSmiles(in_smiles)
#     rings = mol.GetRingInfo().AtomRings()  # Get atom indices of all rings
#     non_aromatic_ring_bonds = []

#     for ring in rings:
#         # Check if it is a non-aromatic ring (if all bonds are not aromatic bonds)
#         bond_list = []
#         is_aromatic = any(mol.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)]).GetIsAromatic() for i in range(len(ring)))
        
#         if not is_aromatic:
#             for i in range(len(ring)):
#                 bond = mol.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])
#                 bond = tuple(sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
#                 bond_list.append(bond)
#             non_aromatic_ring_bonds.append(bond_list)

#     picked_ring_lis = [random.choice(i) for i in non_aromatic_ring_bonds]

#     return picked_ring_lis




def change_dihedral(mol,unapplied_dihedrals,last_ring_bonds):
    new_unapplied_dihedrals = []
    for dihedral in unapplied_dihedrals:
        atom1, atom2, atom3, atom4, angle = dihedral
        last_ring_bonds = [sorted(i) for i in last_ring_bonds]

        if sorted((atom3, atom4)) in last_ring_bonds:
            atom = mol.GetAtomWithIdx(atom3)
            neighbors = atom.GetNeighbors()
            neighbor_indices = [neighbor.GetIdx() for neighbor in neighbors]
            neighbor_indices.remove(atom2)
            neighbor_indices.remove(atom4)
            if len(neighbor_indices) >= 1:
                o_2_d = Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(),atom1, atom2, atom3, neighbor_indices[0])
                o_1_d = Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(),atom1, atom2, atom3, atom4)
                new_angle = o_2_d - o_1_d + angle
                new_unapplied_dihedrals.append((atom1, atom2, atom3, neighbor_indices[0],new_angle))
            else:
                new_unapplied_dihedrals.append((atom1, atom2, atom3, atom4, angle))

        elif sorted((atom1, atom2)) in last_ring_bonds:
            atom = mol.GetAtomWithIdx(atom2)
            neighbors = atom.GetNeighbors()
            neighbor_indices = [neighbor.GetIdx() for neighbor in neighbors]
            neighbor_indices.remove(atom1)
            neighbor_indices.remove(atom3)
            if len(neighbor_indices) >= 1:
                o_2_d = Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(),neighbor_indices[0], atom2, atom3,atom4 )
                o_1_d = Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(),atom1, atom2, atom3, atom4)
                new_angle = o_2_d - o_1_d + angle
                new_unapplied_dihedrals.append((neighbor_indices[0], atom2, atom3,atom4,new_angle))
            else:
                new_unapplied_dihedrals.append((atom1, atom2, atom3, atom4, angle))
                
        else:
            new_unapplied_dihedrals.append((atom1, atom2, atom3, atom4, angle))
            
    return new_unapplied_dihedrals


def updata_last_ring_bonds_dihedral(mol_no_ring,last_ring_bonds,dihedral_list): 
    
    Chem.MolToSmiles(mol_no_ring,canonical = False)
    order_lis = eval(mol_no_ring.GetProp('_smilesAtomOutputOrder'))
    mol_no_ring = Chem.RenumberAtoms(mol_no_ring, order_lis)
    
    
    dic_ring_b_a ={}
    for i in range(len(order_lis)):
        dic_ring_b_a[order_lis[i]] = i
        
    #print(dic_ring_b_a)


    new_last_ring_bonds = []
    for bond in last_ring_bonds:
        i,j = bond
        i,j = dic_ring_b_a[i],dic_ring_b_a[j]
        new_last_ring_bonds.append((i,j))


    new_dihedral_list = []
    for dihedral in dihedral_list:
        a,b,c,d,angle = dihedral
        a,b,c,d = dic_ring_b_a[a],dic_ring_b_a[b],dic_ring_b_a[c],dic_ring_b_a[d]
        new_dihedral_list.append((a,b,c,d,angle))    

    return mol_no_ring,new_last_ring_bonds,new_dihedral_list

def del_add_chiral_from_TD_smiles(in_smiles,TD_smiles):
    TD_smiles_ = TD_smiles.replace('{ <','{<').replace('> {','>{').replace('} <','}<').replace('> }','>}')
    
    atom_pair_chiral_dihedrals_dic  = get_atom_pair_dihedrals_dic_from_TD_SMILES(in_smiles,TD_smiles_,False)
    #print(3,atom_pair_chiral_dihedrals_dic)
    
    atom_chiral_dic = {}
    for key,value in atom_pair_chiral_dihedrals_dic.items():
        if value[0] == "{":
            atom_chiral_dic[key[0]] = Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
        elif value[0] == "}":
            atom_chiral_dic[key[0]] = Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
        elif value[-1] == "{":
            atom_chiral_dic[key[-1]] = Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
        elif value[-1] == "}":
            atom_chiral_dic[key[-1]] = Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
            
    #return TD_smiles.replace('{ <','<').replace('> {','>').replace('} <','<').replace('> }','>'),p_chiral_dic
    return TD_smiles.replace('{ <','<').replace('> {','>').replace('} <','<').replace('> }','>'),atom_chiral_dic 


def get_mol_from_ConfSeq_pair(in_smiles,TD_smiles,is_op = True):


    ######2025_3_2####Angle
    pattern = r'<-?\d+>\s*\|'
    # Find all matches
    matches = re.findall(pattern, TD_smiles)
    
    for match in matches:
        TD_smiles = TD_smiles.replace(' '+match,'')
    in_smiles =  in_smiles.replace('^ |','')
    angles = [i.replace('<','').replace('> |','') for i in matches]
    angles = [int(i) for i in angles]
    ######

    
    
    in_smiles = in_smiles.replace(' !','')
    in_smiles = in_smiles.replace('/ -','/').replace('\\ -','\\')
    
    TD_smiles = TD_smiles.replace('/ -','/').replace('\\ -','\\')
    TD_smiles = TD_smiles.replace('/ <','<').replace('\\ <','<')
    TD_smiles = TD_smiles.replace('> /','>').replace('> \\','>')
    
    TD_smiles,p_chiral_dic =  del_add_chiral_from_TD_smiles(in_smiles,TD_smiles)
    
    TD_smiles_lis = TD_smiles.split(' ')
    
    mol = get_p_chiral_mol_3d(in_smiles.replace(' ',''),p_chiral_dic,is_op = is_op)  # There was a bug here, was is_op = True, 2024_12_15
    ######2025_3_2####Angle
    angle_dict = get_angle(mol)
    updata_angle_dict = {}
    count = 0
    for i,j in angle_dict.items():
        updata_angle_dict[i] = angles[count]
        count += 1
    mol = set_angle(mol, updata_angle_dict)
    ######
    atom_pair_dihedrals_dic = get_atom_pair_dihedrals_dic_from_TD_SMILES(in_smiles,TD_smiles,True)  #new
    dihedral_list = reset_dihedrals(mol,atom_pair_dihedrals_dic)
    
    mol,unapplied_dihedrals = apply_dihedrals(mol,dihedral_list[:])

    
    last_ring_bonds = get_last_ring_bonds(in_smiles.replace(' ',''))
    
    
    mol_no_ring = copy.deepcopy(mol)
    emol = Chem.EditableMol(mol_no_ring)
    
    
    type_lis = []
    for bond in last_ring_bonds:
        type_lis.append(mol.GetBondBetweenAtoms(bond[0],bond[1]).GetBondType())
        emol.RemoveBond(*bond)
    
    mol_no_ring = emol.GetMol()
    
    unapplied_dihedrals = change_dihedral(mol,unapplied_dihedrals,last_ring_bonds)
    mol_no_ring,last_ring_bonds,unapplied_dihedrals = updata_last_ring_bonds_dihedral(mol_no_ring,last_ring_bonds,unapplied_dihedrals)
    mol_no_ring,unapplied_dihedrals = apply_dihedrals(mol_no_ring,unapplied_dihedrals[:])
    mol_no_ring,unapplied_dihedrals = apply_dihedrals(mol_no_ring,unapplied_dihedrals[:])
    emol = Chem.EditableMol(mol_no_ring)
    
    for bond,type_ in zip(last_ring_bonds,type_lis):
        emol.AddBond(*bond, order=type_)
    
    ring_mol = emol.GetMol()

    #ring_mol,unapplied_dihedrals = apply_dihedrals(ring_mol,dihedral_list[:])

    return ring_mol


def set_angle(mol, angle_dict):
    """
    Sets the bond angles of atoms in a molecule according to a given dictionary.
    :param mol: RDKit molecule object, must have 3D coordinates
    :param angle_dict: Dictionary with central atom index as key and Angle (in degrees) as value
    """
    conf = mol.GetConformer()
    
    for atom_idx, angle in angle_dict.items():
        atom = mol.GetAtomWithIdx(atom_idx)
        
        # Get all neighbor atoms
        neighbors = atom.GetNeighbors()
        
        # Exclude oxygen atoms connected by double bonds
        valid_neighbors = []
        for neighbor in neighbors:
            if neighbor.GetAtomicNum() == 8 and any(
                bond.GetBondType() == Chem.rdchem.BondType.DOUBLE for bond in neighbor.GetBonds()
            ):
                continue  # Exclude oxygen atoms connected by double bonds
            valid_neighbors.append(neighbor)
        
        # Only consider atoms with 2 heavy atom neighbors
        heavy_neighbors = [n for n in valid_neighbors if n.GetAtomicNum() > 1]
        if len(heavy_neighbors) != 2:
            continue  # Skip atoms that don't meet criteria
        
        # Get neighbor atom indices
        neighbor1_idx, neighbor2_idx = [n.GetIdx() for n in heavy_neighbors]
        
        # Set bond angle
        rdMolTransforms.SetAngleDeg(conf, neighbor1_idx, atom_idx, neighbor2_idx, angle)
    
    return mol
    atom_pair_dihedrals_dic = get_atom_pair_dihedrals_dic_from_TD_SMILES(in_smiles,TD_smiles,True)  #new
    dihedral_list = reset_dihedrals(mol,atom_pair_dihedrals_dic)
    
    mol,unapplied_dihedrals = apply_dihedrals(mol,dihedral_list[:])

    
    last_ring_bonds = get_last_ring_bonds(in_smiles.replace(' ',''))
    
    
    mol_no_ring = copy.deepcopy(mol)
    emol = Chem.EditableMol(mol_no_ring)
    
    
    type_lis = []
    for bond in last_ring_bonds:
        type_lis.append(mol.GetBondBetweenAtoms(bond[0],bond[1]).GetBondType())
        emol.RemoveBond(*bond)
    
    mol_no_ring = emol.GetMol()
    
    unapplied_dihedrals = change_dihedral(mol,unapplied_dihedrals,last_ring_bonds)
    mol_no_ring,last_ring_bonds,unapplied_dihedrals = updata_last_ring_bonds_dihedral(mol_no_ring,last_ring_bonds,unapplied_dihedrals)
    mol_no_ring,unapplied_dihedrals = apply_dihedrals(mol_no_ring,unapplied_dihedrals[:])
    mol_no_ring,unapplied_dihedrals = apply_dihedrals(mol_no_ring,unapplied_dihedrals[:])
    emol = Chem.EditableMol(mol_no_ring)
    
    for bond,type_ in zip(last_ring_bonds,type_lis):
        emol.AddBond(*bond, order=type_)
    
    ring_mol = emol.GetMol()

    #ring_mol,unapplied_dihedrals = apply_dihedrals(ring_mol,dihedral_list[:])

    return ring_mol

@timeout_decorator.timeout(10)
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
    except timeout_decorator.timeout_decorator.TimeoutError:
        print('time out error')
        print(data)
        txt = ''
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

def replace_angle_brackets_with_line(input_string):
    # Use regex to match content in angle brackets and replace with underscores
    result_string = re.sub(r'<-?\d+>\s*\|', '^ |', input_string)
    result_string = re.sub(r'<.*?>', '-', result_string)
    result_string = result_string.replace('{', '!').replace('}', '!')
    return result_string


def random_adjust_numbers(text):
    # This function is applied to each match
    def adjust_number(match):
        number = int(match.group(1))  # Get the number as an integer
        # Randomly add 1, subtract 1, or leave it unchanged
        adjusted_number = number + random.choice([-3,-2,-2,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,2,2,3])

        if adjusted_number > 180:
            adjusted_number = adjusted_number - 360
        elif adjusted_number < -180:
            adjusted_number = adjusted_number + 360
        
        return f"<{adjusted_number}>"

    # Replace each number in angle brackets with its adjusted value
    
    adjusted_text = re.sub(r"<(-?\d+)>", adjust_number, text)
    return adjusted_text


def remove_degree_in_molblock(content):
    lines = content.split('\n')
    atom_num = int(lines[3][:3].strip(' '))
    bond_num = int(lines[3][3:6].strip(' '))
            
    for atom_idx in range(0,atom_num):
        lines[4+atom_idx] = lines[4+atom_idx][:48] + '  0' + lines[4+atom_idx][51:]
    
    content = '\n'.join(lines)
    return content