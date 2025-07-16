from rdkit.Chem.rdMolTransforms import GetBondLength, GetAngleDeg, GetDihedralDeg
from functools import partial
from tqdm.contrib.concurrent import process_map


# Bond distance
def get_bond_symbol(bond_n):
    """
    Return the symbol representation of a bond
    """
    a0 = bond_n.GetBeginAtom().GetSymbol()
    a1 = bond_n.GetEndAtom().GetSymbol()
    b = str(int(bond_n.GetBondType()))  # single:1, double:2, triple:3, aromatic: 12
    return ''.join([a0, b, a1]), ''.join([a1, b, a0])


def cal_bond_distance_single(mol, top_bond_syms):
    """
    Calculate bond length information for each specified bond type for a single molecule.
    """
    distance_dict = {bond_sym: [] for bond_sym in top_bond_syms}
    conf = mol.GetConformer()
    for bond in mol.GetBonds():
        atom_id0, atom_id1 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt, reverse_bt = get_bond_symbol(bond)
        if bt in top_bond_syms:
            distance_dict[bt].append(GetBondLength(conf, atom_id0, atom_id1))
        elif reverse_bt in top_bond_syms:
            distance_dict[reverse_bt].append(GetBondLength(conf, atom_id1, atom_id0))
    return distance_dict


def cal_bond_distance(mol_list, top_bond_syms, max_workers=20, chunksize=100):
    """
    Use process_map to calculate bond length statistics for specified bond types in molecules in parallel.

    Parameters:
        mol_list (list): List of molecule objects.
        top_bond_syms (iterable): Set or list of bond symbols for which to calculate bond lengths.
        max_workers (int): Maximum number of processes, defaults to None, automatically assigned by system.
        chunksize (int): Number of tasks assigned to each process at once, recommended to adjust based on task granularity.

    Returns:
        dict: A dictionary where keys are bond symbols and values are corresponding bond length lists.
    """
    # Bind top_bond_syms parameter via partial
    worker_func = partial(cal_bond_distance_single, top_bond_syms=top_bond_syms)
    # Use process_map to process molecule list in parallel
    results = process_map(worker_func, mol_list, max_workers=max_workers, chunksize=chunksize)
    
    # Merge results returned from all processes
    bond_distance_dict = {bond_sym: [] for bond_sym in top_bond_syms}
    for result in results:
        for bond_sym in top_bond_syms:
            bond_distance_dict[bond_sym].extend(result[bond_sym])
    
    return bond_distance_dict


# Bond Angle
def get_bond_pairs(mol):
    """Get all the bond pairs in a molecule"""
    valid_bond_pairs = []
    for idx_bond, bond in enumerate(mol.GetBonds()):
        idx_end_atom = bond.GetEndAtomIdx()
        end_atom = mol.GetAtomWithIdx(idx_end_atom)
        end_bonds = end_atom.GetBonds()
        for end_bond in end_bonds:
            if end_bond.GetIdx() == idx_bond:
                continue
            else:
                valid_bond_pairs.append([bond, end_bond])
                # bond_idx.append((bond.GetIdx(), end_bond.GetIdx()))
    return valid_bond_pairs


def get_bond_pair_symbol(bond_pairs):
    """Return the symbol representation of a bond angle"""
    atom0_0 = bond_pairs[0].GetBeginAtomIdx()
    atom0_1 = bond_pairs[0].GetEndAtomIdx()
    atom0_0_sym = bond_pairs[0].GetBeginAtom().GetSymbol()
    atom0_1_sym = bond_pairs[0].GetEndAtom().GetSymbol()
    bond_left = str(int(bond_pairs[0].GetBondType()))

    atom1_0 = bond_pairs[1].GetBeginAtomIdx()
    atom1_1 = bond_pairs[1].GetEndAtomIdx()
    atom1_0_sym = bond_pairs[1].GetBeginAtom().GetSymbol()
    atom1_1_sym = bond_pairs[1].GetEndAtom().GetSymbol()
    bond_right = str(int(bond_pairs[1].GetBondType()))

    if atom0_0 == atom1_0:
        sym = ''.join([atom0_1_sym, bond_left, atom0_0_sym]) + '-' + ''.join([atom1_0_sym, bond_right, atom1_1_sym])
        ijk = (atom0_1, atom0_0, atom1_1)
    elif atom0_0 == atom1_1:
        sym = ''.join([atom0_1_sym, bond_left, atom0_0_sym]) + '-' + ''.join([atom1_1_sym, bond_right, atom1_0_sym])
        ijk = (atom0_1, atom0_0, atom1_0)
    elif atom0_1 == atom1_0:
        sym = ''.join([atom0_0_sym, bond_left, atom0_1_sym]) + '-' + ''.join([atom1_0_sym, bond_right, atom1_1_sym])
        ijk = (atom0_0, atom0_1, atom1_1)
    elif atom0_1 == atom1_1:
        sym = ''.join([atom0_0_sym, bond_left, atom0_1_sym]) + '-' + ''.join([atom1_1_sym, bond_right, atom1_0_sym])
        ijk = (atom0_0, atom0_1, atom1_0)
    else:
        raise ValueError("Bond pair error.")

    return sym, ijk


def worker_angle(mol, top_angle_syms):
    """
    Calculate bond angle information for each specified angle for a single molecule.

    Parameters:
        mol: Single molecule object.
        top_angle_syms (iterable): Set or list of bond angle symbols for which to calculate angles.

    Returns:
        dict: Dictionary where keys are bond angle symbols and values are corresponding bond angle lists for this molecule.
    """
    # Initialize dictionary with each angle symbol corresponding to an empty list
    angle_dict = {angle_sym: [] for angle_sym in top_angle_syms}
    conf = mol.GetConformer()
    bond_pairs = get_bond_pairs(mol)
    for bond_pair in bond_pairs:
        # Get forward bond angle symbol and corresponding atom index triplet
        angle_sym, ijk = get_bond_pair_symbol(bond_pair)
        i, j, k = ijk
        # Get reverse bond angle symbol (by reversing bond_pair)
        reverse_angle_sym, _ = get_bond_pair_symbol(bond_pair[::-1])
        if angle_sym in top_angle_syms:
            angle_dict[angle_sym].append(GetAngleDeg(conf, i, j, k))
        elif reverse_angle_sym in top_angle_syms:
            angle_dict[reverse_angle_sym].append(GetAngleDeg(conf, k, j, i))
    return angle_dict

def cal_bond_angle(mol_list, top_angle_syms, max_workers=20, chunksize=100):
    """
    Use process_map to calculate bond angle statistics for specified bond angles in molecules in parallel.

    Parameters:
        mol_list (list): List of molecule objects.
        top_angle_syms (iterable): Set or list of bond angle symbols for which to calculate angles.
        max_workers (int): Maximum number of processes, defaults to 10
        chunksize (int): Number of tasks assigned to each worker process at once, adjust appropriately based on task granularity.

    Returns:
        dict: A dictionary where keys are bond angle symbols and values are corresponding bond angle lists (in degrees).
    """
    # Use partial to bind top_angle_syms parameter to worker_angle function,
    # so that process_map only passes individual molecule objects
    worker_func = partial(worker_angle, top_angle_syms=top_angle_syms)
    
    # Use process_map to process molecule list in parallel
    results = process_map(worker_func, mol_list, max_workers=max_workers, chunksize=chunksize)
    
    # Initialize final statistics result dictionary
    bond_angle_dict = {angle_sym: [] for angle_sym in top_angle_syms}
    # Merge results returned from each process
    for result in results:
        for angle_sym in top_angle_syms:
            bond_angle_dict[angle_sym].extend(result[angle_sym])
    
    return bond_angle_dict


# Dihedral Angle
def get_triple_bonds(mol):
    """Get all the bond triples in a molecule"""
    valid_triple_bonds = []
    for idx_bond, bond in enumerate(mol.GetBonds()):
        idx_begin_atom = bond.GetBeginAtomIdx()
        idx_end_atom = bond.GetEndAtomIdx()
        begin_atom = mol.GetAtomWithIdx(idx_begin_atom)
        end_atom = mol.GetAtomWithIdx(idx_end_atom)
        begin_bonds = begin_atom.GetBonds()
        valid_left_bonds = []
        for begin_bond in begin_bonds:
            if begin_bond.GetIdx() == idx_bond:
                continue
            else:
                valid_left_bonds.append(begin_bond)
        if len(valid_left_bonds) == 0:
            continue

        end_bonds = end_atom.GetBonds()
        for end_bond in end_bonds:
            if end_bond.GetIdx() == idx_bond:
                continue
            else:
                for left_bond in valid_left_bonds:
                    valid_triple_bonds.append([left_bond, bond, end_bond])

    return valid_triple_bonds


def get_triple_bond_symbol(triple_bonds):
    """Return the symbol representation of a dihedral angle"""
    atom0_0 = triple_bonds[0].GetBeginAtomIdx()
    atom0_1 = triple_bonds[0].GetEndAtomIdx()
    atom0_0_sym = triple_bonds[0].GetBeginAtom().GetSymbol()
    atom0_1_sym = triple_bonds[0].GetEndAtom().GetSymbol()
    bond_left = str(int(triple_bonds[0].GetBondType()))

    atom1_0 = triple_bonds[1].GetBeginAtomIdx()
    atom1_1 = triple_bonds[1].GetEndAtomIdx()
    atom1_0_sym = triple_bonds[1].GetBeginAtom().GetSymbol()
    atom1_1_sym = triple_bonds[1].GetEndAtom().GetSymbol()
    bond_mid = str(int(triple_bonds[1].GetBondType()))

    atom2_0 = triple_bonds[2].GetBeginAtomIdx()
    atom2_1 = triple_bonds[2].GetEndAtomIdx()
    atom2_0_sym = triple_bonds[2].GetBeginAtom().GetSymbol()
    atom2_1_sym = triple_bonds[2].GetEndAtom().GetSymbol()
    bond_right = str(int(triple_bonds[2].GetBondType()))

    ijkl = []
    if atom0_0 == atom1_0:
        sym = ''.join([atom0_1_sym, bond_left, atom0_0_sym]) + '-' + ''.join([atom1_0_sym, bond_mid, atom1_1_sym])
        last_id = atom1_1
        ijkl += [atom0_1, atom0_0, atom1_1]
    elif atom0_0 == atom1_1:
        sym = ''.join([atom0_1_sym, bond_left, atom0_0_sym]) + '-' + ''.join([atom1_1_sym, bond_mid, atom1_0_sym])
        last_id = atom1_0
        ijkl += [atom0_1, atom0_0, atom1_0]
    elif atom0_1 == atom1_0:
        sym = ''.join([atom0_0_sym, bond_left, atom0_1_sym]) + '-' + ''.join([atom1_0_sym, bond_mid, atom1_1_sym])
        last_id = atom1_1
        ijkl += [atom0_0, atom0_1, atom1_1]
    elif atom0_1 == atom1_1:
        sym = ''.join([atom0_0_sym, bond_left, atom0_1_sym]) + '-' + ''.join([atom1_1_sym, bond_mid, atom1_0_sym])
        last_id = atom1_0
        ijkl += [atom0_0, atom0_1, atom1_0]
    else:
        raise ValueError("Left and middle bonds error.")

    if atom2_0 == last_id:
        sym = sym + '-' + ''.join([atom2_0_sym, bond_right, atom2_1_sym])
        ijkl.append(atom2_1)
    elif atom2_1 == last_id:
        sym = sym + '-' + ''.join([atom2_1_sym, bond_right, atom2_0_sym])
        ijkl.append(atom2_0)
    else:
        raise ValueError("Right bond error.")

    return sym, ijkl


def worker_dihedral(mol, top_dihedral_syms):
    """
    Calculates specified dihedral angles for a single molecule,
    excluding dihedrals where the central bond is in a three-membered ring.

    Args:
        mol: A single RDKit molecule object with a conformer.
        top_dihedral_syms (iterable): A set or list of dihedral symbols to be calculated.

    Returns:
        dict: A dictionary where keys are dihedral symbols and values are lists
              of corresponding dihedral angles found in the molecule.
    """
    dihedral_dict = {dihedral_sym: [] for dihedral_sym in top_dihedral_syms}
    
    if mol is None or mol.GetNumConformers() == 0:
        return dihedral_dict

    conf = mol.GetConformer()
    
    triple_bonds_list = get_triple_bonds(mol)
    
    for triple_bond in triple_bonds_list:
        dihedral_sym, ijkl = get_triple_bond_symbol(triple_bond)
        
        # If the bonds don't form a valid dihedral chain, skip.
        if ijkl is None:
            continue
            
        i, j, k, l = ijkl

        # --- START OF MODIFICATION (using get_st_dihedrals logic) ---
        # Get the neighbors of the central atoms j and k
        atom_j = mol.GetAtomWithIdx(j)
        atom_k = mol.GetAtomWithIdx(k)

        # Get neighbors of j, excluding k. Using sets is more efficient.
        neighbors_j = {neighbor.GetIdx() for neighbor in atom_j.GetNeighbors() if neighbor.GetIdx() != k}
        
        # Get neighbors of k, excluding j.
        neighbors_k = {neighbor.GetIdx() for neighbor in atom_k.GetNeighbors() if neighbor.GetIdx() != j}

        # Check if the neighbor sets have any common atoms.
        # If they do, the bond j-k is in a 3-membered ring.
        # set.isdisjoint(other_set) is a fast way to check for intersection.
        if not neighbors_j.isdisjoint(neighbors_k):
            continue # Skip this dihedral because its central bond is in a 3-ring.
        # --- END OF MODIFICATION ---

        # Get the reverse symbol as well
        reverse_dihedral_sym, _ = get_triple_bond_symbol(triple_bond[::-1])

        if dihedral_sym in top_dihedral_syms:
            dihedral_dict[dihedral_sym].append(GetDihedralDeg(conf, i, j, k, l))
        elif reverse_dihedral_sym in top_dihedral_syms:
            dihedral_dict[reverse_dihedral_sym].append(GetDihedralDeg(conf, i, j, k, l))
            
    return dihedral_dict


def cal_dihedral_angle(mol_list, top_dihedral_syms, max_workers=20, chunksize=100):
    """
    Use process_map to calculate dihedral angle statistics for specified dihedral angles in molecules in parallel.

    Parameters:
        mol_list (list): List of molecule objects.
        top_dihedral_syms (iterable): Set or list of dihedral angle symbols for which to calculate angles.
        max_workers (int): Maximum number of processes, defaults to None, automatically assigned by system.
        chunksize (int): Number of tasks assigned to each worker process at once, adjust appropriately based on task granularity.

    Returns:
        dict: A dictionary where keys are dihedral angle symbols and values are corresponding dihedral angle lists (in degrees).
    """
    # Bind top_dihedral_syms parameter to worker_dihedral via partial,
    # so that process_map only passes individual molecule objects
    worker_func = partial(worker_dihedral, top_dihedral_syms=top_dihedral_syms)
    
    # Use process_map to calculate dihedral angle information for each molecule in mol_list in parallel
    results = process_map(worker_func, mol_list, max_workers=max_workers, chunksize=chunksize)
    
    # Merge results returned from each process
    dihedral_angle_dict = {dihedral_sym: [] for dihedral_sym in top_dihedral_syms}
    for result in results:
        for dihedral_sym in top_dihedral_syms:
            dihedral_angle_dict[dihedral_sym].extend(result[dihedral_sym])
    
    return dihedral_angle_dict

