from rdkit import Chem
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

