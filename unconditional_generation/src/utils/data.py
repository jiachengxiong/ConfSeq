import random
from rdkit import Chem
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from src.utils.ConfSeq_3_2 import run_aug_mol_get_ConfSeq_pair_0, run_aug_mol_get_ConfSeq_pair_1, run_aug_mol_get_ConfSeq_pair_2, random_adjust_numbers


def mol2TDsmiles(mols_list, num_workers=10, aug_mode=0, aug_times=1, do_random=False, disable_tqdm=False):
    datas = []
    for mol in mols_list:
        if mol is not None:
            datas.append((mol, Chem.MolToSmiles(mol)))

    # Choose different processing functions based on aug_mode, and control whether to display progress bar through tqdm's disable parameter
    if aug_mode == 0:
        results_t0 = process_map(
            run_aug_mol_get_ConfSeq_pair_0,
            tqdm(datas * aug_times, disable=disable_tqdm),
            max_workers=num_workers,
            chunksize=1000
        )
    elif aug_mode == 1:
        results_t0 = process_map(
            run_aug_mol_get_ConfSeq_pair_1,
            tqdm(datas * aug_times, disable=disable_tqdm),
            max_workers=num_workers,
            chunksize=1000
        )
    elif aug_mode == 2:
        results_t0 = process_map(
            run_aug_mol_get_ConfSeq_pair_2,
            tqdm(datas * aug_times, disable=disable_tqdm),
            max_workers=num_workers,
            chunksize=1000
        )
    else:
        raise ValueError(f'Invalid aug_mode: {aug_mode}')

    random.seed(42)  # Set random seed to ensure reproducible results
    if do_random:
        for i in range(len(results_t0)):
            if random.random() >= 0.5:
                results_t0[i] = random_adjust_numbers(results_t0[i])
            results_t0[i] = results_t0[i].replace('<180>', '<-180>')

    # Process results
    td_smiles_list = []
    for i in range(len(results_t0)):
        parts = results_t0[i].split('\t')
        if len(parts) == 3:
            td_smiles_list.append(parts[2])
        else:
            td_smiles_list.append('error')

    return td_smiles_list
