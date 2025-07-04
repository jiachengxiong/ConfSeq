import sys
sys.path.append('.')
import argparse
import os
from tqdm import tqdm

from rdkit.Chem import SDMolSupplier

from src.utils.data import mol2TDsmiles
from src.utils.misc import save_pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the sdf GEOM-DRUGS dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Output folder path to save the TD SMILES')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--aug_mode', type=int, default=0)
    parser.add_argument('--aug_times', type=int, default=1)
    parser.add_argument('--do_random', action='store_true', help='Whether to do random augmentation')
    args = parser.parse_args()

    # first load the data
    suppl = list(SDMolSupplier(args.input_dir, removeHs=False))

    # convert to TD SMILES
    block_size = 100000
    TD_smiles = []

    total_blocks = len(suppl) // block_size + (1 if len(suppl) % block_size != 0 else 0)
    for block_num, mol_list in tqdm(enumerate([suppl[i:i+block_size] for i in range(0, len(suppl), block_size)]), 
                                    total=total_blocks, 
                                    desc="Processing blocks"):
        print(f'Processing block {block_num + 1}/{total_blocks}')
        results = mol2TDsmiles(mol_list, 
                            num_workers=args.num_workers, 
                            aug_mode=args.aug_mode, 
                            aug_times=args.aug_times,
                            do_random=args.do_random)
        
        for result in results:
            if result != 'error':
                TD_smiles.append(result)

    # save the TD SMILES
    split = args.input_dir.split('/')[-1].split('.')[0]
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = f'{args.output_dir}/geomdrugs_{split}_augmode{args.aug_mode}_augtimes{args.aug_times}_angles.pkl'
    save_pickle(TD_smiles, output_dir)


if __name__ == '__main__':
    main()

