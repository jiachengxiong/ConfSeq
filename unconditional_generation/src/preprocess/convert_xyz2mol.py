import numpy as np
import os
from rdkit import Chem
from tqdm.auto import tqdm
from pathlib import Path
import argparse

def load_split_data(conformation_file, val_proportion=0.1, test_proportion=0.1,
                    filter_size=None):
    path = Path(conformation_file)
    base_path = path.parent.absolute()

    # base_path = os.path.dirname(conformation_file)
    all_data = np.load(conformation_file)  # 2d array: num_atoms x 5

    mol_id = all_data[:, 0].astype(int)
    conformers = all_data[:, 1:]
    # Get ids corresponding to new molecules
    split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
    data_list = np.split(conformers, split_indices)

    # Filter based on molecule size.
    if filter_size is not None:
        # Keep only molecules <= filter_size
        data_list = [molecule for molecule in data_list
                     if molecule.shape[0] <= filter_size]

        assert len(data_list) > 0, 'No molecules left after filter.'

    # CAREFUL! Only for first time run:
    # perm = np.random.permutation(len(data_list)).astype('int32')
    # print('Warning, currently taking a random permutation for '
    #       'train/val/test partitions, this needs to be fixed for'
    #       'reproducibility.')
    # assert not os.path.exists(os.path.join(base_path, 'geom_permutation.npy'))
    # np.save(os.path.join(base_path, 'geom_permutation.npy'), perm)
    # del perm

    perm = np.load(os.path.join(base_path, 'geom_permutation.npy'))
    data_list = [data_list[i] for i in perm]

    num_mol = len(data_list)
    val_index = int(num_mol * val_proportion)
    test_index = val_index + int(num_mol * test_proportion)
    # val_data, test_data, train_data = np.split(data_list, [val_index, test_index])
    split_indices = [val_index, test_index]
    split_data = [data_list[i:j] for i, j in zip([0]+split_indices, split_indices+[None])]
    val_data, test_data, train_data = split_data

    return train_data, val_data, test_data


def write_data_to_xyz(data, file_path):
    periodic_table = Chem.GetPeriodicTable()
    with open(file_path, 'w') as f:
        for i, molecule in enumerate(tqdm(data)):
            f.write(f'{len(molecule)}\n{i}\n')
            for line in molecule:
                element = periodic_table.GetElementSymbol(int(line[0]))
                f.write(f"{element}\t{line[1]}\t{line[2]}\t{line[3]}\n")


def convert_and_cleanup(data, xyz_path, sdf_path):
    write_data_to_xyz(data, xyz_path)
    os.system(f'obabel {xyz_path} -O {sdf_path}')
    os.remove(xyz_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--geom_file', type=str, default='./data/geom_raw/geom_drugs_30.npy', help='Path to the splited geom file')
    args = parser.parse_args()

    train_data, val_data, test_data = load_split_data(args.geom_file)

    convert_and_cleanup(train_data, './tmp/train_original.xyz', './data/geom_sdf/train.sdf')
    print('Successfully converted train data to sdf.')
    convert_and_cleanup(val_data, './tmp/val_original.xyz', './data/geom_sdf/valid.sdf')
    print('Successfully converted valid data to sdf.')
    convert_and_cleanup(test_data, './tmp/test_original.xyz', './data/geom_sdf/test.sdf')
    print('Successfully converted test data to sdf.')

if __name__ == '__main__':
    main()