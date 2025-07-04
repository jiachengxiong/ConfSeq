import sys
sys.path.append('.')
import argparse
import os

from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from rdkit.Chem import SDWriter
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

from src.utils.misc import load_config, load_smileslist, get_logger
from src.utils.reconstruct import convert_tdsmiles_to_mol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/unconditional_generation.yaml')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_path)

    # Set up logger
    logger = get_logger('convert_confseq_to_mol', config.save_path)
    logger.info('---------------------------------------------')

    for seed_i in range(getattr(config, 'sample_times', 1)):
        logger.info('*****************************************')
        run_save_path = os.path.join(config.save_path, f'run_{seed_i}')
        logger.info('Generating molecules from SMILES...\n')

        generated_smiles = load_smileslist(os.path.join(run_save_path, 'generated_smiles.txt'))
        writer = SDWriter(os.path.join(run_save_path, 'generated_mols.sdf'))

        # Process SMILES in parallel
        results = process_map(convert_tdsmiles_to_mol, 
                              generated_smiles, 
                              chunksize=config.chunksize, 
                              max_workers=config.num_workers)
        
        success = 0
        for result in results:
            if isinstance(result, str) and result.startswith('Error:'):
                pass
            elif result is not None:
                writer.write(result)
                success += 1

        logger.info(f'Generated {success} molecules from {len(generated_smiles)} SMILES\n')

if __name__ == '__main__':
    main()
