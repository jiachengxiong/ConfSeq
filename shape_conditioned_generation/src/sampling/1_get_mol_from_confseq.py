import sys
sys.path.append('.')
import argparse
import os

from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from rdkit.Chem import SDWriter
from rdkit.Chem.PropertyMol import PropertyMol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

from src.utils.misc import load_config, get_logger, save_pickle, load_smileslist, load_scores
from src.utils.reconstruct import convert_tdsmiles_to_mol, convert_smiles_to_mol

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/surfbart_generation.yaml')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_path)

    # Set up logger
    logger = get_logger('convert_confseq_to_mol', config.save_path)
    logger.info('---------------------------------------------')

    # Load the TD SMILES
    logger.info('Loading the TD SMILES...')
    smiles = load_smileslist(os.path.join(config['save_path'], 'generated_smiles.txt'))
    logger.info(f'Number of TD SMILES: {len(smiles)}')

    # Convert TD SMILES to molecules
    logger.info('Converting TD SMILES to molecules...')
    if config['data']['use_smiles']:
        all_results = process_map(
            convert_smiles_to_mol,
            smiles,
            max_workers=config['data']['num_workers'],
            chunksize=20
        )
    else:
        all_results = process_map(
            convert_tdsmiles_to_mol,
            smiles,
            max_workers=config['data']['num_workers'],
            chunksize=20
        )

    # split results into batches
    batch_size = config['generation_config']['num_return_sequences']
    all_socres = load_scores(os.path.join(config['save_path'], 'generated_scores.txt'))
    batched_results = [all_results[i:i+batch_size] for i in range(0, len(all_results), batch_size)]
    batched_scores = [all_socres[i:i+batch_size] for i in range(0, len(all_socres), batch_size)]
    logger.info(f'Number of batches: {len(batched_results)}')

    # Save the molecules
    generated_mols = []
    for i, (results, scores) in enumerate(zip(batched_results, batched_scores)):
        clean = []
        for result, score in zip(results, scores):
            if isinstance(result, str) and result.startswith('Error:'):
                pass
            elif result is not None:
                # Convert to PropertyMol
                result = PropertyMol(result)
                result.SetProp('score', str(score))
                clean.append(result)
        generated_mols.append(clean)

    # Save the molecules
    save_pickle(generated_mols, os.path.join(config['save_path'], 'generated_mols.pkl'))
    logger.info(f'Saved {len(generated_mols)} results')


if __name__ == '__main__':
    main()
