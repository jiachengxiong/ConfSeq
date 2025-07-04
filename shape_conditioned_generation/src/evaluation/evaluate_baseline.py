import sys
sys.path.append('.')
import os
import argparse
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

from src.utils.scoring_func import (
    compute_druglike_properties, 
    cal_mean_druglike_properties, 
    compute_posebusters_parallel, 
    get_posebusters_summary,
    compute_basic_metrics_confseq,
    get_similarity_all_mols,
    compute_similarity_statistics,
    compute_diversity_metrics,
    compute_basic_metrics_baseline
)
from src.utils.misc import load_config, load_pickle, get_logger, load_smileslist


def main():
    """
    Main function to evaluate confseq results, compute various metrics, and save the final summary.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, required=True, help='Path to the result file')
    parser.add_argument('--posebusters', default=False, action='store_true', help='Whether to compute PoseBusters metrics')
    parser.add_argument('--druglike', default=False, action='store_true', help='Whether to compute druglike properties')
    parser.add_argument('--diversity', default=False, action='store_true', help='Whether to compute diversity metrics')
    args = parser.parse_args()

    # set save path
    save_path = os.path.dirname(args.result_path)

    # Set up logger
    logger = get_logger('evaluate_baseline', save_path)
    logger.info('---------------------------------------------')

    # Load reference SMILES
    ref_smiles = load_pickle('data/MOSES/shapemol/MOSES2_training_val_smiles.pkl')
    ref_mols = load_pickle('data/MOSES/shapemol/MOSES2_test_mol.pkl')
    logger.info(f'Loaded {len(ref_smiles)} reference SMILES from MOSES2 train_val dataset.\n')

    # set up df
    metrics_df = pd.DataFrame()
    logger.info('*****************************************')
    logger.info('Evaluating generated SMILES...')

    # load data
    gen_data = load_pickle(args.result_path)
    logger.info(f'Loaded {len(gen_data)} generated data')

    # convert to mols list
    mols = [mol for mol_list in gen_data for mol in mol_list]
    logger.info(f'Loaded {len(mols)} generated molecules')

    # Compute similarity metrics
    logger.info('Computing similarity metrics...')
    shape_similarity_all, tanimoto_similarity_all = get_similarity_all_mols(ref_mols, gen_data)
    similarity_metrics = compute_similarity_statistics(shape_similarity_all, tanimoto_similarity_all)
    logger.info(similarity_metrics)

    # Compute basic metrics
    logger.info('Computing basic metrics...')
    basic_metrics = compute_basic_metrics_baseline(mols, ref_smiles, num_samples=50000)
    basic_metrics = basic_metrics.map(lambda x: round(x * 100, 2))
    logger.info(basic_metrics)

    # Compute diversity metrics (optional)
    if args.diversity:
        logger.info('Computing diversity metrics...')
        diversity_metrics = compute_diversity_metrics(mols, logger)
        logger.info(diversity_metrics)
    else:
        diversity_metrics = pd.DataFrame()
        logger.info('Diversity metrics computation skipped.')

    # Compute PoseBusters metrics (optional)
    if args.posebusters:
        logger.info('Computing PoseBusters metrics...')
        pb_df = compute_posebusters_parallel(mols, 
                                    save_path=f'{save_path}/posebusters.csv', max_workers=20)
        logger.info(f'PoseBusters metrics saved to {save_path}/posebusters.csv')
    else:
        pb_df = None
        logger.info('PoseBusters metrics computation skipped.')

    # Compute druglike properties (optional)
    if args.druglike:
        logger.info('Computing druglike properties...')
        druglike_metrics = compute_druglike_properties(mols, save_path=f'{save_path}/druglike.csv',
                                                        max_workers=40)
        logger.info(f'Druglike properties saved to {save_path}/druglike.csv')
    else:
        druglike_metrics = None
        logger.info('Druglike properties computation skipped.')

    # Merge all metrics and add to final DataFrame
    pb_summary, druglike_summary = None, None
    if pb_df is not None or os.path.exists(f'{save_path}/posebusters.csv'):
        pb_df = pd.read_csv(f'{save_path}/posebusters.csv')
        pb_summary = get_posebusters_summary(pb_df, num_samples=50000)
    if druglike_metrics is not None or os.path.exists(f'{save_path}/druglike.csv'):
        druglike_metrics = pd.read_csv(f'{save_path}/druglike.csv')
        druglike_summary = cal_mean_druglike_properties(druglike_metrics, num_samples=50000)

    if pb_summary is not None and druglike_summary is not None:
        metrics_df = pd.concat([metrics_df, basic_metrics, pb_summary, druglike_summary], axis=1)
    elif pb_summary is not None:
        metrics_df = pd.concat([metrics_df, basic_metrics, pb_summary], axis=1)
    elif druglike_summary is not None:
        metrics_df = pd.concat([metrics_df, basic_metrics, druglike_summary], axis=1)
    else:
        metrics_df = pd.concat([metrics_df, basic_metrics], axis=1)
    logger.info('*****************************************')

    # Append similarity metrics and optional diversity metrics
    metrics_df = pd.concat([metrics_df, similarity_metrics, diversity_metrics], axis=1)

    # Save the final metrics DataFrame
    metrics_df.to_csv(os.path.join(save_path, 'final_metrics_summary.csv'))
    logger.info(f"Final metrics summary saved to {save_path}/final_metrics_summary.csv")
    logger.info('----------------------------------------')
    logger.info('Evaluation finished!')

if __name__ == '__main__':
    main()
