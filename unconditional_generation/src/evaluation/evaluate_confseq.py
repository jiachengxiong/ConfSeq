import sys
sys.path.append('.')
import os
import argparse
import pandas as pd
from rdkit.Chem import SDMolSupplier
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from src.utils.scoring_func import (compute_druglike_properties, 
                                    cal_mean_druglike_properties, 
                                    compute_posebusters_parallel,
                                    get_posebusters_summary,
                                    compute_basic_metrics_confseq,
                                    compute_diversity_metrics,
                                    compute_geometry_metrics,
                                    get_rdkit_rmsd_tfd)
from src.utils.misc import load_config, load_pickle, get_logger, load_smileslist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/unconditional_generation.yaml', help='Path to the config file')
    parser.add_argument('--dataset', type=str, choices=['qm9', 'geom'], help='Dataset to evaluate')
    parser.add_argument('--diversity', default=False, action='store_true', help='Whether to compute diversity metrics')
    parser.add_argument('--posebusters', default=False, action='store_true', help='Whether to compute PoseBusters metrics')
    parser.add_argument('--druglike', default=False, action='store_true', help='Whether to compute druglike properties')
    parser.add_argument('--geometry', default=False, action='store_true', help='Whether to compute geometry metrics')
    parser.add_argument('--rdkit_rmsd_tfd', default=False, action='store_true', help='Whether to compute RDKit RMSD')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set up logger
    logger = get_logger('evaluate_confseq', config.save_path)
    logger.info('---------------------------------------------')

    # load reference SMILES
    if args.dataset == 'qm9':
        ref_smiles = load_pickle('data/QM9/qm9_processed/qm9_smiles.pkl')
    elif args.dataset == 'geom':
        train_smiles = load_pickle('data/geom/geom_smiles/train_smiles.pkl')
        test_smiles = load_pickle('data/geom/geom_smiles/test_smiles.pkl')
    logger.info(f'Loaded {len(train_smiles)} train SMILES from {args.dataset}\n')
    logger.info(f'Loaded {len(test_smiles)} test SMILES from {args.dataset}\n')

    final_metrics_df = pd.DataFrame()
    for seed_i in range(getattr(config, 'sample_times', 1)):
        logger.info('*****************************************')
        run_save_path = os.path.join(config.save_path, f'run_{seed_i}')
        logger.info('Evaluating generated SMILES...')

        # Load generated SMILES
        gen_smiles = load_smileslist(os.path.join(run_save_path, 'generated_smiles.txt'))
        logger.info(f'Loaded {len(gen_smiles)} generated SMILES')

        # load generated molecules
        mols = [mol for mol in SDMolSupplier(os.path.join(run_save_path, 'generated_mols.sdf')) if mol is not None]
        logger.info(f'Loaded {len(mols)} generated molecules\n')

        # Compute basic metrics
        logger.info('Computing basic metrics...')
        basic_metrics = compute_basic_metrics_confseq(gen_smiles, 
                                                      train_smiles=train_smiles, 
                                                      test_smiles=test_smiles, 
                                                      num_samples=len(gen_smiles))
        logger.info(basic_metrics)

        # Compute PoseBusters metrics
        if args.posebusters and not os.path.exists(f'{run_save_path}/posebusters.csv'):
            logger.info('Computing PoseBusters metrics...')
            pb_df = compute_posebusters_parallel(mols, 
                                        save_path=f'{run_save_path}/posebusters.csv')
            logger.info(f'PoseBusters metrics saved to {run_save_path}/posebusters.csv')
        else:
            pb_df = None
            logger.info('PoseBusters metrics computation skipped.')

        # Compute druglike properties
        if args.druglike and not os.path.exists(f'{run_save_path}/druglike.csv'):
            logger.info('Computing druglike properties...')
            druglike_metrics = compute_druglike_properties(mols, save_path=f'{run_save_path}/druglike.csv',
                                                           max_workers=config.num_workers)
            logger.info(f'Druglike properties saved to {run_save_path}/druglike.csv')
        else:
            druglike_metrics = None
            logger.info('Druglike properties computation skipped.')

        # Compute diversity metrics (optional)
        if args.diversity:
            logger.info('Computing diversity metrics...')
            diversity_metrics = compute_diversity_metrics(mols, logger)
            logger.info(diversity_metrics)
        else:
            diversity_metrics = None
            logger.info('Diversity metrics computation skipped.')

        # compute geometry metrics (optional)
        if args.geometry:
            logger.info('Computing geometry metrics...')
            geometry_metrics = compute_geometry_metrics(mols)
            # only keep 'bond_length_mean', 'bond_angle_mean', 'dihedral_angle_mean'
            geometry_metrics = {key: geometry_metrics[key] for key in ['bond_length_mean', 'bond_angle_mean', 'dihedral_angle_mean']}
            geometry_metrics = pd.DataFrame(geometry_metrics, index=[0])
            logger.info(geometry_metrics)
        else:
            geometry_metrics = None
            logger.info('Geometry metrics computation skipped.')

        # compute rdkit rmsds(optional)
        if args.rdkit_rmsd_tfd:
            logger.info('Computing rdkit rmsds...')
            rmsd_tfd_metric = get_rdkit_rmsd_tfd(mols, save_path=os.path.join(run_save_path, 'rdkit_rmsd_tfd.pkl'))
            logger.info(rmsd_tfd_metric)
        else:
            rmsd_tfd_metric = None
            logger.info('RDKit rmsd computation skipped.')
            
        # Merge all metrics and add to final DataFrame
        pb_summary = None
        druglike_summary = None
        if os.path.exists(f'{run_save_path}/posebusters.csv'):
            logger.info('Loading PoseBusters metrics...')
            pb_df = pd.read_csv(f'{run_save_path}/posebusters.csv')
            pb_summary = get_posebusters_summary(pb_df, num_samples=len(mols))
            logger.info(pb_summary)
        if os.path.exists(f'{run_save_path}/druglike.csv'):
            logger.info('Loading druglike properties...')
            druglike_metrics = pd.read_csv(f'{run_save_path}/druglike.csv')
            druglike_summary = cal_mean_druglike_properties(druglike_metrics)
            logger.info(druglike_summary)

        # only concat available metrics
        metrics_df = pd.concat([basic_metrics, 
                                diversity_metrics, 
                                pb_summary, 
                                druglike_summary, 
                                geometry_metrics, 
                                rmsd_tfd_metric], axis=1)
        logger.info(metrics_df)

        # Append the metrics to the final DataFrame
        final_metrics_df = pd.concat([final_metrics_df, metrics_df], axis=0)
        logger.info('*****************************************')

    # Calculate average and standard deviation for all runs
    logger.info('Calculating summary metrics...')
    mean_std_df = final_metrics_df.agg(['mean', 'std']).T
    no_scale_columns = ['QED', 'SAS', 'logP', 'TPSA', 
                        'IntDiv', 'SumBottleneck', 'NumCircles',
                        'FCD', 'SNN', 'Frag', 'Scaf', 
                        'bond_length_mean', 'bond_angle_mean', 'dihedral_angle_mean',
                        'RMSD', 'TFD']
    mean_std_df['mean_std'] = mean_std_df.apply(
        lambda row: (
            f"{row['mean']:.4g}±{row['std']:.2f}" if row.name in no_scale_columns
            else f"{(row['mean'] * 100):.2f}±{(row['std'] * 100):.2f}"
        ),
        axis=1
    )

    # Append the mean and std values as a new row
    summary_row = pd.DataFrame(mean_std_df['mean_std']).T
    summary_row.columns = final_metrics_df.columns
    summary_row.index = ['summary']
    final_metrics_df = pd.concat([final_metrics_df, summary_row], axis=0)

    # Save the final metrics DataFrame
    final_metrics_df.to_csv(os.path.join(config.save_path, 'final_metrics_summary.csv'))
    logger.info(f"Final metrics summary saved to {config.save_path}/final_metrics_summary.csv")
    logger.info('----------------------------------------')
    logger.info('Evaluation finished!')

if __name__ == '__main__':
    main()
