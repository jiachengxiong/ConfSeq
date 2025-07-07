import sys
sys.path.append('.')
import os
import argparse
import pandas as pd
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

from tqdm.auto import tqdm
from src.utils.scoring_func import (
    compute_posebusters_parallel, 
    get_posebusters_summary,
    compute_basic_metrics_confseq,
    compute_similarity_dataframe,
    compute_similarity_statistics,
)
from src.utils.misc import load_config, load_pickle, get_logger, load_smileslist, flatten_easydict, save_pickle


def main():
    """
    Main function to evaluate confseq results, compute various metrics, and save the final summary.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/surfbart_generation.yaml', help='Path to the config file')
    parser.add_argument('--train_smiles_dir', type=str, default='data/MOSES/MOSES2_training_val_smiles.pkl', help='Path to the training SMILES directory')
    parser.add_argument('--test_mols_dir', type=str, default='data/MOSES/MOSES2_test_mol.pkl', help='Path to the test molecules directory')
    parser.add_argument('--posebusters', default=False, action='store_true', help='Whether to compute PoseBusters metrics')
    parser.add_argument('--method', type=str, choices=['rdkit', 'shaep'], help='Method used for similarity computation')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    save_path = config['save_path']

    # Set up logger
    logger = get_logger('evaluate_confseq', save_path)
    logger.info('---------------------------------------------')

    # Load test mols
    test_mols = load_pickle(args.test_mols_dir)
    logger.info(f'Loaded {len(test_mols)} reference SMILES from MOSES2 train_val dataset.\n')

    metrics_df = pd.DataFrame([flatten_easydict(config)])
    logger.info('*****************************************')
    logger.info('Evaluating generated SMILES...')

    # Load generated SMILES and molecules
    gen_smiles = load_smileslist(os.path.join(save_path, 'generated_smiles.txt'))
    logger.info(f'Loaded {len(gen_smiles)} generated SMILES')

    gen_data = load_pickle(os.path.join(save_path, 'generated_mols.pkl'))
    logger.info(f'Loaded {len(gen_data)} generated data')

    train_smiles = load_pickle(args.train_smiles_dir)
    logger.info(f'Loaded {len(train_smiles)} training SMILES from MOSES2 train_val dataset.\n')

    # Compute similarity metrics
    logger.info('Computing similarity metrics...')
    similarity_df = compute_similarity_dataframe(test_mols, 
                                            gen_data, 
                                            save_path=os.path.join(save_path, 'similarity.csv'))
    similarity_metrics = compute_similarity_statistics(similarity_df)
    logger.info(similarity_metrics)

    # Compute basic metrics
    logger.info('Computing basic metrics...')
    basic_metrics = pd.DataFrame()
    batched_gen_smiles = [gen_smiles[i : i+config['generation_config']['num_return_sequences']] 
                        for i in range(0, len(gen_smiles), config['generation_config']['num_return_sequences'])]  

    for smiles_list in tqdm(batched_gen_smiles):
        basic_metrics_ = compute_basic_metrics_confseq(smiles_list, 
                                                        train_smiles=train_smiles, 
                                                        num_samples=50)
        basic_metrics = pd.concat([basic_metrics, basic_metrics_], axis=0)
    basic_metrics.to_csv(os.path.join(save_path, 'basic_metrics.csv'))
    logger.info(f'Basic metrics saved to {save_path}/basic_metrics.csv')

    # get basic metrics summary
    basic_metrics_summary = pd.DataFrame()
    for col in basic_metrics.columns:
        basic_metrics_summary[col] = [f'{basic_metrics[col].mean()*100:.2f} ± {basic_metrics[col].std():.4g}']
    logger.info(basic_metrics_summary)

    # compute other optional metrics
    pb_summary = pd.DataFrame()
    for i, mols in tqdm(enumerate(gen_data)):
        if len(mols) == 0:
            continue
        else:
            # Compute PoseBusters metrics (optional)
            if args.posebusters and not os.path.exists(f'{save_path}/posebusters.csv'):
                pb_df_ = compute_posebusters_parallel(mols,
                                            max_workers=10,
                                            chunk_size=1)
                pb_summary_ = get_posebusters_summary(pb_df_, num_samples=len(mols))
                pb_summary = pd.concat([pb_summary, pb_summary_], axis=0)

    # Merge all metrics and add to final DataFrame
    final_pb_summary = pd.DataFrame()
    if not pb_summary.empty or os.path.exists(f'{save_path}/posebusters.csv'):
        pb_summary = pd.read_csv(f'{save_path}/posebusters.csv')
        final_pb_summary = pb_summary.mean()
        final_pb_summary = final_pb_summary.to_frame().T

    # Append similarity metrics and optional diversity metrics
    metrics_df = pd.concat([metrics_df,
                            basic_metrics_summary, 
                            similarity_metrics,
                            pb_summary], axis=1)


    # Save the final metrics DataFrame
    metrics_df.to_csv(os.path.join(save_path, 'final_metrics_summary.csv'))
    logger.info(f"Final metrics summary saved to {save_path}/final_metrics_summary.csv")
    logger.info('----------------------------------------')
    logger.info('Evaluation finished!')

if __name__ == '__main__':
    main()
