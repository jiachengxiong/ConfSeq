import pickle
import numpy as np
import argparse

import pandas as pd
from easydict import EasyDict
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem.rdMolAlign import GetBestRMS

long_term_log = []

def reshape_ref_set(raw_ref_set):
    """
    reshape the raw ref set to a dict with smiles as key and a list of rdmol as value.
    """
    ref_set = {}

    for raw_data in raw_ref_set:
        smiles = raw_data['smiles']
        if smiles not in ref_set:
            ref_set[smiles] = []
        ref_set[smiles].append(Chem.RemoveHs(raw_data['rdmol']))

    return ref_set

def get_rmsd_confusion_matrix(test_mol_list, ref_mol_list, useFF=False):
    num_test = len(test_mol_list)
    num_ref = len(ref_mol_list)

    rmsd_confusion_mat = -1 * np.ones([num_ref, num_test], dtype=np.float64)

    for i in range(num_test):
        test_mol = Chem.RemoveHs(test_mol_list[i])
        if useFF:
            Chem.AddHs(test_mol)
            MMFFOptimizeMolecule(test_mol)
            Chem.RemoveHs(test_mol)

        for j in range(num_ref):
            try:
                rmsd_confusion_mat[j, i] = GetBestRMS(test_mol, ref_mol_list[j])
            except:
                # add to log 
                long_term_log.append((Chem.MolToSmiles(test_mol), Chem.MolToSmiles(ref_mol_list[j])))
                
    return rmsd_confusion_mat

def evaluate_conf(test_set: dict, ref_set: dict, useFF=False, threshold=1.25):
    covr_scores = []
    matr_scores = []
    covp_scores = []
    matp_scores = []

    single_molecule_results = []

    for smiles in tqdm(test_set.keys()):
        test_mol_list = test_set[smiles]
        ref_mol_list = ref_set[smiles]
        confusion_mat = get_rmsd_confusion_matrix(test_mol_list, ref_mol_list, useFF=useFF)
        
        rmsd_ref_min = confusion_mat.min(-1)
        rmsd_gen_min = confusion_mat.min(0)
        rmsd_cov_thres = rmsd_ref_min.reshape(-1, 1) <= threshold
        rmsd_jnk_thres = rmsd_gen_min.reshape(-1, 1) <= threshold

        covr_score = rmsd_cov_thres.mean()
        matr_score = rmsd_ref_min.mean(0, keepdims=True)
        covp_score = rmsd_jnk_thres.mean()
        matp_score = rmsd_gen_min.mean(0, keepdims=True)

        covr_scores.append(covr_score)
        matr_scores.append(matr_score)
        covp_scores.append(covp_score)
        matp_scores.append(matp_score)

        single_molecule_results.append({
            'SMILES': smiles,
            'CoverageR': rmsd_cov_thres.flatten(),
            'MatchingR': rmsd_ref_min.flatten(),
            'CoverageP': rmsd_jnk_thres.flatten(),
            'MatchingP': rmsd_gen_min.flatten()
        })

    covr_scores = np.vstack(covr_scores)
    matr_scores = np.array(matr_scores)
    covp_scores = np.vstack(covp_scores)
    matp_scores = np.array(matp_scores)

    results = EasyDict({
        'CoverageR': covr_scores,
        'MatchingR': matr_scores,
        'thresholds': threshold,
        'CoverageP': covp_scores,
        'MatchingP': matp_scores,
        'single_molecule_results': single_molecule_results
    })

    return results


def print_covmat_results(results):
    summary_df = pd.DataFrame({
        'COV-R_mean': np.round(np.mean(results.CoverageR, 0), 4),
        'COV-R_median': np.round(np.median(results.CoverageR, 0), 4),
        'MAT-R_mean': np.round(np.mean(results.MatchingR), 5),
        'MAT-R_median': np.round(np.median(results.MatchingR), 5),    
        'COV-P_mean': np.round(np.mean(results.CoverageP, 0), 4),
        'COV-P_median': np.round(np.median(results.CoverageP, 0), 4),
        'MAT-P_mean': np.round(np.mean(results.MatchingP), 5),
        'MAT-P_median': np.round(np.median(results.MatchingP), 5),
        'COV-R_std': np.round(np.std(results.CoverageR, 0), 4),
        'MAT-R_std': np.round(np.std(results.MatchingR), 5),
        'COV-P_std': np.round(np.std(results.CoverageP, 0), 4),
        'MAT-P_std': np.round(np.std(results.MatchingP), 5),
    })
    print('\nSummary Results:\n' + str(summary_df))

    single_molecule_results = results.single_molecule_results
    single_molecule_data = []

    for res in single_molecule_results:
        single_molecule_data.append({
            'SMILES': res['SMILES'],
            'CoverageR_mean': np.round(np.mean(res['CoverageR']), 4),
            'CoverageR_median': np.round(np.median(res['CoverageR']), 4),
            'MatchingR_mean': np.round(np.mean(res['MatchingR']), 4),
            'MatchingR_median': np.round(np.median(res['MatchingR']), 4),
            'CoverageP_mean': np.round(np.mean(res['CoverageP']), 4),
            'CoverageP_median': np.round(np.median(res['CoverageP']), 4),
            'MatchingP_mean': np.round(np.mean(res['MatchingP']), 4),
            'MatchingP_median': np.round(np.median(res['MatchingP']), 4),
            'CoverageR_std': np.round(np.std(res['CoverageR']), 4),
            'MatchingR_std': np.round(np.std(res['MatchingR']), 4),
            'CoverageP_std': np.round(np.std(res['CoverageP']), 4),
            'MatchingP_std': np.round(np.std(res['MatchingP']), 4)
        })

    single_molecule_df = pd.DataFrame(single_molecule_data)
    print('\nSingle Molecule Results:\n' + str(single_molecule_df))

    return summary_df, single_molecule_df



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_dir', type=str)
    parser.add_argument('--ref_data_dir', type=str, default='test_data_200.pkl')
    parser.add_argument('--threshold', type=float, default=1.25)
    parser.add_argument('--useFF', action='store_true')
    args = parser.parse_args()

    # load data
    with open(args.test_data_dir, 'rb') as f:
        test_set = pickle.load(f)
    with open(args.ref_data_dir, 'rb') as f:
        raw_ref_set = pickle.load(f)

    # reshape ref_set
    ref_set = reshape_ref_set(raw_ref_set)

    # evaluate
    results = evaluate_conf(test_set, ref_set, useFF=args.useFF, threshold=args.threshold)
    summary_df, single_molecule_df = print_covmat_results(results)

    # save result
    summary_csv_fn = args.test_data_dir[:-4] + '_covmat_summary.csv'
    single_molecule_csv_fn = args.test_data_dir[:-4] + '_single_molecule_covmat.csv'
    results_fn = args.test_data_dir[:-4] + '_covmat.pkl'

    summary_df.to_csv(summary_csv_fn, index=False)
    single_molecule_df.to_csv(single_molecule_csv_fn, index=False)
    # with open(results_fn, 'wb') as f:
    #     pickle.dump(results, f)

    long_term_log = list(set(long_term_log))
    print(long_term_log)
