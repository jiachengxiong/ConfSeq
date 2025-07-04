# sample confseq
python src/sampling/unconditional/0_sample_confseq_unconditional_partialtemp.py
# convert to rdmol 
python src/sampling/unconditional/1_get_mol_from_confseq.py

# use taskset to control cpu
taskset -c 0-40 python src/evaluation/unconditional/evaluate_confseq.py --config ./configs/unconditional_generation.yaml --dataset geom --posebusters --diversity --druglike --geometry --rdkit_rmsd_tfd