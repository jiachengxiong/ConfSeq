# use taskset to control cpu
taskset -c 0-40 python src/evaluation/evaluate_confseq.py --config ./configs/unconditional_generation.yaml --posebusters --rdkit_rmsd