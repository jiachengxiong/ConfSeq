export CUDA_VISIBLE_DEVICES=0
# sample confseq
taskset -c 0-40 python src/sampling/conditional/0_sample_confseq_surfbart_scores_prefix.py
# convert to rdmol 
python src/sampling/conditional/1_get_mol_from_confseq.py
# evaluate
python src/evaluation/conditional/evaluate_confseq.py --method shaep