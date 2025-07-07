export CUDA_VISIBLE_DEVICES=7
# sample confseq
taskset -c 0-40 python src/sampling/0_sample_confseq_surfbart_scores_prefix.py
# convert to rdmol 
taskset -c 0-40 python src/sampling/1_get_mol_from_confseq.py