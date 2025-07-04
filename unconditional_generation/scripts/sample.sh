export CUDA_VISIBLE_DEVICES=0

# sample confseq
python src/sampling/unconditional/0_sample_confseq_unconditional_partialtemp.py
# convert to rdmol 
python src/sampling/unconditional/1_get_mol_from_confseq.py