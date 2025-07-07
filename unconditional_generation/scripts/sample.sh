export CUDA_VISIBLE_DEVICES=3

# sample confseq
python src/sampling/0_sample_confseq_unconditional_partialtemp.py
# convert to rdmol 
python src/sampling/1_get_mol_from_confseq.py