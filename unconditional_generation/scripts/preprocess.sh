# convert the GEOM-DRUGS dataset from xyz format to sdf format
python src/preprocess/convert_xyz2mol.py --geom_file ./data/geom/geom_raw/geom_drugs_30.npy

# convert sdf to ConfSeq
python src/preprocess/convert_mol2confseqs.py --input_dir data/geom/geom_sdf/train.sdf --output_dir data/geom/geom_TDsmiles/ --num_workers 30 --aug_mode 1 --aug_time
s 5 --do_random
python src/preprocess/convert_mol2confseqs.py --input_dir data/geom/geom_sdf/valid.sdf --output_dir data/geom/geom_TDsmiles/ --num_workers 30 --aug_mode 0 --aug_time
s 1