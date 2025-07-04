# convert the GEOM-DRUGS dataset from xyz format to sdf format
python src/preprocess/convert_xyz2mol_geomdrugs.py --geom_file ./data/geom/geom_raw/geom_drugs_30.npy

# convert sdf to ConfSeq
python src/preprocess/convert_mol2TDsmiles_geomdrugs.py --input_dir data/geom/geom_sdf/train.sdf --output_dir data/geom/geom_TDsmiles/ --num_workers 30 --aug_mode 1 --aug_time
s 10 --do_random