# train split
python src/preprocess/build_pointcloud_lmdb.py --split train 
python src/preprocess/build_pointcloud_lmdb.py --split train --aug_mode 1 --augtimes 2
python src/preprocess/merge_lmdb_new.py

# test split
python src/preprocess/build_pointcloud_lmdb.py --data_path ./data/MOSES/MOSES2_test_mol.pkl --split test --map_size 1