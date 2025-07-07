# train split
python src/preprocess/build_pointcloud_lmdb.py --split train 

# test split
python src/preprocess/build_pointcloud_lmdb.py --data_path ./data/MOSES/MOSES2_test_mol.pkl --split test --map_size 1