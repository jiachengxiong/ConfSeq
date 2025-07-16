import lmdb, pickle, random
from tqdm import tqdm

# ---------- Adjustable parameters ----------
orig_lmdb_path = 'data/MOSES/pointcloud_train_samples1024_augmode0_augtimes1_lmdb'
aug_lmdb_path  = 'data/MOSES/pointcloud_train_samples1024_augmode1_augtimes2_lmdb'
combined_path  = 'data/MOSES/pointcloud_train_samples1024_std2x_aug2x_merge_lmdb'

orig_dup_factor = 3                      # <std> SMILES lightweight replication factor
heavy_keys      = ('pointcloud', 'normals', 'mesh')
map_size_bytes  = 150 * 1024 ** 3        # 150 GB

# ---------- 1) Collect augmented SMILES ----------
aug_dict = {}  # {rdmol_str: [td1, td2, ...]}
with lmdb.open(aug_lmdb_path, readonly=True, lock=False) as env, env.begin() as txn:
    for _, val in tqdm(txn.cursor(), total=txn.stat()['entries'], desc='Load aug'):
        d = pickle.loads(val); aug_dict.setdefault(d['rdmol'], []).append(d['TD_smiles'])
print(f"[INFO] {len(aug_dict):,} unique rdmol in aug LMDB")

# ---------- 2) Create merged database ----------
def make_lite(td, rdmol, pc_key, typ):
    return {'TD_smiles': td, 'rdmol': rdmol, 'pc_key': pc_key, 'dup_type': typ}

with lmdb.open(combined_path, map_size=map_size_bytes) as comb_env, \
     lmdb.open(orig_lmdb_path, readonly=True, lock=False) as orig_env:

    comb_txn = comb_env.begin(write=True)
    orig_txn = orig_env.begin()
    cursor   = orig_txn.cursor()

    for key, val in tqdm(cursor, total=orig_txn.stat()['entries'], desc='Merge'):
        base = pickle.loads(val)
        base['TD_smiles'] = "<std> " + base['TD_smiles']
        pc_key = key  # Directly reuse original key

        # 2.1 Write base record (with point cloud)
        comb_txn.put(pc_key, pickle.dumps(base))

        # 2.2 Generate n lightweight <std> records
        for i in range(orig_dup_factor):
            lite_key = f"{key.decode()}_std{i}".encode()
            comb_txn.put(lite_key, pickle.dumps(
                make_lite(base['TD_smiles'], base['rdmol'], pc_key, 'std')
            ))

        # 2.3 Generate all <aug> lightweight records
        for j, td_aug in enumerate(aug_dict.get(base['rdmol'], [])):
            lite_key = f"{key.decode()}_aug{j}".encode()
            comb_txn.put(lite_key, pickle.dumps(
                make_lite("<aug> " + td_aug, base['rdmol'], pc_key, 'aug')
            ))

        # Commit in batches as needed
        if cursor.key() and int.from_bytes(cursor.key()[-3:], 'little') % 10000 == 0:
            comb_txn.commit(); comb_txn = comb_env.begin(write=True)

    comb_txn.commit()
print("[DONE] combined LMDB ready.")
