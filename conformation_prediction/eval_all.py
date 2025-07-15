import os

lis = os.listdir('./prediction_data/')
lis = ['./prediction_data/' + i for i in lis if i.endswith('.pkl') and 'with_scores' not in i]
for i in lis:
    command = 'python eval_covmat_eachmol.py --test_data_dir {} --ref_data_dir ./raw_data/test_data_200.pkl --threshold 0.75'.format(i)
    os.system(command)