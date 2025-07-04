import sys
sys.path.append('./')
import os
import argparse
import shutil

from tqdm.contrib.concurrent import process_map
import numpy as np
import pandas as pd
from torch.utils.data import random_split
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, set_seed
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from src.utils.misc import load_config, load_pickle
from src.model.dataset import PointCloudDataset, PointCloudCollator
from src.model.tokenizer import WhitespaceTokenizer
from src.model.SurfBart import SurfaceBartv2
from src.utils.scoring_func import (compute_basic_metrics_confseq, 
                                    compute_basic_metrics_baseline,
                                    compute_similarity_dataframe,
                                    )
from src.utils.reconstruct import convert_tdsmiles_to_mol, convert_smiles_to_mol


def compute_similarity_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 group_id 分组后，计算 shape_similarity 和 graph_similarity 的组内均值，
    并对这些组均值分别计算整体均值和标准差，最终以“均值±标准差”字符串形式返回。

    参数：
    - df: DataFrame，必须包含以下三列
        • group_id          ：分组标识
        • shape_similarity  ：形状相似度
        • graph_similarity  ：图谱（Tanimoto）相似度

    返回：
    - summary: DataFrame，索引为 ['shape', 'graph']，
        列为 ['mean±std']，数值均为格式化后的字符串形式。
    """
    # 1. 按组聚合，得到每组的均值
    df_group = (
        df
        .groupby('group_id', as_index=False)
        .agg(
            shape_mean=('shape_similarity', 'mean'),
            graph_mean=('graph_similarity', 'mean')
        )
    )

    # 2. 分别计算“组均值”的整体均值 & 标准差
    shape_mean_of_means = df_group['shape_mean'].mean()
    graph_mean_of_means = df_group['graph_mean'].mean()

    # 3. 格式化为 “均值±标准差” 的字符串，保留三位小数
    summary = pd.DataFrame({
        'Avg_shape': [
            shape_mean_of_means
        ],
        'Avg_graph': [
            graph_mean_of_means
        ]
    })

    return summary



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/surfbart_training.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(config['train']['output_dir'], exist_ok=True)
    shutil.copyfile(args.config, os.path.join(config['train']['output_dir'], 'config.yaml'))

    tokenizer = WhitespaceTokenizer()
    set_seed(config['train']['seed'])

    # 加载数据
    dataset = PointCloudDataset(config['data'], tokenizer, max_length=config['model']['bart']['max_position_embeddings'])
    train_size = len(dataset) - 1000
    train_dataset, valid_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    collator = PointCloudCollator()

    # load train smiles
    train_smiles = load_pickle(config['data']['train_smiles_dir'])
    print(f'Loaded {len(train_smiles)} training smiles from MOSES2 dataset.')

    # 加载模型
    model = SurfaceBartv2(config['model'], tokenizer=tokenizer)
    if config['train']['resume_path']:
        model.load_weights(config['train']['resume_path'])
        print(f"Loaded model from {config['train']['resume_path']}")
    else:
        print("Training from scratch.")

    # 定义自定义评估指标
    def compute_metrics(eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids
        
        gen_smiles = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref_smiles = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        gen_smiles = [smiles.lstrip('<aug>').lstrip('<std>').lstrip() for smiles in gen_smiles]
        ref_smiles = [smiles.lstrip('<aug>').lstrip('<std>').lstrip() for smiles in ref_smiles]

        batched_gen_smiles = [gen_smiles[i : i+config['model']['generation_config']['num_return_sequences']]
                              for i in range(0, len(gen_smiles), config['model']['generation_config']['num_return_sequences'])]
    
        # 根据配置选择不同的分子构建和指标计算方式
        try:
            if config['data']['use_smiles']:
                # SMILES场景
                all_results_pred = process_map(
                    convert_smiles_to_mol,
                    gen_smiles,
                    max_workers=config['data']['num_workers'],
                    chunksize=20, 
                    disable=True
                )
                all_results_label = process_map(
                    convert_smiles_to_mol,
                    ref_smiles,
                    max_workers=config['data']['num_workers'],
                    chunksize=20,
                    disable=True
                )
            else:
                # TDSmiles场景
                all_results_pred = process_map(
                    convert_tdsmiles_to_mol,
                    gen_smiles,
                    max_workers=config['data']['num_workers'],
                    chunksize=20,
                    disable=True
                )
                all_results_label = process_map(
                    convert_tdsmiles_to_mol,
                    ref_smiles,
                    max_workers=config['data']['num_workers'],
                    chunksize=20,
                    disable=True
                )
                # basic metrics
                basic_metrics = pd.DataFrame()
                for smiles_list in batched_gen_smiles:
                    basic_metrics_ = compute_basic_metrics_confseq(smiles_list, 
                                                                train_smiles=train_smiles,
                                                                test_smiles=None,
                                                                num_samples=len(smiles_list))
                    basic_metrics = pd.concat([basic_metrics, basic_metrics_])

            # 对预测结果进行分批处理（用于后续相似度计算）
            batch_size = config['model']['generation_config']['num_return_sequences']
            batched_results = [all_results_pred[i : i+batch_size] 
                            for i in range(0, len(all_results_pred), batch_size)]

            # 仅保留有效的结果（分子转换错误会返回字符串“Error:”）
            gen_data = []
            for i, results in enumerate(batched_results):
                clean = []
                for result in results:
                    if isinstance(result, str) and result.startswith('Error:'):
                        continue
                    elif result is not None:
                        clean.append(result)
                gen_data.append(clean)

            if config['data']['use_smiles']:
                # 计算基本指标
                basic_metrics = pd.DataFrame()
                for mols in gen_data:
                    basic_metrics_ = compute_basic_metrics_baseline(mols, 
                                                                    train_smiles=train_smiles,
                                                                    test_smiles=None,
                                                                    num_samples=len(mols))
                    basic_metrics = pd.concat([basic_metrics, basic_metrics_])

            # 计算相似度（形状相似度和Tanimoto）
            similarity_df = compute_similarity_dataframe(
                all_results_label,
                gen_data,
                method='rdkit',
                has_scores=False)
            similarity_summary = compute_similarity_statistics(similarity_df)

            # 汇总所有指标
            metrics = {
                'avg_Validity': np.mean(basic_metrics['Validity']),
                'avg_Uniqueness': np.mean(basic_metrics['Uniqueness']),
                'avg_Validity * Uniqueness': np.mean(basic_metrics['Validity * Uniqueness']),
                'avg_Novelty': np.mean(basic_metrics['Novelty']),
                'avg_shape_similarity': similarity_summary['Avg_shape'][0],
                'avg_tanimoto_similarity': similarity_summary['Avg_graph'][0],
            }
            print(metrics)
        
        except Exception as e:
            print(f"Error: {e}")
            metrics = {
                'avg_Validity': 0,
                'avg_Uniqueness': 0,
                'avg_Validity * Uniqueness': 0,
                'avg_shape_similarity': 0,
                'max_shape_similarity': 0,
                'avg_tanimoto_similarity': 0,
                'max_tanimoto_similarity': 0,
            }

        return metrics


    # 训练参数设定
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['train']['output_dir'],
        overwrite_output_dir=config['train']['overwrite_output_dir'],
        num_train_epochs=config['train']['num_train_epochs'],
        per_device_train_batch_size=config['train']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['train']['per_device_eval_batch_size'],
        save_total_limit=config['train']['save_total_limit'],
        logging_steps=config['train']['logging_steps'],
        eval_strategy=config['train']['eval_strategy'],
        eval_steps=config['train']['eval_steps'],
        do_eval=config['train']['do_eval'],
        learning_rate=float(config['train']['learning_rate']),
        dataloader_num_workers=config['train']['dataloader_num_workers'],
        warmup_ratio=config['train']['warmup_ratio'],
        save_strategy=config['train']['save_strategy'],
        save_steps=config['train']['save_steps'],
        load_best_model_at_end=config['train']['load_best_model_at_end'],
        logging_first_step=config['train']['logging_first_step'],
        report_to='wandb',
        bf16=config['train']['bf16'],
        save_only_model=True,
        save_safetensors=False,
        predict_with_generate=True,
        eval_on_start=True,
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config['train']['early_stopping_patience'],
        early_stopping_threshold=config['train']['early_stopping_threshold'],
    )

    # 创建Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[early_stopping],
        compute_metrics=compute_metrics
    )

    print(f"Total number of parameters: {trainer.get_num_trainable_parameters() / 1e6:.2f}M")

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
