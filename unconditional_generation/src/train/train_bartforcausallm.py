import os
import argparse
import sys
sys.path.append('./')
import shutil
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from src.model.dataset import SMILESDataset
from src.utils.misc import load_pickle, load_config, get_logger
from src.model.tokenizer import WhitespaceTokenizer
from src.model.BartForCausalLM import MyBartForCausalLM
from src.utils.scoring_func import compute_basic_metrics_confseq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train_drugs.yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    os.makedirs(config['train']['output_dir'], exist_ok=True)
    logger = get_logger('BartForCausalLM', config['train']['output_dir'])
    
    logger.info("Configuration loaded successfully.")
    shutil.copy(args.config, config['train']['output_dir'])
    logger.info(f"Configuration file copied to {config['train']['output_dir']}")
    
    tokenizer = WhitespaceTokenizer()
    logger.info(f"Vocab size: {tokenizer.vocab_size}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['train']['device'])
    logger.info(f"Device: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    logger.info("Loading datasets...")
    train_dataset = SMILESDataset(load_pickle(config['train_data']), tokenizer, max_length=config['model']['bart']['max_position_embeddings'])
    valid_dataset = SMILESDataset(load_pickle(config['valid_data']), tokenizer, max_length=config['model']['bart']['max_position_embeddings'])
    logger.info(f"Train dataset size: {len(train_dataset)}, Valid dataset size: {len(valid_dataset)}")
    
    subset_size = 10000
    if len(valid_dataset) >= subset_size:
        sampled_indices = np.random.choice(len(valid_dataset), subset_size, replace=False)
        valid_dataset = [valid_dataset[i] for i in sampled_indices]
        logger.info(f"Valid dataset downsampled to {subset_size} samples.")
    else:
        logger.error(f"Dataset size ({len(valid_dataset)}) is smaller than subset size ({subset_size}).")
        raise ValueError(f"Dataset size ({len(valid_dataset)}) is smaller than subset size ({subset_size}).")
    
    # load train smiles to cal novelty
    train_smiles = load_pickle('data/geom/geom_smiles/train_smiles.pkl')
    logger.info(f"Loaded {len(train_smiles)} training SMILES strings for novelty calculation.")

    # load model
    model = MyBartForCausalLM(config['model'], tokenizer)
    if config['train']['resume_path']:
        logger.info(f"Loading model from {config['train']['resume_path']}")
        model.load_weights(config['train']['resume_path'])
    else:
        logger.info("No resume path provided. Initializing model from scratch.")
    logger.info("Model loaded successfully.")

    def compute_metrics(eval_pred):
        pred_ids = eval_pred.predictions
        gen_smiles = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        logger.info(f"Generated {len(gen_smiles)} SMILES strings for evaluation.")

        basic_metrics = compute_basic_metrics_confseq(
                gen_smiles, 
                train_smiles=train_smiles, 
                test_smiles=None,
                num_samples=len(gen_smiles)
            )
        logger.info(f"Basic metrics computed: {basic_metrics}")
        
        metrics = {
            'Validity': basic_metrics['Validity'][0],
            'Uniqueness': basic_metrics['Uniqueness'][0],
            'Validity * Uniqueness': basic_metrics['Validity * Uniqueness'][0],
            'Novelty': basic_metrics['Novelty'][0],
        }
        logger.info(f"Final evaluation metrics: {metrics}")
        return metrics
    
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
        warmup_ratio=config['train']['warmup_ratio'],
        save_strategy=config['train']['save_strategy'],
        save_steps=config['train']['save_steps'],
        dataloader_num_workers=config['train']['dataloader_num_workers'],
        load_best_model_at_end=config['train']['load_best_model_at_end'],
        logging_first_step=config['train']['logging_first_step'],
        report_to='wandb',
        bf16=config['train']['bf16'],
        save_only_model=True,
        predict_with_generate=True,
        metric_for_best_model='Validity * Uniqueness',
        greater_is_better=True,
        save_safetensors=False,
        torch_compile=True,
        ddp_find_unused_parameters=True,
        ddp_backend='nccl',
        eval_on_start=True,
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config['train']['early_stopping_patience'],
        early_stopping_threshold=config['train']['early_stopping_threshold'],
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        callbacks=[early_stopping],
        compute_metrics=compute_metrics,
    )
    
    logger.info(f"Total number of parameters: {trainer.get_num_trainable_parameters() / 1e6:.2f}M")
    trainer.train()
    logger.info("Training complete.")

if __name__ == '__main__':
    main()
