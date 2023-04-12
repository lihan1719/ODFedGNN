import os
from argparse import ArgumentParser

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import wandb
import models
from utils import SaveNodeEncodings

#导入模型
from models.od_prediction.od_Centralized import *

import warnings

# warnings.filterwarnings('ignore', category=UserWarning)


def main(args, temp_args):
    seed_everything(args.seed)
    # 是否启用wandb（调试时禁止）
    if temp_args.wandb == True:
        logname = ''
        if hasattr(temp_args, 'base_model_name'):
            logname += temp_args.model_name
            logname += '-'
        logname += '{}-{}-seed-{}'.format(temp_args.dataset,
                                          temp_args.base_model_name,
                                          temp_args.seed)
        tags = [
            temp_args.dataset, temp_args.model_name, temp_args.base_model_name
        ]
        wandb.init(name=logname,
                   config=temp_args,
                   project=args.project,
                   tags=tags,
                   resume=args.resume,
                   id=args.id,
                   dir='./cash/wandb_logger')
        config = wandb.config
        for key, value in config.items():
            setattr(args, key, value)
    # 模型保存（保留最后一个，以及val最好的一个）
    # 提前停止训练
    early_stop_callback = EarlyStopping(monitor='val/weighted_loss',
                                        patience=args.early_stop_patience,
                                        mode='min',
                                        verbose=True)
    checkpoint_callback = ModelCheckpoint(monitor='val/weighted_loss',
                                          save_top_k=1,
                                          save_last=True,
                                          mode='min',
                                          verbose=True)

    trainer = Trainer.from_argparse_args(
        args,
        default_root_dir='cash/artifacts',
        deterministic=True,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        gpus=args.gpus,
    )

    if args.model_name == 'HApredicter':
        ha = HApredicter(hparams=args, TIMESTEP_IN=6, TIMESTEP_OUT=6)
        mse, mae, rmse = ha.train_test()
        wandb.log({'test/mse': mse, 'test/mae': mae, 'test/rmse': rmse})
        return None

    if args.train:
        model = getattr(models, args.model_name)(args)
        # 从已经训练的模型中导入
        if args.restore_train_ckpt_path != '':
            trainer = Trainer(
                resume_from_checkpoint=args.restore_train_ckpt_path,
                default_root_dir='cash/artifacts',
                deterministic=True,
                early_stop_callback=early_stop_callback,
                checkpoint_callback=checkpoint_callback,
                gpus=args.gpus,
            )
        trainer.fit(model)

    # 测试（如果有需要的话）
    if args.load_test_ckpt_path != '':
        load_test_ckpt_path = args.load_test_ckpt_path
    else:
        load_test_ckpt_path = checkpoint_callback.best_model_path
    print(load_test_ckpt_path)
    model = getattr(models,
                    args.model_name).load_from_checkpoint(load_test_ckpt_path)
    trainer.test(model)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, default='NYC_bike')
    parser.add_argument('--data_name',
                        type=str,
                        default='nyc_2019-07_2019-09_60T')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--model_name',
                        type=str,
                        default='ODFedGNN',
                        help='name of the model')
    parser.add_argument('--base_model_name',
                        type=str,
                        default='GRUSeq2SeqWithGraphNet')
    parser.add_argument('--wandb', action='store_true')
    temp_args, _ = parser.parse_known_args()
    ModelClass = getattr(models, temp_args.model_name)
    BaseModelClass = getattr(models, temp_args.base_model_name, None)
    parser = ModelClass.add_model_specific_args(parser)
    if BaseModelClass is not None:
        parser = BaseModelClass.add_model_specific_args(parser)
    temp_args, _ = parser.parse_known_args()
    parser.add_argument('--load_test_ckpt_path', type=str, default='')
    parser.add_argument('--restore_train_ckpt_path', type=str, default='')
    parser.add_argument('--notrain', dest='train', action='store_false')
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--project', type=str, default='odfedgnn')
    # 设置wandb参数
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/NYC_bike/processed_bike_data/')
    parser.add_argument('--id', type=str, default='')
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args, temp_args)
