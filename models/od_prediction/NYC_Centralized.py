# 使用集中式模型训练NYC数据集

from argparse import ArgumentParser
from multiprocessing import cpu_count
from copy import deepcopy
from collections import defaultdict
import os

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import to_dense_adj
from torch.utils.data import TensorDataset
from datasets.NYC_BOD import load_nyc_data
from datasets.st_datasets import load_dataset
import models.base_models as base_models
from models.base_models import *
import numpy as np
import wandb


# 定义评价指标
def unscaled_metrics(y_pred, y, name, feature_scaler):
    try:
        y_pred = feature_scaler.inverse_transform(
            y_pred.detach().cpu().numpy())
        y = feature_scaler.inverse_transform(y.detach().cpu().numpy())
    except:
        y_pred = feature_scaler.inverse_transform(y_pred)
        y = feature_scaler.inverse_transform(y)

    def WRMSE(y_pred, y_true):
        errors = y_true - y_pred
        squared_errors = errors**2
        weighted_squared_errors = y_true * squared_errors
        sum_weighted_squared_errors = np.sum(weighted_squared_errors)
        sum_weights = np.sum(y_true)
        return np.sqrt(sum_weighted_squared_errors / (sum_weights + 0.001))

    wrmse = WRMSE(y_pred, y)
    if np.isnan(wrmse):
        print('error')
    mse = ((y_pred - y)**2).mean()
    # RMSE
    rmse = np.sqrt(mse)
    # MAE
    mae = np.abs(y_pred - y).mean()
    return {
        '{}/rmse'.format(name): rmse,
        '{}/wrmse'.format(name): wrmse,
    }


# HA 方法(0)
class HApredicter:
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.hparams = hparams

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    def train_test(self):
        # 导入数据
        data = load_nyc_data(path=self.hparams.data_dir, params=self.hparams)
        self.data = data
        train_x, val_x, test_x = data['train']['x'], data['val']['x'], data[
            'test']['x']
        x = np.concatenate([train_x, val_x, test_x], axis=0)
        train_y, val_y, test_y = data['train']['y'], data['val']['y'], data[
            'test']['y']
        y = np.concatenate([train_y, val_y, test_y], axis=0)
        # x的时间步的平均是要预测y时间步的第一个值
        y_true = y[:, 0:1, :, :]
        y_pred = np.mean(x, axis=1, keepdims=True)
        metrics = unscaled_metrics(y_pred, y_true, 'test',
                                   self.data['feature_scaler'])
        y_true = self.data['feature_scaler'].inverse_transform(y_true)
        y_pred = self.data['feature_scaler'].inverse_transform(y_pred)
        os.makedirs(self.hparams.output_dir + '/' + self.hparams.model_name,
                    exist_ok=True)
        # 保存指标与预测结果
        with open(
                self.hparams.output_dir + '/' + self.hparams.model_name + '/' +
                'prediction_scores.txt', 'a') as f:
            f.write('历史观测长度:{0},预测长度:{1}'.format(self.hparams.obs_len,
                                                 self.hparams.pred_len))
            f.write('\n')
            for key, value in metrics.items():
                f.write('{0}:{1}'.format(key, value))
            f.write('\n\n\n')
        np.save(
            self.hparams.output_dir + '/' + self.hparams.model_name + '/' +
            'od_prediction.npy', y_pred)
        np.save(
            self.hparams.output_dir + '/' + self.hparams.model_name + '/' +
            'od_groundtruth.npy', y_true)

        return None


# GRUSeq2Seq(65k)
class ODGRUSeq2Seq(LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.hparams = hparams
        self.epoch = 0
        self.base_model = None
        self.base_model_class = getattr(base_models,
                                        self.hparams.base_model_name)
        self.save_hyperparameters()
        self.setup(None)

    def forward(self, x):
        if self.hparams.base_model_name == 'GRUSeq2SeqWithGraphNet':
            return self.base_model(x, batches_seen=self.global_step)
        return self.base_model(x, self.global_step)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--hetero_graph', action='store_true')
        return parser

    def prepare_data(self):
        pass

    def setup(self, step):
        if self.base_model is not None:
            return
        data = load_nyc_data(path=self.hparams.data_dir, params=self.hparams)
        self.data = data
        input_size = self.data['train']['x'].shape[-1] + self.data['train'][
            'x_attr'].shape[-1]
        output_size = self.data['train']['y'].shape[-1]

        self.base_model = self.base_model_class(input_size=input_size,
                                                output_size=output_size,
                                                **self.hparams)
        self.datasets = {}
        for name in ['train', 'val', 'test']:
            if self.hparams.hetero_graph:
                datalist = [
                    Data(x=self.data[name]['x'][t].permute(1, 0, 2),
                         y=self.data[name]['y'][t].permute(1, 0, 2),
                         x_attr=self.data[name]['x_attr'][t].permute(1, 0, 2),
                         y_attr=self.data[name]['y_attr'][t].permute(1, 0, 2),
                         edge_index=self.data[name]['edge_index'][t],
                         edge_attr=self.data[name]['edge_attr'][t])
                    for t in range(self.data[name]['x'].shape[0])
                ]
                self.datasets[name] = {'dataset': datalist}
            else:
                self.datasets[name] = {
                    'dataset':
                    TensorDataset(self.data[name]['x'], self.data[name]['y'],
                                  self.data[name]['x_attr'],
                                  self.data[name]['y_attr']),
                    'graph':
                    dict(edge_index=self.data[name]['edge_index'],
                         edge_attr=self.data[name]['edge_attr'])
                }

    def train_dataloader(self):
        return DataLoader(self.datasets['train']['dataset'],
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.datasets['val']['dataset'],
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.datasets['test']['dataset'],
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams['lr'],
                                weight_decay=self.hparams.weight_decay)

    def training_step(self, batch, batch_idx):
        if self.hparams.hetero_graph:
            x, y, x_attr, y_attr = batch['x'], batch['y'], batch[
                'x_attr'], batch['y_attr']
            data = batch
        else:
            x, y, x_attr, y_attr = batch
            graph = self.datasets['train']['graph']
            data = dict(x=x,
                        x_attr=x_attr,
                        y=y,
                        y_attr=y_attr,
                        edge_index=graph['edge_index'].to(x.device),
                        edge_attr=graph['edge_attr'].to(x.device))
        y_pred = self(data)
        loss = nn.MSELoss()(y_pred, y)

        log = {'train/loss': loss, 'num': y_pred.shape[0]}
        # log.update(**unscaled_metrics(y_pred, y, self.data['feature_scaler'],
        #                               'train'))
        log.update(**unscaled_metrics(y_pred, y, 'train',
                                      self.data['feature_scaler']))
        return {'loss': loss, 'progress_bar': log, 'log': log}

    def training_epoch_end(self, outputs):
        # average all statistics (weighted by sample counts)
        log = {}
        if isinstance(outputs[0], tuple):
            for output in outputs:
                for k in output[0]['log']:
                    if k not in log:
                        log[k] = 0
                    if k == 'num':
                        log[k] += output[0]['log'][k]
                    else:
                        log[k] += (output[0]['log'][k] *
                                   output[0]['log']['num'])
            for k in log:
                if k != 'num':
                    log[k] = log[k] / log['num']
            log.pop('num')
            with open(
                    self.hparams.output_dir + '/' +
                    self.hparams['model_name'] + '/' + 'prediction_scores.txt',
                    'a') as f:
                f.write('历史观测长度:{0},预测长度:{1}'.format(self.hparams.obs_len,
                                                     self.hparams.pred_len))
                f.write('\n')
                for key, value in log.items():
                    f.write('{0}:{1}'.format(key, value))
                    f.write('\n')
                f.write('\n\n\n')
            od_predictions = []
            od_groundtruths = []
            for pred in outputs:
                od_predictions.append(pred[1]['prediction'])
                od_groundtruths.append(pred[1]['prediction'])
            od_predictions = np.concatenate(od_predictions)
            od_groundtruths = np.concatenate(od_groundtruths)
            np.save(
                self.hparams.output_dir + '/' + self.hparams['model_name'] +
                '/' + 'od_prediction.npy', od_predictions)
            np.save(
                self.hparams.output_dir + '/' + self.hparams['model_name'] +
                '/' + 'od_groundtruth.npy', od_groundtruths)

        else:
            for output in outputs:
                for k in output['log']:
                    if k not in log:
                        log[k] = 0
                    if k == 'num':
                        log[k] += output['log'][k]
                    else:
                        log[k] += (output['log'][k] * output['log']['num'])
            for k in log:
                if k != 'num':
                    log[k] = log[k] / log['num']
            log.pop('num')

        if self.hparams['wandb']:
            wandb.log(log)
        return {'log': log, 'progress_bar': log}

    def validation_step(self, batch, batch_idx):
        if self.hparams.hetero_graph:
            x, y, x_attr, y_attr = batch['x'], batch['y'], batch[
                'x_attr'], batch['y_attr']
            data = batch
        else:
            x, y, x_attr, y_attr = batch
            graph = self.datasets['val']['graph']
            data = dict(x=x,
                        x_attr=x_attr,
                        y=y,
                        y_attr=y_attr,
                        edge_index=graph['edge_index'].to(x.device),
                        edge_attr=graph['edge_attr'].to(x.device))
        y_pred = self(data)
        loss = nn.MSELoss()(y_pred, y)

        log = {'val/loss': loss, 'num': y_pred.shape[0]}
        log.update(
            **unscaled_metrics(y_pred, y, 'val', self.data['feature_scaler']))

        return {'loss': loss, 'progress_bar': log, 'log': log}

    def validation_epoch_end(self, outputs):
        return self.training_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        # if 'od_predictions' not in locals():
        #     od_predictions = []
        if self.hparams.hetero_graph:
            x, y, x_attr, y_attr = batch['x'], batch['y'], batch[
                'x_attr'], batch['y_attr']
            data = batch
        else:
            x, y, x_attr, y_attr = batch
            graph = self.datasets['test']['graph']
            data = dict(x=x,
                        x_attr=x_attr,
                        y=y,
                        y_attr=y_attr,
                        edge_index=graph['edge_index'].to(x.device),
                        edge_attr=graph['edge_attr'].to(x.device))
        y_pred = self(data)
        loss = nn.MSELoss()(y_pred, y)
        log = {'test/loss': loss, 'num': y_pred.shape[0]}
        log.update(
            **unscaled_metrics(y_pred, y, 'test', self.data['feature_scaler']))

        od_prediction = self.data['feature_scaler'].inverse_transform(
            y_pred).detach().cpu().numpy()
        od_groundtruth = self.data['feature_scaler'].inverse_transform(
            y).cpu().numpy()

        os.makedirs(self.hparams.output_dir + '/' + self.hparams['model_name'],
                    exist_ok=True)

        return {
            'loss': loss,
            'progress_bar': log,
            'log': log,
            'mode': 'test'
        }, {
            'prediction': od_prediction,
            'groundtruth': od_groundtruth
        }

    def test_epoch_end(self, outputs):
        return self.training_epoch_end(outputs)
