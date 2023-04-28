# Two Stage Dynamic OD predictor
from argparse import ArgumentParser
from copy import deepcopy
import os
import numpy as np
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader, Data
from collections import defaultdict

import wandb
from NYCDataset.datasets import load_nyc_data, ODsplit_nyc_data
from torch.utils.data import TensorDataset

import models.base_models as base_models
from .NYC_Centralized import unscaled_metrics
from models.base_models import GeoSemNodeEm


class ODFedNodePredictorClient(nn.Module):
    def __init__(self, base_model_name, optimizer_name, train_dataset,
                 val_dataset, test_dataset, sync_every_n_epoch, lr,
                 weight_decay, batch_size, client_device, start_global_step,
                 od_max, od_min, *args, **kwargs):
        super().__init__()
        self.base_model_name = base_model_name
        self.optimizer_name = optimizer_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.sync_every_n_epoch = sync_every_n_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.base_model_kwargs = kwargs
        self.device = client_device
        self.od_max = od_max
        self.od_min = od_min

        self.base_model_class = getattr(base_models, self.base_model_name)
        self.init_base_model(None)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=False,
                                         pin_memory=True)
        self.val_dataloader = self.train_dataloader
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          pin_memory=True)
        self.test_dataloader = self.train_dataloader
        self.global_step = start_global_step

    def forward(self, x, server_graph_encoding):
        return self.base_model(x,
                               self.global_step,
                               server_graph_encoding=server_graph_encoding)

    def init_base_model(self, state_dict):
        self.base_model = self.base_model_class(**self.base_model_kwargs).to(
            self.device)
        if state_dict is not None:
            self.base_model.load_state_dict(state_dict)
        self.optimizer = getattr(torch.optim, self.optimizer_name)(
            self.base_model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay)

    def local_train(self, state_dict_to_load):
        if state_dict_to_load is not None:
            self.base_model.load_state_dict(state_dict_to_load)
        self.train()
        with torch.enable_grad():
            for epoch_i in range(self.sync_every_n_epoch):
                num_samples = 0
                epoch_log = defaultdict(lambda: 0.0)
                for batch in self.train_dataloader:
                    x, y, x_attr, y_attr, server_graph_encoding = batch
                    server_graph_encoding = server_graph_encoding.permute(
                        1, 0, 2, 3)
                    x = x.to(self.device) if (x is not None) else None
                    y = y.to(self.device) if (y is not None) else None
                    x_attr = x_attr.to(self.device) if (x_attr
                                                        is not None) else None
                    y_attr = y_attr.to(self.device) if (y_attr
                                                        is not None) else None
                    server_graph_encoding = server_graph_encoding.to(
                        self.device)
                    data = dict(x=x, x_attr=x_attr, y=y, y_attr=y_attr)
                    # y_pred_val, y_pred_prob = self(data, server_graph_encoding)
                    y_pred = self(data, server_graph_encoding)
                    # loss = MyLoss()(y_pred_val, y_pred_prob, y)
                    loss = nn.MSELoss()(y_pred, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    num_samples += x.shape[0]
                    metrics = unscaled_metrics(y_pred, y, 'train', self.od_max,
                                               self.od_min)
                    epoch_log['train/weight_loss'] += loss.detach(
                    ) * x.shape[0]
                    for k in metrics:
                        epoch_log[k] += metrics[k] * x.shape[0]
                    self.global_step += 1
                for k in epoch_log:
                    epoch_log[k] /= num_samples
                    epoch_log[k] = epoch_log[k].cpu()
        # self.cpu()
        state_dict = self.base_model.to('cpu').state_dict()
        epoch_log['num_samples'] = num_samples
        epoch_log['global_step'] = self.global_step
        epoch_log = dict(**epoch_log)

        return {'state_dict': state_dict, 'log': epoch_log}

    def local_eval(self, dataloader, name, state_dict_to_load):
        if state_dict_to_load is not None:
            self.base_model.load_state_dict(state_dict_to_load)
        self.eval()
        if name == 'test':
            od_prediction = []
        with torch.no_grad():
            num_samples = 0
            epoch_log = defaultdict(lambda: 0.0)
            for batch in dataloader:
                x, y, x_attr, y_attr, server_graph_encoding = batch
                server_graph_encoding = server_graph_encoding.permute(
                    1, 0, 2, 3)
                x = x.to(self.device) if (x is not None) else None
                y = y.to(self.device) if (y is not None) else None
                x_attr = x_attr.to(self.device) if (x_attr
                                                    is not None) else None
                y_attr = y_attr.to(self.device) if (y_attr
                                                    is not None) else None
                server_graph_encoding = server_graph_encoding.to(self.device)
                data = dict(x=x, x_attr=x_attr, y=y, y_attr=y_attr)
                y_pred = self(data, server_graph_encoding)
                if name == 'test':
                    od_prediction.append(np.exp(y_pred.detach().cpu()) - 1.0)
                loss = nn.MSELoss()(y_pred, y)
                num_samples += x.shape[0]
                metrics = unscaled_metrics(y_pred, y, name, self.od_max,
                                           self.od_min)
                epoch_log['{}/weighted_loss'.format(
                    name)] += loss.detach() * x.shape[0]
                for k in metrics:
                    epoch_log[k] += metrics[k] * x.shape[0]
            if name == 'test':
                od_prediction = np.concatenate(od_prediction, axis=0)

            for k in epoch_log:
                epoch_log[k] /= num_samples
                epoch_log[k] = epoch_log[k].cpu()
        # self.cpu()
        epoch_log['num_samples'] = num_samples
        epoch_log = dict(**epoch_log)
        if name == 'test':
            return {'log': epoch_log, 'od_prediction': od_prediction}
        return {'log': epoch_log}

    def local_validation(self, state_dict_to_load):
        return self.local_eval(self.val_dataloader, 'val', state_dict_to_load)

    def local_test(self, state_dict_to_load):
        return self.local_eval(self.test_dataloader, 'test',
                               state_dict_to_load)

    @staticmethod
    def client_local_execute(device, state_dict_to_load, order, hparams_list):
        if (type(device) is str) and (device.startswith('cuda:')):
            cuda_id = int(device.split(':')[1])
            device = torch.device('cuda:{}'.format(cuda_id))
            torch.cuda.set_device(device)
        elif type(device) is torch.device:
            torch.cuda.set_device(device)
        else:
            device = torch.device('cpu')
        res_list = []
        if type(hparams_list) == 'list':
            for hparams in hparams_list:
                client = ODFedNodePredictorClient(client_device=device,
                                                  **hparams)
                if order == 'train':
                    res = client.local_train(state_dict_to_load)
                elif order == 'val':
                    res = client.local_validation(state_dict_to_load)
                elif order == 'test':
                    res = client.local_test(state_dict_to_load)
                else:
                    del client
                    raise NotImplementedError()
                del client
                res_list.append(res)
        else:
            client = ODFedNodePredictorClient(client_device=device,
                                              **hparams_list)
            if order == 'train':
                res = client.local_train(state_dict_to_load)
            elif order == 'val':
                res = client.local_validation(state_dict_to_load)
            elif order == 'test':
                res = client.local_test(state_dict_to_load)
            else:
                del client
                raise NotImplementedError()
            del client
            res_list.append(res)
        return res_list


class ODFedNodePredictorServer(LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.hparams = hparams
        self.base_model = None
        self.setup(None)

    def forward(self, x):
        raise NotImplementedError()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--server_batch_size', type=int, default=8)
        parser.add_argument('--sync_every_n_epoch', type=int, default=5)
        parser.add_argument('--server_epoch', type=int, default=1)
        parser.add_argument('--mp_worker_num', type=int, default=1)
        parser.add_argument('--server_gn_layer_num', type=int, default=2)
        return parser

    def prepare_data(self):
        pass

    def setup(self, step):
        # must avoid repeated init!!! otherwise loaded model weights will be re-initialized!!!
        if self.base_model is not None:
            return
        data = ODsplit_nyc_data(path=self.hparams.data_dir,
                                data_name=self.hparams.data_name)
        self.data = data
        # Each node (client) has its own model and optimizer
        # Assigning data, model and optimizer for each client
        num_clients = data['train']['x'].shape[2]
        input_size = self.data['train']['x'].shape[-1] + self.data['train'][
            'x_attr'].shape[-1]
        output_size = self.data['train']['y'].shape[-1]
        client_params_list = []
        for client_i in range(num_clients):
            client_datasets = {}
            for name in ['train', 'val', 'test']:
                client_datasets[name] = TensorDataset(
                    data[name]['x'][:, :, client_i:client_i + 1, :],
                    data[name]['y'][:, client_i:client_i + 1, :],
                    data[name]['x_attr'][:, :, client_i:client_i + 1, :],
                    data[name]['y_attr'][:, client_i:client_i + 1, :],
                    torch.zeros(1, data[name]['x'].shape[0],
                                self.hparams.gru_num_layers,
                                self.hparams.hidden_size).float().permute(
                                    1, 0, 2,
                                    3)  # default_server_graph_encoding
                )
            client_params = {}
            client_params.update(optimizer_name='Adam',
                                 od_max=data['od_max'],
                                 od_min=data['od_min'],
                                 train_dataset=client_datasets['train'],
                                 val_dataset=client_datasets['val'],
                                 test_dataset=client_datasets['test'],
                                 input_size=input_size,
                                 output_size=output_size,
                                 start_global_step=0,
                                 **self.hparams)
            client_params_list.append(client_params)
        self.client_params_list = client_params_list

        self.base_model = getattr(base_models, self.hparams.base_model_name)(
            input_size=input_size, output_size=output_size, **self.hparams)
        self.gcn = GeoSemNodeEm(node_input_size=self.hparams.hidden_size,
                                hidden_size=256,
                                node_output_size=self.hparams.hidden_size,
                                gn_layer_num=self.hparams.server_gn_layer_num,
                                activation='ReLU',
                                dropout=self.hparams.dropout)
        self.server_optimizer = getattr(torch.optim, 'Adam')(
            self.gcn.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay)
        self.server_datasets = {}
        for name in ['train', 'val', 'test']:
            self.server_datasets[name] = TensorDataset(
                self.data[name]['x'], self.data[name]['y'],
                self.data[name]['x_attr'], self.data[name]['y_attr'])

    def _train_server_gcn_with_agg_clients(self, device):
        # here we assume all clients are aggregated! Simulate running on clients with the aggregated copy on server
        # this only works when (1) clients are aggregated and (2) no optimization on client models
        self.base_model.to(device)
        self.gcn.to(device)
        server_train_dataloader = DataLoader(
            self.server_datasets['train'],
            batch_size=self.hparams.server_batch_size,
            shuffle=True,
            pin_memory=True,
        )
        updated_graph_encoding = None
        global_step = self.client_params_list[0]['start_global_step']
        with torch.enable_grad():
            self.base_model.train()
            self.gcn.train()
            for epoch_i in range(self.hparams.server_epoch + 1):
                updated_graph_encoding = []
                if epoch_i == self.hparams.server_epoch:
                    server_train_dataloader = DataLoader(
                        self.server_datasets['train'],
                        batch_size=self.hparams.server_batch_size,
                        shuffle=False,
                        pin_memory=True,
                    )
                for batch in server_train_dataloader:
                    x, y, x_attr, y_attr = batch
                    x = x.to(device) if (x is not None) else None
                    y = y.to(device) if (y is not None) else None
                    x_attr = x_attr.to(device) if (x_attr
                                                   is not None) else None
                    y_attr = y_attr.to(device) if (y_attr
                                                   is not None) else None

                    if 'selected' in self.data['train']:
                        train_mask = self.data['train']['selected'].flatten()
                        x, y, x_attr, y_attr = x[:, :,
                                                 train_mask, :], y[:, :,
                                                                   train_mask, :], x_attr[:, :,
                                                                                          train_mask, :], y_attr[:, :,
                                                                                                                 train_mask, :]

                    data = dict(x=x, x_attr=x_attr, y=y, y_attr=y_attr)
                    h_encode = self.base_model.forward_encoder(
                        data)  # L x (B x N) x F
                    batch_num, node_num = data['x'].shape[0], data['x'].shape[
                        2]
                    graph_encoding = h_encode.view(
                        h_encode.shape[0], batch_num, node_num,
                        h_encode.shape[2]).permute(2, 1, 0, 3)  # N x B x L x F
                    graph_encoding = self.gcn(
                        Data(x=graph_encoding,
                             edge_index=self.data['train']['edge_index'].to(
                                 graph_encoding.device),
                             edge_attr=self.data['train']['edge_attr'].
                             unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(
                                 graph_encoding.device)))
                    # N x B x L x F
                    if epoch_i == self.hparams.server_epoch:
                        updated_graph_encoding.append(
                            graph_encoding.detach().clone().cpu())
                    else:
                        y_pred = self.base_model.forward_decoder(
                            data,
                            h_encode,
                            batches_seen=global_step,
                            return_encoding=False,
                            server_graph_encoding=graph_encoding)
                        loss = nn.MSELoss()(y_pred, y)
                        self.server_optimizer.zero_grad()
                        loss.backward()
                        self.server_optimizer.step()
                        global_step += 1
        # update global step for all clients
        for client_params in self.client_params_list:
            client_params.update(start_global_step=global_step)
        # update server_graph_encoding
        updated_graph_encoding = torch.cat(updated_graph_encoding,
                                           dim=1)  # N x B x L x F
        sel_client_i = 0
        for client_i, client_params in enumerate(self.client_params_list):
            if 'selected' in self.data['train']:
                if self.data['train']['selected'][client_i, 0].item() is False:
                    continue
            client_params.update(train_dataset=TensorDataset(
                self.data['train']['x'][:, :, client_i:client_i + 1, :],
                self.data['train']['y'][:, client_i:client_i +
                                        1, :], self.data['train']['x_attr']
                [:, :, client_i:client_i +
                 1, :], self.data['train']['y_attr'][:,
                                                     client_i:client_i + 1, :],
                updated_graph_encoding[sel_client_i:sel_client_i +
                                       1, :, :, :].permute(1, 0, 2, 3)))
            sel_client_i += 1

    def _eval_server_gcn_with_agg_clients(self, name, device):
        assert name in ['val', 'test']
        self.base_model.to(device)
        self.gcn.to(device)
        server_dataloader = DataLoader(
            self.server_datasets[name],
            batch_size=self.hparams.server_batch_size,
            shuffle=False,
            pin_memory=True,
        )
        updated_graph_encoding = []
        with torch.no_grad():
            self.base_model.eval()
            self.gcn.eval()
            for batch in server_dataloader:
                x, y, x_attr, y_attr = batch
                x = x.to(device) if (x is not None) else None
                y = y.to(device) if (y is not None) else None
                x_attr = x_attr.to(device) if (x_attr is not None) else None
                y_attr = y_attr.to(device) if (y_attr is not None) else None
                data = dict(x=x, x_attr=x_attr, y=y, y_attr=y_attr)
                # client利用本地数据提取时间特征
                h_encode = self.base_model.forward_encoder(
                    data)  # L x (B x N) x F
                batch_num, node_num = data['x'].shape[0], data['x'].shape[2]
                graph_encoding = h_encode.view(h_encode.shape[0], batch_num,
                                               node_num,
                                               h_encode.shape[2]).permute(
                                                   2, 1, 0, 3)  # N x B x L x F
                #上面是hc，i 下面是hs，i
                graph_encoding = self.gcn(
                    Data(x=graph_encoding,
                         edge_index=self.data[name]['edge_index'].to(
                             graph_encoding.device),
                         edge_attr=self.data[name]['edge_attr'].unsqueeze(
                             -1).unsqueeze(-1).unsqueeze(-1).to(
                                 graph_encoding.device)))  # N x B x L x F
                updated_graph_encoding.append(
                    graph_encoding.detach().clone().cpu())
        # update server_graph_encoding
        updated_graph_encoding = torch.cat(updated_graph_encoding,
                                           dim=1)  # N x B x L x F
        for client_i, client_params in enumerate(self.client_params_list):
            keyname = '{}_dataset'.format(name)
            client_params.update({
                keyname:
                TensorDataset(
                    self.data[name]['x'][:, :, client_i:client_i + 1, :],
                    self.data[name]['y'][:, client_i:client_i + 1, :],
                    self.data[name]['x_attr'][:, :, client_i:client_i + 1, :],
                    self.data[name]['y_attr'][:, client_i:client_i + 1, :],
                    updated_graph_encoding[client_i:client_i +
                                           1, :, :, :].permute(1, 0, 2, 3))
                # B x N x L x F
            })

    def train_dataloader(self):
        # return a fake dataloader for running the loop
        return DataLoader([
            0,
        ])

    def val_dataloader(self):
        return DataLoader([
            0,
        ])

    def test_dataloader(self):
        return DataLoader([
            0,
        ])

    def configure_optimizers(self):
        return None

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        return None

    def training_step(self, batch, batch_idx):
        # 1. train locally and collect uploaded local train results
        local_train_results = []
        server_device = next(self.gcn.parameters()).device
        self.base_model.to('cpu')
        self.gcn.to('cpu')
        if self.hparams.mp_worker_num <= 1:
            for client_i, client_params in enumerate(self.client_params_list):
                if 'selected' in self.data['train']:
                    if self.data['train']['selected'][client_i,
                                                      0].item() is False:
                        continue
                local_train_result = ODFedNodePredictorClient.client_local_execute(
                    batch.device, deepcopy(self.base_model.state_dict()),
                    'train', client_params)
                local_train_results.append(local_train_result)
        else:
            pool = mp.Pool(self.hparams.mp_worker_num)
            temp_client_params_list = []
            for client_i, client_params in enumerate(self.client_params_list):
                if 'selected' in self.data['train']:
                    if self.data['train']['selected'][client_i,
                                                      0].item() is False:
                        continue
                temp_client_params_list.append(client_params)
            for worker_i, client_params in enumerate(
                    np.array_split(temp_client_params_list,
                                   self.hparams.mp_worker_num)):
                # gpu_list = list(map(str, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
                gpu_list = list(
                    range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
                device_name = 'cuda:{}'.format(gpu_list[worker_i %
                                                        len(gpu_list)])
                # device_name = 'cpu'
                local_train_results.append(
                    pool.apply_async(
                        SplitFedNodePredictorClient.client_local_execute,
                        args=(device_name,
                              deepcopy(self.base_model.state_dict()), 'train',
                              client_params)))
            pool.close()
            pool.join()
            local_train_results = list(
                map(lambda x: x.get(), local_train_results))
            local_train_results = list(
                itertools.chain.from_iterable(local_train_results))
        # update global steps for all clients
        for ltr, client_params in zip(local_train_results,
                                      self.client_params_list):
            if self.hparams.mp_worker_num <= 1:
                client_params.update(
                    start_global_step=ltr[0]['log']['global_step'])
            else:
                client_params.update(
                    start_global_step=ltr['log']['global_step'])
        # 2. aggregate (optional? kept here to save memory, otherwise need to store 1 model for each node)
        agg_local_train_results = self.aggregate_local_train_results(
            local_train_results)
        # 2.1. update aggregated weights
        if agg_local_train_results['state_dict'] is not None:
            self.base_model.load_state_dict(
                agg_local_train_results['state_dict'])
        # TODO: 3. train GNN on server in split learning way (optimize server-side params only)
        # client_local_execute, return_encoding
        # run decoding on all clients, run backward on all clients
        # run backward on server-side GNN and optimize GNN
        # TODO: 4. run forward on updated GNN to renew server_graph_encoding
        self._train_server_gcn_with_agg_clients(server_device)
        agg_log = agg_local_train_results['log']
        log = agg_log
        return {
            'loss': torch.tensor(0).float(),
            'progress_bar': log,
            'log': log
        }

    def aggregate_local_train_results(self, local_train_results):
        if self.hparams.mp_worker_num <= 1:
            return {
                'state_dict':
                self.aggregate_local_train_state_dicts(
                    [ltr[0]['state_dict'] for ltr in local_train_results]),
                'log':
                self.aggregate_local_logs(
                    [ltr[0]['log'] for ltr in local_train_results])
            }
        else:
            return {
                'state_dict':
                self.aggregate_local_train_state_dicts(
                    [ltr['state_dict'] for ltr in local_train_results]),
                'log':
                self.aggregate_local_logs(
                    [ltr['log'] for ltr in local_train_results])
            }

    def aggregate_local_train_state_dicts(self, local_train_state_dicts):
        raise NotImplementedError()

    def aggregate_local_logs(self, local_logs, selected=None):
        agg_log = deepcopy(local_logs[0])
        for k in agg_log:
            agg_log[k] = 0
            for local_log_idx, local_log in enumerate(local_logs):
                if k == 'num_samples':
                    agg_log[k] += local_log[k]
                else:
                    agg_log[k] += local_log[k] * local_log['num_samples']
        for k in agg_log:
            if k != 'num_samples':
                agg_log[k] /= agg_log['num_samples']
        return agg_log

    def training_epoch_end(self, outputs):
        # already averaged!
        log = outputs[0]['log']
        log.pop('num_samples')
        if self.hparams['wandb']:
            wandb.log(log)
        return {'log': log, 'progress_bar': log}

    def validation_step(self, batch, batch_idx):
        server_device = next(self.gcn.parameters()).device
        self._eval_server_gcn_with_agg_clients('val', server_device)
        # 1. vaidate locally and collect uploaded local train results
        local_val_results = []
        self.base_model.to('cpu')
        self.gcn.to('cpu')
        if self.hparams.mp_worker_num <= 1:
            for client_i, client_params in enumerate(self.client_params_list):
                local_val_result = ODFedNodePredictorClient.client_local_execute(
                    batch.device, deepcopy(self.base_model.state_dict()),
                    'val', client_params)
                local_val_results.append(local_val_result)
        else:
            pool = mp.Pool(self.hparams.mp_worker_num)
            for worker_i, client_params in enumerate(
                    np.array_split(self.client_params_list,
                                   self.hparams.mp_worker_num)):
                # gpu_list = list(map(str, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
                gpu_list = list(
                    range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
                device_name = 'cuda:{}'.format(gpu_list[worker_i %
                                                        len(gpu_list)])
                # device_name = 'cpu'
                local_val_results.append(
                    pool.apply_async(
                        SplitFedNodePredictorClient.client_local_execute,
                        args=(device_name,
                              deepcopy(self.base_model.state_dict()), 'val',
                              client_params)))
            pool.close()
            pool.join()
            local_val_results = list(map(lambda x: x.get(), local_val_results))
            local_val_results = list(
                itertools.chain.from_iterable(local_val_results))
        self.base_model.to(server_device)
        self.gcn.to(server_device)
        # 2. aggregate
        if self.hparams.mp_worker_num <= 1:
            log = self.aggregate_local_logs(
                [x[0]['log'] for x in local_val_results])
        else:
            log = self.aggregate_local_logs(
                [x['log'] for x in local_val_results])
        return {'progress_bar': log, 'log': log}

    def validation_epoch_end(self, outputs):
        return self.training_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        server_device = next(self.gcn.parameters()).device
        self._eval_server_gcn_with_agg_clients('test', server_device)
        # 1. vaidate locally and collect uploaded local train results
        local_val_results = []
        self.base_model.to('cpu')
        self.gcn.to('cpu')
        od_prediction = []
        if self.hparams.mp_worker_num <= 1:
            for client_i, client_params in enumerate(self.client_params_list):
                local_val_result = ODFedNodePredictorClient.client_local_execute(
                    batch.device, deepcopy(self.base_model.state_dict()),
                    'test', client_params)
                y_pred = local_val_result[0].pop('od_prediction')
                od_prediction.append(y_pred)
                local_val_results.append(local_val_result)
        else:
            pool = mp.Pool(self.hparams.mp_worker_num)
            for worker_i, client_params in enumerate(
                    np.array_split(self.client_params_list,
                                   self.hparams.mp_worker_num)):
                # gpu_list = list(map(str, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
                gpu_list = list(
                    range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
                device_name = 'cuda:{}'.format(gpu_list[worker_i %
                                                        len(gpu_list)])
                # device_name = 'cpu'
                local_val_results.append(
                    pool.apply_async(
                        SplitFedNodePredictorClient.client_local_execute,
                        args=(device_name,
                              deepcopy(self.base_model.state_dict()), 'test',
                              client_params)))
            pool.close()
            pool.join()
            local_val_results = list(map(lambda x: x.get(), local_val_results))
            local_val_results = list(
                itertools.chain.from_iterable(local_val_results))
        self.base_model.to(server_device)
        self.gcn.to(server_device)
        od_prediction = np.concatenate(od_prediction, axis=2)
        if not os.path.exists('./output'):
            os.makedirs('./output')
        np.save(self.hparams['data_name'] + '_od_prediction.npy',
                od_prediction)
        # 2. aggregate
        # separate seen and unseen nodes if necessary
        if 'selected' in self.data['train']:
            log = self.aggregate_local_logs(
                [x['log'] for x in local_val_results],
                self.data['train']['selected'])
        else:
            if self.hparams.mp_worker_num <= 1:
                log = self.aggregate_local_logs(
                    [x[0]['log'] for x in local_val_results])
            else:
                log = self.aggregate_local_logs(
                    [x['log'] for x in local_val_results])
        return {'progress_bar': log, 'log': log}

    def test_epoch_end(self, outputs):
        return self.training_epoch_end(outputs)


class ODFedAvgNodePredictor(ODFedNodePredictorServer):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams, *args, **kwargs)

    def aggregate_local_train_state_dicts(self, local_train_state_dicts):
        agg_state_dict = {}
        for k in local_train_state_dicts[0]:
            agg_state_dict[k] = 0
            for ltsd in local_train_state_dicts:
                agg_state_dict[k] += ltsd[k]
            agg_state_dict[k] /= len(local_train_state_dicts)
        return agg_state_dict