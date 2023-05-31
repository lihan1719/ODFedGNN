import os
import sys
import shutil
import argparse
import numpy as np
from datetime import datetime
from torch import nn, optim
import torch
import Utils, DCRNN


class WMSELoss(nn.Module):
    def __init__(self, scaler, non_zero_weight=10):
        super(WMSELoss, self).__init__()
        self.non_zero_weight = non_zero_weight
        self.scaler = scaler

    def forward(self, y_pred, y_true):
        # 计算绝对误差
        squared_errors = (y_true - y_pred)**2
        # abs_errors = torch.abs(y_true - y_pred)

        # 计算加权系数
        y = self.scaler.inverse_transform(y_true)
        weights = torch.where(y != 0, y, torch.tensor(1, device=y_true.device))

        # 计算WMAE损失函数
        wmse_loss = torch.sum(weights * squared_errors) / torch.sum(weights)
        # wmae_loss = torch.sum(weights * abs_errors) / torch.sum(weights)

        # 计算MAE损失函数
        # mae_loss = torch.mean(abs_errors)
        mse_loss = torch.mean(squared_errors)
        # 计算总损失
        total_loss = mse_loss + wmse_loss
        return total_loss


class ModelTrainer(object):
    def __init__(self, params: dict, data: dict, data_container):
        self.params = params
        self.data_container = data_container
        self.get_static_graph(
            graph=data['adj'])  # initialize static graphs and K values
        self.model = self.get_model().to(params['GPU'])
        self.criterion = self.get_loss()
        self.optimizer = self.get_optimizer()

    def get_static_graph(self, graph: np.array):
        self.K = self.get_support_K(self.params['kernel_type'],
                                    self.params['cheby_order'])
        self.G = self.preprocess_adj(graph, self.params['kernel_type'],
                                     self.params['cheby_order'])
        return

    @staticmethod
    def get_support_K(kernel_type, cheby_order):
        if kernel_type == 'localpool':
            assert cheby_order == 1
            K = 1
        elif (kernel_type == 'chebyshev') | (kernel_type
                                             == 'random_walk_diffusion'):
            K = cheby_order + 1
        elif kernel_type == 'dual_random_walk_diffusion':
            K = cheby_order * 2 + 1
        else:
            raise ValueError(
                'Invalid kernel_type. Must be one of '
                '[chebyshev, localpool, random_walk_diffusion, dual_random_walk_diffusion].'
            )
        return K

    def preprocess_adj(self, adj_mtx: np.array, kernel_type, cheby_order):
        adj_preprocessor = Utils.AdjProcessor(kernel_type, cheby_order)
        adj = torch.from_numpy(adj_mtx).float()
        adj = adj_preprocessor.process(adj)
        return adj.to(self.params['GPU'])  # G: (support_K, N, N)

    def get_model(self):
        if self.params['model'] == 'DCRNN':
            model = DCRNN.DCRNN(
                num_nodes=self.params['N'],
                input_dim=self.params['N'],
                hidden_dim=self.params['hidden_dim'],
                K=self.K,
                num_layers=self.params['nn_layers'],
                out_horizon=self.params['pred_len'],
                bias=True,
                activation=None)  # should not apply activation in each cell

        else:
            raise NotImplementedError('Invalid model name.')
        return model

    def get_loss(self):
        if self.params['loss'] == 'WMSE':
            criterion = WMSELoss(scaler=self.params['scaler'])
        elif self.params['loss'] == 'MSE':
            criterion = nn.MSELoss(reduction='mean')
        elif self.params['loss'] == 'MAE':
            criterion = nn.L1Loss(reduction='mean')
        elif self.params['loss'] == 'Huber':
            criterion = nn.SmoothL1Loss(reduction='mean')
        else:
            raise NotImplementedError('Invalid loss function.')
        return criterion

    def get_optimizer(self):
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(params=self.model.parameters(),
                                   lr=self.params['learn_rate'])
        else:
            raise NotImplementedError('Invalid optimizer name.')
        return optimizer

    def train(self, data_loader: dict, modes: list, early_stop_patience=10):
        checkpoint = {'epoch': 0, 'state_dict': self.model.state_dict()}
        val_loss = np.inf
        patience_count = early_stop_patience

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} model training begins:')
        for epoch in range(1, 1 + self.params['num_epochs']):
            starttime = datetime.now()
            running_loss = {mode: 0.0 for mode in modes}
            for mode in modes:
                if mode == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                step = 0
                for x_seq, y_true in data_loader[mode]:
                    with torch.set_grad_enabled(mode=(mode == 'train')):
                        if self.params['model'] == 'DCRNN':
                            y_pred = self.model(x_seq=x_seq.squeeze(dim=-1),
                                                P=self.G).unsqueeze(dim=-1)
                        else:
                            raise NotImplementedError('Invalid model name.')

                        loss = self.criterion(y_pred[..., 0], y_true)
                        if mode == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                    running_loss[mode] += loss * y_true.shape[
                        0]  # loss reduction='mean': batchwise average
                    step += y_true.shape[0]
                    torch.cuda.empty_cache()

                # epoch end: evaluate on validation set for early stopping
                if mode == 'validate':
                    epoch_val_loss = running_loss[mode] / step
                    if epoch_val_loss <= val_loss:
                        print(
                            f'Epoch {epoch}, validation loss drops from {val_loss:.5} to {epoch_val_loss:.5}. '
                            f'Update model checkpoint..',
                            f'used {(datetime.now() - starttime).seconds}s')
                        val_loss = epoch_val_loss
                        checkpoint.update(epoch=epoch,
                                          state_dict=self.model.state_dict())
                        torch.save(
                            checkpoint, self.params['output_dir'] +
                            f'/{self.params["model"]}_od.pkl')
                        patience_count = early_stop_patience
                    else:
                        print(
                            f'Epoch {epoch}, validation loss does not improve from {val_loss:.5}.',
                            f'used {(datetime.now() - starttime).seconds}s')
                        patience_count -= 1
                        if patience_count == 0:
                            print('\n',
                                  datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
                            print(
                                f'    Early stopping at epoch {epoch}. {self.params["model"]} model training ends.'
                            )
                            return
        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} model training ends.')
        torch.save(
            checkpoint,
            self.params['output_dir'] + f'/{self.params["model"]}_od.pkl')
        return

    def test(self, data_loader: dict, modes: list):
        trained_checkpoint = torch.load(self.params['output_dir'] +
                                        f'/{self.params["model"]}_od.pkl')
        self.model.load_state_dict(trained_checkpoint['state_dict'])
        self.model.eval()
        # 保存指标与预测结果
        f = open(
            self.params['output_dir'] + '/' + self.params['model'] +
            '_prediction_scores.txt', 'a')
        f.write('历史观测长度:{0},预测长度:{1}'.format(self.params['obs_len'],
                                             self.params['pred_len']))
        f.close()
        for mode in modes:
            print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            print(
                f'     {self.params["model"]} model testing on {mode} data begins:'
            )
            forecast, ground_truth = [], []
            for x_seq, y_true in data_loader[mode]:
                if self.params['model'] == 'DCRNN':
                    y_pred = self.model(x_seq=x_seq.squeeze(dim=-1),
                                        P=self.G).unsqueeze(dim=-1)
                else:
                    raise NotImplementedError('Invalid model name.')

                forecast.append(y_pred.cpu().detach().numpy())
                ground_truth.append(y_true.cpu().detach().numpy())

            forecast = np.concatenate(forecast, axis=0)
            ground_truth = np.concatenate(ground_truth, axis=0)
            forecast = self.params['scaler'].inverse_transform(forecast)
            ground_truth = self.params['scaler'].inverse_transform(
                ground_truth)
            if mode == 'test':
                np.save(
                    self.params['output_dir'] + '/' + self.params['model'] +
                    '_prediction.npy', forecast)
                np.save(
                    self.params['output_dir'] + '/' + self.params['model'] +
                    '_groundtruth.npy', ground_truth)

            # evaluate on metrics
            # MSE, RMSE, MAE, MAPE = self.evaluate(forecast[..., 0],
            #                                      ground_truth)
            RMSE, WRMSE = self.evaluate(forecast[..., 0], ground_truth)
            f = open(
                self.params['output_dir'] + '/' + self.params['model'] +
                '_prediction_scores.txt', 'a')
            f.write("%s, RMSE, WRMSE,  %.10f, %.10f\n" % (mode, RMSE, WRMSE))
            f.close()

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} model testing ends.')
        return

    @staticmethod
    def evaluate(y_pred: np.array, y_true: np.array, precision=10):
        def MSE(y_pred: np.array, y_true: np.array):
            return np.mean(np.square(y_pred - y_true))

        def RMSE(y_pred: np.array, y_true: np.array):
            return np.sqrt(MSE(y_pred, y_true))

        def WRMSE(y_pred: np.array, y_true: np.array):
            errors = y_true - y_pred
            squared_errors = errors**2
            weighted_squared_errors = y_true * squared_errors
            sum_weighted_squared_errors = np.sum(weighted_squared_errors)
            sum_weights = np.sum(y_true)
            return np.sqrt(sum_weighted_squared_errors / sum_weights)

        # def MAE(y_pred: np.array, y_true: np.array):
        #     return np.mean(np.abs(y_pred - y_true))

        # def MAPE(y_pred: np.array,
        #          y_true: np.array,
        #          epsilon=1e-0):  # avoid zero division
        #     return np.mean(np.abs(y_pred - y_true) / (y_true + epsilon))
        # print('MSE:', round(MSE(y_pred, y_true), precision))
        print('RMSE:', round(RMSE(y_pred, y_true), precision))
        print('WRMSE:', WRMSE(y_pred, y_true))
        # print('MAE:', round(MAE(y_pred, y_true), precision))
        # print('MAPE:', round(MAPE(y_pred, y_true) * 100, precision), '%')
        # return MSE(y_pred,
        #            y_true), RMSE(y_pred,
        #                          y_true), MAE(y_pred,
        #                                       y_true), MAPE(y_pred, y_true)
        return RMSE(y_pred, y_true), WRMSE(y_pred, y_true)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run OD Prediction with DCRNN')

    # command line arguments
    parser.add_argument('-GPU',
                        '--GPU',
                        type=str,
                        help='Specify GPU usage',
                        default='cuda:0')
    parser.add_argument('-in',
                        '--input_dir',
                        type=str,
                        default='../data/NYC_bike/processed_bike_data')
    parser.add_argument('-out', '--output_dir', type=str, default='output')
    parser.add_argument('-model',
                        '--model',
                        type=str,
                        help='Specify model',
                        choices=['DCRNN'],
                        default='DCRNN')
    parser.add_argument('-obs',
                        '--obs_len',
                        type=int,
                        help='Length of observation sequence',
                        default=4)
    parser.add_argument('-pred',
                        '--pred_len',
                        type=int,
                        help='Length of prediction sequence',
                        default=4)
    parser.add_argument('-split',
                        '--split_ratio',
                        type=float,
                        nargs='+',
                        help='训练、验证、测试天数'
                        ' Example: 22 4 5',
                        default=[7, 1, 2])
    parser.add_argument('-batch', '--batch_size', type=int, default=4)
    parser.add_argument('-hidden', '--hidden_dim', type=int, default=100)
    parser.add_argument('-kernel',
                        '--kernel_type',
                        type=str,
                        choices=[
                            'chebyshev', 'localpool', 'random_walk_diffusion',
                            'dual_random_walk_diffusion'
                        ],
                        default='random_walk_diffusion')  # GCN kernel type
    parser.add_argument('-K', '--cheby_order', type=int,
                        default=2)  # GCN chebyshev order
    parser.add_argument('-nn', '--nn_layers', type=int, default=2)  # layers
    parser.add_argument('-epoch', '--num_epochs', type=int, default=200)
    parser.add_argument('-loss',
                        '--loss',
                        type=str,
                        help='Specify loss function',
                        choices=['MSE', 'MAE', 'Huber'],
                        default='WMSE')
    parser.add_argument('-optim',
                        '--optimizer',
                        type=str,
                        help='Specify optimizer',
                        default='Adam')
    parser.add_argument('-lr', '--learn_rate', type=float, default=1e-3)
    parser.add_argument('-test', '--test_only', type=int,
                        default=0)  # 1 for test only

    params = parser.parse_args().__dict__  # save in dict

    # paths
    os.makedirs(params['output_dir'], exist_ok=True)

    # load data
    data_input = Utils.DataInput(data_dir=params['input_dir'])
    data, scaler = data_input.load_data()
    params['N'] = data['OD'].shape[1]
    # 缩放器
    params['scaler'] = scaler
    # get data loader
    data_generator = Utils.DataGenerator(
        obs_len=params['obs_len'],
        pred_len=params['pred_len'],
        data_split_ratio=params['split_ratio'])
    data_loader = data_generator.get_data_loader(data=data, params=params)

    # get model
    trainer = ModelTrainer(params=params, data=data, data_container=data_input)

    if bool(params['test_only']) == False:
        trainer.train(data_loader=data_loader, modes=['train', 'validate'])
    trainer.test(data_loader=data_loader, modes=['train', 'test'])
