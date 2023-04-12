# HA (centralized,)
python main.py --dataset NYC_bike --seed 2023 --model_name HApredicter --base_model_name HistoricalAverage --wandb  --project odfedgnn --data_dir data/NYC_bike/processed_bike_data/ --data_name nyc_2019-07_2019-09_60T

# GRU (centralized, 63K)
python main.py --dataset NYC_bike --seed 2023 --model_name ODPredictor --base_model_name GRUSeq2Seq --batch_size 64 --hidden_size 100 --gru_num_layers 1 --gpus 0, --use_curriculum_learning --wandb  --project ODFedGNN-Main-201907_300_15T --early_stop_patience 10 --data_dir data/NYC_bike/processed_bike_data/ --data_name 2019-07_2019-07_grid_300_15T

# ODFedGNN(64K + 1M)
python main.py --dataset NYC_bike --seed 2023 --model_name SplitFedAvgNodePredictor --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --server_batch_size 64 --hidden_size 64 --use_curriculum_learning --mp_worker_num 1 --sync_every_n_epoch 1 --server_epoch 5 --gcn_on_server --gpus 0, --max_epochs 200 --early_stop_patience 10 --gru_num_layers 2 --wandb --project ODFedGNN-Main-201907_300_15T --resume --data_dir data/NYC_bike/processed_bike_data/ --data_name 2019-07_2019-07_grid_300_15T
# 继续训练（微调）
python main.py --dataset NYC_bike --seed 2023 --model_name ODFedAvgNodePredictor --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --server_batch_size 64 --hidden_size 64 --use_curriculum_learning --mp_worker_num 1 --sync_every_n_epoch 1 --server_epoch 1 --gcn_on_server --gpus 0, --max_epochs 200 --early_stop_patience 10 --gru_num_layers 1 --wandb --project ODFedGNN-Weight_loss --data_dir data/NYC_bike/processed_bike_data/ --data_name 2019-07_2019-07_grid_300_15T  --restore_train_ckpt_path cash/artifacts/lightning_logs/version_22/checkpoints/epoch=14.ckpt

# DCRNN与ODCRN 切换至baseline/ODCRN-main/model下相应目录运行main.py

# 测试
python main.py --dataset NYC_bike --seed 2023 --model_name SplitFedAvgNodePredictor --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --server_batch_size 64 --hidden_size 64 --use_curriculum_learning --mp_worker_num 1 --sync_every_n_epoch 1 --server_epoch 5 --gcn_on_server --gpus 0, --max_epochs 200 --early_stop_patience 10 --gru_num_layers 2 --wandb --project ODtest --resume --data_dir data/NYC_bike/processed_bike_data/ --data_name 2019-07_2019-07_grid_300_15T 