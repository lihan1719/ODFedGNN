# # 不同预测步长下,模型的性能

# # HA
# python main.py --dataset NYC_bike --model_name HApredicter --data_dir data/NYC_bike/processed_bike_data/ --data_name nyc_2019-07_2019-09_60T --obs 1 --pred 1

# python main.py --dataset NYC_bike --model_name HApredicter --data_dir data/NYC_bike/processed_bike_data/ --data_name nyc_2019-07_2019-09_60T --obs 1 --pred 2

# python main.py --dataset NYC_bike --model_name HApredicter --data_dir data/NYC_bike/processed_bike_data/ --data_name nyc_2019-07_2019-09_60T --obs 1 --pred 3

# python main.py --dataset NYC_bike --model_name HApredicter --data_dir data/NYC_bike/processed_bike_data/ --data_name nyc_2019-07_2019-09_60T --obs 1 --pred 4



# # GRUSeq2Seq
# python main.py --dataset NYC_bike --seed 2023 --model_name ODGRUSeq2Seq --base_model_name GRUSeq2Seq --batch_size 128 --hidden_size 100 --gru_num_layers 1 --gpus 0, --use_curriculum_learning --early_stop_patience 10 --data_dir data/NYC_bike/processed_bike_data/ --data_name 2019-07_2019-07_grid_300_15T --obs 1 --pred 1

# python main.py --dataset NYC_bike --seed 2023 --model_name ODGRUSeq2Seq --base_model_name GRUSeq2Seq --batch_size 128 --hidden_size 100 --gru_num_layers 1 --gpus 0, --use_curriculum_learning --early_stop_patience 10 --data_dir data/NYC_bike/processed_bike_data/ --data_name 2019-07_2019-07_grid_300_15T --obs 1 --pred 2

# python main.py --dataset NYC_bike --seed 2023 --model_name ODGRUSeq2Seq --base_model_name GRUSeq2Seq --batch_size 128 --hidden_size 100 --gru_num_layers 1 --gpus 0, --use_curriculum_learning --early_stop_patience 10 --data_dir data/NYC_bike/processed_bike_data/ --data_name 2019-07_2019-07_grid_300_15T --obs 1 --pred 3

# python main.py --dataset NYC_bike --seed 2023 --model_name ODGRUSeq2Seq --base_model_name GRUSeq2Seq --batch_size 128 --hidden_size 100 --gru_num_layers 1 --gpus 0, --use_curriculum_learning --early_stop_patience 10 --data_dir data/NYC_bike/processed_bike_data/ --data_name 2019-07_2019-07_grid_300_15T --obs 1 --pred 4

# # DCRNN

# python baseline-DCRNN/Main.py -in data/NYC_bike/processed_bike_data -out baseline-DCRNN/output  --loss MSE --obs 1 --pred 1

# python baseline-DCRNN/Main.py -in data/NYC_bike/processed_bike_data -out baseline-DCRNN/output --loss MSE --obs 1 --pred 2

# python baseline-DCRNN/Main.py -in data/NYC_bike/processed_bike_data -out baseline-DCRNN/output --loss MSE --obs 1 --pred 3

# python baseline-DCRNN/Main.py -in data/NYC_bike/processed_bike_data -out baseline-DCRNN/output --loss MSE --obs 1 --pred 4

# ODFedGNN

python main.py --dataset NYC_bike --seed 2023 --model_name ODFedAvg --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --server_batch_size 24 --hidden_size 64 --use_curriculum_learning --mp_worker_num 1 --sync_every_n_epoch 1 --server_epoch 5 --gcn_on_server --gpus 0, --max_epochs 200 --early_stop_patience 10 --gru_num_layers 2 --data_dir data/NYC_bike/processed_bike_data/ --data_name 2019-07_2019-07_grid_300_15T --obs 1 --pred 1

python main.py --dataset NYC_bike --seed 2023 --model_name ODFedAvg --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --server_batch_size 24 --hidden_size 64 --use_curriculum_learning --mp_worker_num 1 --sync_every_n_epoch 1 --server_epoch 5 --gcn_on_server --gpus 0, --max_epochs 200 --early_stop_patience 10 --gru_num_layers 2 --data_dir data/NYC_bike/processed_bike_data/ --data_name 2019-07_2019-07_grid_300_15T --obs 1 --pred 2

python main.py --dataset NYC_bike --seed 2023 --model_name ODFedAvg --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --server_batch_size 24 --hidden_size 64 --use_curriculum_learning --mp_worker_num 1 --sync_every_n_epoch 1 --server_epoch 5 --gcn_on_server --gpus 0, --max_epochs 200 --early_stop_patience 10 --gru_num_layers 2 --data_dir data/NYC_bike/processed_bike_data/ --data_name 2019-07_2019-07_grid_300_15T --obs 1 --pred 3

python main.py --dataset NYC_bike --seed 2023 --model_name ODFedAvg --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --server_batch_size 24 --hidden_size 64 --use_curriculum_learning --mp_worker_num 1 --sync_every_n_epoch 1 --server_epoch 5 --gcn_on_server --gpus 0, --max_epochs 200 --early_stop_patience 10 --gru_num_layers 2 --data_dir data/NYC_bike/processed_bike_data/ --data_name 2019-07_2019-07_grid_300_15T --obs 1 --pred 4
