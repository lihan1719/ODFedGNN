# python=3.8.0
conda install -y numpy scipy pandas jupyter matplotlib pyyaml scikit-learn tqdm networkx jupyterlab pytables
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install pytorch-lightning==0.9.0
pip install selenium
pip install h5py
pip install seaborn
pip install wandb
pip install pyg_lib==0.1.0+pt112cu113 -f https://pytorch-geometric.com/whl/torch-1.12.1%2Bcu113.html
pip install torch_scatter==2.1.0+pt112dcu113 -f https://pytorch-geometric.com/whl/torch-1.12.1%2Bcu113.html
pip install torch_sparse==0.6.16+pt112cu113 -f https://pytorch-geometric.com/whl/torch-1.12.1%2Bcu113.html
pip install torch_cluster==1.6.0+pt112cu113 -f https://pytorch-geometric.com/whl/torch-1.12.1%2Bcu113.html
pip install torch_spline_conv==1.2.1+pt112cu113 -f https://pytorch-geometric.com/wddhl/torch-1.12.1%2Bcu113.html
pip install torch-geometric
pip install protobuf==3.20.*
