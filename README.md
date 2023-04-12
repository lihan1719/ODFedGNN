# ODFedGNN


## Setup

进入项目目录，执行以下命令

```bash
conda create -n odfedgnn python=3.8.0
conda activate fedgnn
sh install.sh
```


## Experiments

### Main Experiments

`submission_exps/exp_main.sh` contains all commands used for experiments in Table 2 and Table 3.

### Inductive Learning on Unseen Nodes

Run `python submission_exps/exp_inductive.py` to print all commands for Table 4.

### Ablation Study: Effect of Alternating Training of Node-Level and Spatial Models

Run `python submission_exps/exp_at.py` to print all commands for Figure 2.

### Ablation Study: Effect of Client Rounds and Server Rounds

Run `python submission_exps/exp_crsr.py` to print all commands for Figure 3.
