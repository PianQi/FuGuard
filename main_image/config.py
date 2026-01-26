#%%
import torch

CONFIG = {
    "DATA_FOLDER": "/home/modal-workbench/Projects/Pian/FuGuard/data/IMAGE",
    "SAVE_DATA_FOLDER": "/home/modal-workbench/Projects/Pian/FuGuard/backdoor/processed_IMAGE",
    "SAVE_RESULTS_FOLDER": "/home/modal-workbench/Projects/Pian/FuGuard/results_image_ot_sen",
    "seed": 42,

    "dataset_name": 'CIFAR10',   # SVHN, EuroSAT, CIFAR10, CIFAR100
    "NUM_CLASSES": 10,
    "IMAGE_SIZE": (32, 32),
    "num_clients": 10,
    "dirichlet_alpha": 0.1,
    "inject_ratio": 1,

    "BACKDOOR_CLIENT": "client_3",
    "target_label": 9,

    "ROUNDS": 60,
    "EPOCHS_LOCAL": 5,
    "BATCH_SIZE": 256,
    "LR": 1e-3,

    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    #fuguard
    'samples_scale': 0.1,
    'gen_bs': 64,  #64
    'unlearn_bs': 32,  #64
    'ot_lambda': 0.1,     # 0.1

    #quickdrop
    "scale": 0.01,
    "init": "real",
    "LR_IMG": 1,
    "batch_real": 256,
    "quick_method": 'DC',
    "dsa": False,
    "dsa_strategy": None,
    "directly_update": False,
    "dd_scale_s": 100,            # 论文里默认 s=100（约 1% 数据量）
    "dd_steps_per_local_step": 1, # ζS：每个 local step 更新 synthetic 的步数
    "dd_syn_lr": 0.1,             # ηS：synthetic 更新学习率（可调）
    "dd_batch_syn": 64,          # synthetic batch size
    "dd_batch_real": 64,         # real batch size
    "unlearn_rounds": 1,          # unlearning rounds 数
    "recover_rounds": 1,          # recovery rounds 数
    "recover_mixed_ratio": 1.0,   # sampled real : synthetic = 1:1（写成比例）
    "unlearn_local_steps": 5,    # unlearning 每轮每客户端本地步数 T_u
    "recover_local_steps": 10,    # recovery 每轮每客户端本地步数 T_r

    'forgetting_epoch': 5,
    'forgetting_lr': 1e-3,

    "EXTERNAL_ALERTS": {
    "client_3": [50],
    },

    "method": 'fuguard',   # retrain, fedsga, fu, quick-drop, not, fast-fedul, fuguard
    'distance_threshold': 2.2,
}


