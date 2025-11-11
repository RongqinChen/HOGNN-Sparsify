from pytorch_lightning.loggers import CometLogger
import argparse
import os
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch_geometric.transforms import Compose

from transforms import transform_dict


class cfg(object):
    cfg_file: str = ''

    class dataset(object):
        path: str = None
        dataname: str = None
        node_attr_dim: Optional[int] = None
        edge_attr_dim: Optional[int] = None
        full_pre_transform: Optional[dict] = None  # transformed data is saved for next loading
        inmemory_transform: Optional[dict] = None  # transforming before training
        onthefly_transform: Optional[dict] = None  # transforming during training
        poly_method = None                         # name of polynomials for graph structrual encoding
        poly_dim = None                            # dimension of graph structrual encoding
        follow_batch = None                        # Creates assignment batch vectors for each key of data in the list

    class model(object):
        # basic config
        name: str = ""
        node_encoder: str = None
        edge_encoder: str = None
        loop_encoder: str = None
        pair_encoder: str = None
        gse_encoder: str = None
        hidden_dim: int = 0
        layer_type: str = None
        num_layers: int = 0
        mlp_depth: int = 0
        pooling: str = ""
        drop_prob = 0.0
        output_drop_prob = 0.0
        jk_mode: str = ""
        task_type: str = ""
        num_tasks: int = 0
        norm_type: str = ""
        act_type: str = "relu"
        agg: str = ""
        ffn: bool = True
        residual: bool = True

        # DenseInputEncoder config
        max_num_nodes: int = None

        # Graph Transformer config
        attn_heads = 0
        attn_drop_prob = 0.20
        clamp = 5.0
        weight_fn = "softmax"
        degree_scaler: str = "log(1+deg)"

    class train(object):
        runs: int = 10
        save_dir: str = "results"
        num_workers: int = 0
        batch_size: int = 64
        lr: float = 1e-3  # learning rate
        min_lr: float = 1e-6  # minimal learning rate
        weight_decay: float = 1e-5
        num_warmup_epochs: int = 50
        num_epochs: int = 2000
        test_eval_interval: int = 10
        optimizer: str = "AdamW"
        scheduler: str = "cosine_with_warmup"
        accumulate_grad_batches = 1
        log_every_n_steps = 50
        ckpt_period = None

    class log(object):
        local_log = False
        monitor = 'val/mae'
        monitor_mode = 'min'


def load_cfg(args: argparse.Namespace):
    with open(args.cfg_file, "r") as file:
        cfg_dict = yaml.safe_load(file)

    args_dict = args.__dict__

    # looping every config item
    for key1, val1 in cfg.__dict__.items():
        if isinstance(val1, str):
            if key1 in cfg_dict:
                setattr(cfg, key1, convert(cfg_dict[key1]))
            if args_dict.get(key1, None) is not None:
                setattr(cfg, key1, convert(args_dict[key1]))

        elif hasattr(val1, "__dict__"):
            for key2 in val1.__dict__.keys():
                if key1 in cfg_dict and key2 in cfg_dict[key1]:
                    setattr(getattr(cfg, key1), key2, convert(cfg_dict[key1][key2]))
                if args_dict.get(key2, None) is not None:
                    setattr(getattr(cfg, key1), key2, convert(args_dict[key2]))

    custom_args = {'poly_dim': cfg.dataset.poly_dim}
    if cfg.dataset.poly_method is not None:
        custom_args['poly_method'] = cfg.dataset.poly_method

    if cfg.dataset.full_pre_transform is not None:
        trans = []
        for tran, paras in cfg.dataset.full_pre_transform.items():
            paras.update(custom_args)
            trans.append(transform_dict[tran](**paras))
        if len(trans) > 1:
            cfg.dataset.full_pre_transform = Compose(trans)
        else:
            cfg.dataset.full_pre_transform = trans[0]

    if cfg.dataset.onthefly_transform is not None:
        trans = []
        for tran, paras in cfg.dataset.onthefly_transform.items():
            paras.update(custom_args)
            trans.append(transform_dict[tran](**paras))

        if len(trans) > 1:
            cfg.dataset.onthefly_transform = Compose(trans)
        else:
            cfg.dataset.onthefly_transform = trans[0]

    if cfg.dataset.inmemory_transform is not None:
        trans = []
        if cfg.dataset.inmemory_transform is not None:
            for tran, paras in cfg.dataset.inmemory_transform.items():
                paras.update(custom_args)
                trans.append(transform_dict[tran](**paras))

        if len(trans) > 1:
            cfg.dataset.inmemory_transform = Compose(trans)
        elif len(trans) == 1:
            cfg.dataset.inmemory_transform = trans[0]

    return cfg


def convert(val):
    try:
        # Attempt to convert val to float
        float_val = float(val)

        # Check if the float value is actually an integer
        if float_val.is_integer():
            return int(float_val)
        else:
            return float_val
    except (ValueError, TypeError):
        # Return the original value if conversion fails
        return val


def sanitize_path(path):
    # Ensure the path is safe and normalized
    return str(Path(path).resolve())


def make_logger_trainer(timestamp, run_label, timer, enable_progress_bar=False):

    cfg_label = f"{Path(cfg.cfg_file).stem}"
    if cfg.dataset.poly_method is not None:
        cfg_label += f".{cfg.dataset.poly_method}"
    if cfg.dataset.poly_dim is not None:
        cfg_label += f".{cfg.dataset.poly_dim}"

    save_dir = sanitize_path(f"{cfg.train.save_dir}/{cfg_label}.{timestamp}")
    save_subdir = f"{save_dir}/save"
    os.makedirs(save_subdir, exist_ok=True)

    logger = CometLogger(
        name=f"{run_label}-{cfg_label}",
        workspace="cognns",
        project=f"{cfg_label}",
        online=not cfg.log.local_log,
        offline_directory=save_subdir,
        api_key="Input your key",
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.train.num_epochs,
        enable_checkpointing=True,
        enable_progress_bar=enable_progress_bar,
        logger=logger,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        callbacks=[
            ModelCheckpoint(monitor=cfg.log.monitor, mode=cfg.log.monitor_mode,
                            every_n_epochs=cfg.train.ckpt_period),
            LearningRateMonitor(logging_interval="epoch"),
            timer
        ],
        log_every_n_steps=cfg.train.log_every_n_steps,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches
    )
    return logger, trainer


def log_final_results(model, results_allruns, total_runs):
    print('\n' * 2)
    for line in str(model).split('\n'):
        print(line)
    print(f"torch.cuda.max_memory_reserved: {(torch.cuda.max_memory_reserved() / (1024 ** 3)):.1f}GB")

    max_len = max(list(map(len, results_allruns.keys())))
    print('\n' * 2)
    for key, vals in results_allruns.items():
        pad = ' ' * (max_len + 2 - len(key))
        print(f"{key}:{pad}{vals[-1]:.5f}")

    current_runs = len(results_allruns['final/avg_train_time_epoch'])
    if total_runs > 1 and current_runs == total_runs:
        print('\n' * 2)
        print(f"Total {total_runs} runs:")
        for key, vals in results_allruns.items():
            pad = ' ' * (max_len + 2 - len(key))
            print(f"{key}:{pad}{np.mean(vals):.5f} +- {np.std(vals):.5f}")
