from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import ModelParallelStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from .module import LSPOModule
from .data import LSPODataModule

LARGE_BACKBONES = ("27B", "35B")

def memory_budget_report(config, devices: int = None) -> Tuple[float, float, List[str]]:

    if not torch.cuda.is_available():
        return 0.0, 0.0, ["no CUDA device is visible; check the GPU host and the PyTorch build"]
    count = torch.cuda.device_count()
    per = [torch.cuda.get_device_properties(i).total_memory / 2**30 for i in range(count)]
    total_gb, smallest_gb = sum(per), min(per)
    notes: List[str] = []
    if devices is not None and devices > count:
        notes.append(f"requested {devices} processes but only {count} GPUs are visible")
    backend = config.model.backbone
    if any(tag in backend for tag in LARGE_BACKBONES):
        required = config.distributed.required_total_memory_gb
        peak = config.distributed.target_peak_memory_gb
        if total_gb < required:
            notes.append(
                f"{total_gb:.0f} GB of visible GPU memory is below the "
                f"{required:.0f} GB this configuration is sized for"
            )
        elif smallest_gb < peak:
            notes.append(
                f"the paper run peaked at {peak} GB/GPU, above the "
                f"{smallest_gb:.0f} GB visible here"
            )
    return total_gb, smallest_gb, notes

def build_trainer(config,steps,devices=8,accelerator="gpu",checkpointing=True):
    if accelerator == "gpu":
        for note in memory_budget_report(config, devices)[2]:
            rank_zero_warn(note)
        strategy = ModelParallelStrategy(data_parallel_size=devices,tensor_parallel_size=1,save_distributed_checkpoint=False)
        precision = "32-true"
    else:
        if devices != 1:
            raise ValueError("CPU validation uses one process")
        strategy,precision = "auto","32-true"
    callbacks = [ModelCheckpoint(dirpath=Path(config.output_dir)/"checkpoints",
        every_n_train_steps=max(1,config.save_every),save_last=True,save_top_k=1)] if checkpointing else []
    return Trainer(accelerator=accelerator,devices=devices,strategy=strategy,precision=precision,
        max_steps=steps,max_epochs=-1,use_distributed_sampler=False,
        logger=CSVLogger(config.output_dir,name=config.run_name),callbacks=callbacks,
        enable_checkpointing=checkpointing,enable_model_summary=False,
        log_every_n_steps=max(1,config.log_every),num_sanity_val_steps=0)

def run(config,stage="all",resume=None,limit=None,devices=8,accelerator="gpu",initialize_from=None):
    if stage == "policy" and not (resume or initialize_from):
        raise ValueError("Policy-only training requires --resume or --initialize-from an energy checkpoint")
    if resume and initialize_from:
        raise ValueError("Use either resume or initialize_from")
    if resume:
        payload = torch.load(resume,map_location="cpu",weights_only=False)
        from dataclasses import asdict
        saved = payload["hyper_parameters"]
        if saved["stage"] != stage:
            raise ValueError("Resume must use the saved stage; use initialize_from for stage transfer")
        current = asdict(config)
        previous = saved["config"]
        for key in ("seed","variant","model","objective","rollout","optim","energy_fit","data"):
            if current[key] != previous[key]:
                raise ValueError(f"Resume configuration differs in {key}")
        if "training_limit" not in payload["lspo"]:
            raise ValueError("Checkpoint lacks training_limit; exact resume cannot be verified")
        if payload["lspo"]["training_limit"] != limit:
            raise ValueError("Resume training limit differs from checkpoint")
    model = LSPOModule(config,stage)
    if initialize_from:
        payload = torch.load(initialize_from,map_location="cpu",weights_only=False)
        expected = payload["hyper_parameters"]["config"]["energy_fit"]["steps"]
        if payload["lspo"]["energy_updates"] < expected:
            raise ValueError("Energy fitting must finish before policy-stage transfer")
        model.initial_payload = payload
    data = LSPODataModule(config,limit=limit)
    trainer = build_trainer(config,model.total_steps,devices,accelerator)
    trainer.fit(model,datamodule=data,ckpt_path=resume,weights_only=False)
    target = str(Path(config.output_dir)/"final.ckpt")
    trainer.save_checkpoint(target)
    return target
