from __future__ import annotations
import copy
import random
import time
from dataclasses import asdict
import torch
import torch.distributed as dist
from lightning.pytorch import LightningModule
from ..ablations import apply_variant, NO_CONTROLLER_VARIANTS
from ..config import RunConfig, apply_config_dict
from ..data.corpus import family_index
from ..env.proposals import ProposalKernel
from ..env.rollout import RolloutCollector
from ..env.verifiers import build_checker, build_preference_pairs
from ..models.backbone import build_backbone, format_prompt
from ..models.scoring import ScoringBranch
from ..models.policy import ActionPolicy, build_reference_policy
from ..models.value import ValueHead
from ..models.energy import pairwise_margin_loss, pairwise_logistic_loss
from ..train.compute import estimate_updates_from_paper
from ..train.fit_energy import sample_states, recompute_coordinates
from ..train.muon import build_optimizer, clip_gradients
from ..train.scheduler import build_schedule
from ..train.rl import annotate_values, build_update_batch, controller_loss, value_head_loss
from ..train.si import build_internalization_pairs, internalization_loss
from ..utils.device import describe_environment, device_report, reset_peak_memory
from ..utils.logging import RunLogger
from ..utils.seed import count_parameters

class LSPOModule(LightningModule):
    def __init__(self, config, stage="all"):
        super().__init__()
        self.config = apply_config_dict(RunConfig(), config) if isinstance(config, dict) else copy.deepcopy(config)
        apply_variant(self.config, self.config.variant)
        self.stage = stage
        self.automatic_optimization = False
        self.save_hyperparameters({"config": asdict(self.config), "stage": stage})
        self.generator = None
        self.energy_updates = 0
        self.policy_updates = 0
        self.energy_calibration = ([], [])
        self.energy_states = None
        self.energy_pairs = None
        self.frozen_pool = None
        self.last_components = {}
        self.run_logger = None
        self.eval_with_reflection = True
        self.elapsed_before = 0.0
        self.session_started = None
        self.training_fingerprint = None

    @property
    def energy_steps(self):
        return self.config.energy_fit.steps if self.config.energy_fit.enabled and self.stage != "policy" else 0

    @property
    def total_steps(self):
        return self.energy_steps + (0 if self.stage == "energy" else self.config.optim.updates)

    def configure_model(self):
        if self.generator is not None:
            return
        torch.manual_seed(self.config.seed)
        model_config = copy.deepcopy(self.config.model)
        model_config.device = "cpu"
        if self.device.type == "cuda" and model_config.dtype == "auto":
            model_config.dtype = "bf16"
        self.generator = build_backbone(model_config)
        scoring_config = copy.deepcopy(self.config)
        scoring_config.model = model_config
        self.scoring = ScoringBranch(scoring_config, self.generator)
        d = self.scoring.d_text
        c = self.config.model
        self.policy = ActionPolicy(d, c.d_z, c.d_task, c.policy_hidden)
        self.value_head = ValueHead(d, c.d_z, c.d_task, c.value_hidden, c.value_layers, c.value_activation)
        self.reference_policy = build_reference_policy(self.policy)
        initial = getattr(self,"initial_payload",None)
        if initial is not None:
            self.load_state_dict(initial["state_dict"])
            self.energy_calibration = initial["lspo"]["energy_calibration"]
            del self.initial_payload
        mesh = getattr(self, "device_mesh", None)
        if mesh is not None:
            from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
            dp = mesh["data_parallel"]
            precision = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.float32)
            for backbone in (self.generator, self.scoring.text_encoder.backbone):
                if self.config.distributed.activation_checkpointing and hasattr(backbone.model, "gradient_checkpointing_enable"):
                    backbone.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                for child in backbone.model.modules():
                    if isinstance(child, torch.nn.ModuleList):
                        for block in child:
                            fully_shard(block, mesh=dp, mp_policy=precision)
                fully_shard(backbone.model, mesh=dp, mp_policy=precision)
        self.generator.eval()
        if self.energy_steps == 0:
            self.scoring.freeze_all()
        self.checker = build_checker(self.config)
        self.kernel = ProposalKernel(self.generator, self.config.rollout)
        self.collector = RolloutCollector(self.generator, self.scoring, self.policy,
            self.reference_policy, self.kernel, self.checker, self.config)

    def family_to_index(self, family):
        return family_index(family, self.config.model.task_families)

    def configure_optimizers(self):
        cfg = self.config.optim
        self.energy_parameters = [p for n,p in self.scoring.named_parameters() if not n.startswith("text_encoder")]
        self.policy_parameters = list(self.generator.parameters()) + list(self.policy.parameters()) + list(self.value_head.parameters())
        def make(parameters):
            return build_optimizer(parameters, lr=cfg.lr, momentum=cfg.momentum,
                nesterov=cfg.nesterov, ns_steps=cfg.ns_steps, weight_decay=cfg.weight_decay)
        policy = make(self.policy_parameters)
        self.policy_schedule = build_schedule(policy, self.config, total_steps_override=cfg.updates)
        self.policy_schedule.step_count = self.policy_updates
        self.policy_schedule.set_lr(self.policy_updates)
        if self.energy_steps:
            for p in self.energy_parameters:
                p.requires_grad_(True)
            energy = make(self.energy_parameters)
            self.energy_schedule = build_schedule(energy, self.config, total_steps_override=self.energy_steps)
            self.energy_schedule.step_count = self.energy_updates
            self.energy_schedule.set_lr(self.energy_updates)
            if self.energy_updates >= self.energy_steps:
                self.scoring.freeze_all()
            return [energy, policy]
        return policy

    def elapsed_seconds(self):
        return self.elapsed_before + (time.perf_counter()-self.session_started if self.session_started is not None else 0.0)

    def _start_logger(self):
        self.session_started = time.perf_counter()
        reset_peak_memory(self.device)
        if not self.trainer.is_global_zero:
            return
        self.run_logger = RunLogger(self.config.output_dir, self.config.run_name,
            verbose=self.config.verbose)
        self.run_logger.log({"stage": "setup",
            "generator": count_parameters(self.generator),
            "scoring_branch": count_parameters(self.scoring, trainable_only=False),
            "policy": count_parameters(self.policy),
            "value_head": count_parameters(self.value_head)})
        self.run_logger.log({"stage": "environment", **describe_environment(),
            "sharding": self.config.distributed.sharding if self.trainer.world_size > 1 else "none",
            "sharding_requested": self.config.distributed.sharding,
            "world_size": self.trainer.world_size,
            "rank": self.trainer.global_rank,
            "activation_checkpointing": self.config.distributed.activation_checkpointing,
            "rank_seconds_per_update": round(estimate_updates_from_paper(), 1)})

    def on_train_start(self):
        fingerprint = self.trainer.datamodule.fingerprint()
        if self.training_fingerprint is not None and self.training_fingerprint != fingerprint:
            raise ValueError("Training dataset contents or ordering changed since checkpoint")
        self.training_fingerprint = fingerprint
        self.generator.eval()
        self._start_logger()
        if self.energy_updates < self.energy_steps and self.energy_states is None:
            items = self.trainer.datamodule.items
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(self.config.seed)
                self.energy_states = sample_states(self.generator, self.scoring, self.kernel,
                    self.checker, items, self.family_to_index, self.config)
            records = [s.as_record() for s in self.energy_states]
            index = {id(r): s for r,s in zip(records,self.energy_states)}
            self.energy_pairs = [(index[id(a)],index[id(b)]) for a,b in build_preference_pairs(records)]
            if not self.energy_pairs:
                raise ValueError("No energy preference pairs were formed")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch

    def _seed_update(self):
        torch.manual_seed(self.config.seed + self.energy_updates * 100003 + self.policy_updates * 1009 )

    def _sync_heads(self, parameters):
        if dist.is_available() and dist.is_initialized():
            for p in parameters:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
                p.grad.div_(dist.get_world_size())

    def _weight(self, count):
        if not dist.is_available() or not dist.is_initialized():
            return 1.0
        total = torch.tensor(float(count), device=self.device)
        dist.all_reduce(total)
        return count * dist.get_world_size() / max(1.0, float(total))

    def energy_loss(self):
        rng = random.Random(self.config.seed + 7)
        count = min(self.config.energy_fit.batch_pairs, len(self.energy_pairs))
        for _ in range(self.energy_updates + 1):
            pairs = rng.sample(self.energy_pairs, count)
        winners, losers = [], []
        for winner, loser in pairs:
            task = self.scoring.task_vector(winner.task_id, device=self.device)
            h,z = recompute_coordinates(self.scoring,winner,task)
            winners.append(self.scoring.energy(h,z,task))
            h,z = recompute_coordinates(self.scoring,loser,task)
            losers.append(self.scoring.energy(h,z,task))
        fn = pairwise_margin_loss if self.config.energy_fit.loss == "margin" else pairwise_logistic_loss
        return fn(torch.stack(winners),torch.stack(losers),margin=self.config.energy_fit.margin)

    def collect(self, items):
        self.generator.eval()
        self.policy.eval()
        self.scoring.eval()
        result = []
        size = self.config.optim.rollout_batch_size
        with torch.no_grad():
            for start in range(0,len(items),size):
                chunk = items[start:start+size]
                result.extend(self.collector.collect_batch(chunk,
                    [format_prompt(i.question) for i in chunk],
                    [self.family_to_index(i.family) for i in chunk]))
        return result

    def policy_loss(self, trajectories):
        transitions = [t for trajectory in trajectories for t in trajectory.transitions]
        annotate_values(self.value_head,self.scoring,transitions,self.config)
        batch = build_update_batch(trajectories,self.config)
        if self.config.variant in NO_CONTROLLER_VARIANTS:
            rl = torch.zeros((),device=self.device)
        else:
            rl,_ = controller_loss(self.generator,self.scoring,self.policy,self.reference_policy,batch,self.config)
        pairs = build_internalization_pairs(trajectories)
        si,_ = internalization_loss(self.generator,pairs,self.config)
        value,_ = value_head_loss(self.value_head,self.scoring,batch,self.config)
        self.last_components = {"rl":float(rl.detach()),"si":float(si.detach()),"value":float(value.detach())}
        return self._weight(len(transitions)) * (rl + self.config.objective.value_coef * value) + self._weight(len(pairs)) * self.config.objective.alpha * si

    def training_step(self, batch, batch_idx):
        items = self.trainer.datamodule.items
        count = self.config.optim.prompts_per_update * self.config.optim.gradient_accumulation
        start = (self.energy_updates + self.policy_updates) * count
        batch = [items[(start+i) % len(items)] for i in range(count)]
        self._seed_update()
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        energy_phase = self.energy_updates < self.energy_steps
        optimizer = optimizers[0] if energy_phase else optimizers[-1]
        self.generator.eval()
        self.reference_policy.eval()
        optimizer.zero_grad()
        update_started = time.perf_counter()
        if energy_phase:
            loss = self.energy_loss()
            self.manual_backward(loss)
            self._sync_heads(self.energy_parameters)
            clip_gradients(self.energy_parameters,self.config.optim.grad_clip)
            optimizer.step()
            self.energy_updates += 1
            self.energy_schedule.step()
            if self.energy_updates == self.energy_steps:
                self.finish_energy()
        else:
            self.scoring.freeze_all()
            accumulation = max(1,self.config.optim.gradient_accumulation)
            size = self.config.optim.prompts_per_update
            total = torch.zeros((),device=self.device)
            for micro in range(accumulation):
                subset = batch[micro*size:(micro+1)*size]
                if self.config.rollout.frozen_rollout_pool and self.frozen_pool is not None:
                    trajectories = self.frozen_pool
                else:
                    trajectories = self.collect(subset)
                    if self.config.rollout.frozen_rollout_pool:
                        self.frozen_pool = trajectories
                micro_loss = self.policy_loss(trajectories) / accumulation
                self.manual_backward(micro_loss)
                total += micro_loss.detach()
            loss = total
            self._sync_heads(list(self.policy.parameters()) + list(self.value_head.parameters()))
            clip_gradients(self.policy_parameters,self.config.optim.grad_clip)
            optimizer.step()
            self.policy_updates += 1
            self.policy_schedule.step()
            self._log_update(loss, update_started)
        self.log("energy_loss" if energy_phase else "policy_loss",loss.detach(),on_step=True,on_epoch=False,sync_dist=True,batch_size=len(batch))

    def _log_update(self, loss, started):

        if self.run_logger is None:
            return
        if self.policy_updates % max(1, self.config.log_every):
            return
        self.run_logger.log({
            "stage": "lspo_update",
            "policy_updates": self.policy_updates,
            "elapsed_total_s": self.elapsed_seconds(),
            "variant": self.config.variant,
            "loss": float(loss.detach()),
            "wall_clock_update_s": round(time.perf_counter() - started, 3),
            "peak_memory_gb": round(device_report(self.device).peak_reserved_gb, 2),
            "compute_world_size": float(self.trainer.world_size),
            **self.last_components,
        }, step=self.policy_updates)

    def finish_energy(self):
        self.scoring.freeze_all()
        energies,labels = [],[]
        states = random.Random(self.config.seed).sample(self.energy_states,min(2048,len(self.energy_states)))
        with torch.no_grad():
            for state in states:
                task = self.scoring.task_vector(state.task_id,device=self.device)
                h,z = recompute_coordinates(self.scoring,state,task)
                energies.append(float(self.scoring.energy(h,z,task)))
                labels.append(int(state.verified))
        self.energy_calibration = energies,labels

    def on_save_checkpoint(self, checkpoint):
        checkpoint["lspo"] = {"energy_updates":self.energy_updates,"policy_updates":self.policy_updates,
            "energy_calibration":self.energy_calibration,"frozen_pool":self.frozen_pool,
            "elapsed_total_s":self.elapsed_seconds(),
            "training_limit":self.trainer.datamodule.limit,
            "training_fingerprint":self.training_fingerprint}

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint["lspo"]
        self.energy_updates = state["energy_updates"]
        self.policy_updates = state["policy_updates"]
        self.energy_calibration = state["energy_calibration"]
        self.frozen_pool = state.get("frozen_pool")
        self.elapsed_before = state.get("elapsed_total_s",0.0)
        self.training_fingerprint = state.get("training_fingerprint")
        if self.energy_updates >= self.energy_steps and self.generator is not None:
            self.scoring.freeze_all()


    def on_train_end(self):
        self.elapsed_before = self.elapsed_seconds()
        self.session_started = None
        if self.run_logger is not None:
            self.run_logger.log({"stage":"training_summary","elapsed_total_s":self.elapsed_before,
                "policy_updates":self.policy_updates,"energy_updates":self.energy_updates,
                "compute_world_size":float(self.trainer.world_size),
                "peak_memory_gb":round(device_report(self.device).peak_reserved_gb,2)})
            self.run_logger.close()
            self.run_logger = None

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self._seed_update()
        from ..eval.evaluate import run_evaluation
        split, name, items = batch
        self.scoring.freeze_all()
        result = run_evaluation(self, {name: items}, self.config,
                                with_reflection=self.eval_with_reflection)
        result["verification"] = "execution" if self.config.data.allow_code_exec else "static_only"
        return split, name, result
