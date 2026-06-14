"""Compact energy+force(+stress) trainer for MACE-ICTD (make_fx compile + size-bucketing).

This is the baseline energy+force training path extracted from the FSCETP trainer,
kept numerically faithful but stripped of the modes MACE-ICTD does not ship
(physical-tensor heads / multi-fidelity / external field / CUDA-graph /
compiled-autograd). What it preserves, verbatim in math:

  * energy:  ``E_mol = scatter_sum(E_per_atom, batch_idx) + scatter_sum(E0[A], batch_idx)``
             where ``E0`` is the per-type atomic-energy offset (the dataset stores
             energies already-baseline-subtracted; we add it back here).
  * forces:  ``f = -dE_mol/dpos`` via ``autograd.grad(E_mol.sum(), pos, create_graph=training)``.
             ``E0`` is pos-independent, so this equals ``-dE_conv/dpos``.
  * stress:  optional (``stress_weight > 0``) -- ``sigma = dE/dstrain / V`` via the
             strain-derivative trick: deform positions+cell by ``I + sym(strain)`` and
             differentiate w.r.t. ``strain`` at ``strain=0`` (FSCETP convention).
  * loss:    per-atom-normalized ``SmoothL1(beta=loss_beta)`` on energy/force/stress
             by default (or MSE when requested), ``force_ref *= force_shift_value``;
             ``total = a*E + b*F (+ c*stress)``.

The make_fx fast path flattens ``forward + inner force-autograd`` (and the strain
derivative when stress is on) into one FX graph
and ``torch.compile``s it, so the outer ``loss.backward()`` is a single ordinary
backward (sidestepping the 2nd-order-through-autograd.grad limit). It is keyed per
input shape by :class:`CompiledForceCache`; pair it with the ``BucketBatchSampler``
so each size-bucket is exactly one fixed shape = one compile.

Node padding (``pad_nodes_to_max`` / bucketing) is a numeric no-op: dummy edges get
length >> max_radius (masked by the model), and ``atom_mask`` zeros dummy atoms in the
energy sum and excludes them from the loss denominators.
"""

from __future__ import annotations

import logging
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import SWALR

from mace_ictd.utils.scatter import scatter
from mace_ictd.utils.tensor_utils import map_tensor_values
from mace_ictd.utils.fidelity import smooth_l1_loss_stats
from mace_ictd.models.losses import RMSELoss

log = logging.getLogger(__name__)

# H/C/N/O defaults — the same fallback the FSCETP trainer uses when no E0 is supplied.
_DEFAULT_E0_KEYS = [1, 6, 7, 8]
_DEFAULT_E0_VALUES = [-430.53299511, -821.03326787, -1488.18856918, -2044.3509823]


class ForceTrainer:
    """Energy+force trainer with optional make_fx compile + size-bucketing."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        *,
        val_loader=None,
        device="cuda",
        config=None,
        dtype: torch.dtype | None = None,
        max_radius: float = 5.0,
        tensor_product_mode: str = "pure-cartesian-ictd-fix",
        # loss
        energy_weight: float = 1.0,
        force_weight: float = 10.0,
        stress_weight: float = 0.0,
        force_shift_value: float = 1.0,
        loss_type: str = "smooth_l1",
        loss_beta: float = 0.5,
        atomic_energy_keys=None,
        atomic_energy_values=None,
        # optimization
        learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-6,
        weight_decay: float = 0.0,
        optimizer_type: str = "adamw",
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        amsgrad: bool = False,
        lr_scheduler: str = "cosine",
        warmup_batches: int = 0,
        warmup_start_ratio: float = 0.1,
        lr_factor: float = 0.8,
        scheduler_patience: int = 50,
        lr_scheduler_gamma: float = 0.9993,
        lr_decay_step: int = 1000,
        lr_decay_factor: float = 0.98,
        epochs: int = 1,
        max_steps: int | None = None,
        max_grad_norm: float | None = None,
        # averaged weights
        ema_decay: float = 0.0,
        ema_start_step: int = 0,
        stage_two_enabled: bool = False,
        swa_start_epoch: int = -1,
        swa_start_step: int = -1,
        swa_lr: float | None = None,
        swa_energy_weight: float | None = None,
        swa_force_weight: float | None = None,
        swa_stress_weight: float | None = None,
        swa_anneal_epochs: int = 1,
        swa_anneal_strategy: str = "linear",
        checkpoint_state_source: str = "auto",
        # make_fx
        train_makefx_compile: bool = False,
        require_train_makefx_compile: bool = False,
        makefx_max_slots: int = 8,
        # plumbing
        train_sampler=None,
        log_interval: int = 10,
        checkpoint_path: str | None = None,
        extra_hparams: dict | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.config = config
        self.dtype = dtype if dtype is not None else next(model.parameters()).dtype
        self.max_radius = float(max_radius)
        self.tensor_product_mode = tensor_product_mode

        self.a = float(energy_weight)
        self.b = float(force_weight)
        self.c = float(stress_weight)  # stress/virial loss weight; 0 disables the stress path
        self.force_shift_value = float(force_shift_value)
        self.loss_type = str(loss_type).lower()
        if self.loss_type not in {"smooth_l1", "mse"}:
            raise ValueError(f"unsupported loss_type {loss_type!r}; use 'smooth_l1' or 'mse'")
        self.loss_beta = float(loss_beta)
        if self.loss_type == "smooth_l1" and self.loss_beta <= 0:
            raise ValueError("loss_beta must be positive for smooth_l1 loss")

        keys = atomic_energy_keys if atomic_energy_keys is not None else _DEFAULT_E0_KEYS
        values = atomic_energy_values if atomic_energy_values is not None else _DEFAULT_E0_VALUES
        self.keys = torch.as_tensor(list(keys), dtype=torch.long, device=self.device)
        self.values = torch.as_tensor(list(values), dtype=self.dtype, device=self.device)
        self.criterion_2 = RMSELoss()

        self.learning_rate = float(learning_rate)
        self.min_learning_rate = float(min_learning_rate)
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.min_learning_rate < 0:
            raise ValueError("min_learning_rate must be non-negative")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must be <= learning_rate")
        self.weight_decay = float(weight_decay)
        self.optimizer_type = str(optimizer_type).lower()
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_eps = float(adam_eps)
        if not (0.0 <= self.adam_beta1 < 1.0 and 0.0 <= self.adam_beta2 < 1.0):
            raise ValueError("Adam betas must satisfy 0 <= beta < 1")
        if self.adam_eps <= 0:
            raise ValueError("adam_eps must be positive")
        self.amsgrad = bool(amsgrad)
        self.lr_scheduler_kind = self._normalize_scheduler_kind(lr_scheduler)
        self.warmup_batches = int(max(0, warmup_batches))
        self.warmup_start_ratio = float(warmup_start_ratio)
        if self.warmup_start_ratio < 0.0 or self.warmup_start_ratio > 1.0:
            raise ValueError("warmup_start_ratio must be in [0, 1]")
        self.lr_factor = float(lr_factor)
        if self.lr_factor <= 0.0 or self.lr_factor >= 1.0:
            raise ValueError("lr_factor must satisfy 0 < lr_factor < 1")
        self.scheduler_patience = int(max(0, scheduler_patience))
        self.lr_scheduler_gamma = float(lr_scheduler_gamma)
        if self.lr_scheduler_gamma <= 0.0 or self.lr_scheduler_gamma > 1.0:
            raise ValueError("lr_scheduler_gamma must satisfy 0 < gamma <= 1")
        self.lr_decay_step = int(lr_decay_step)
        self.lr_decay_factor = float(lr_decay_factor)

        self.epochs = int(epochs)
        self.max_steps = int(max_steps) if max_steps is not None and int(max_steps) > 0 else None
        self.global_step = 0
        self.max_grad_norm = max_grad_norm
        self.log_interval = int(log_interval)
        self.checkpoint_path = checkpoint_path
        self.train_sampler = train_sampler
        # Construction choices the deploy-side from_checkpoint reads but that are NOT
        # ModelConfig fields (save_contraction_order / ictd_save_tp_mode / invariant_channels /
        # radial_sqrt_num_basis / avg_num_neighbors ...). Merged into model_hyperparameters
        # so the saved checkpoint rebuilds the exact same architecture.
        self.extra_hparams = dict(extra_hparams or {})

        # make_fx state (cache held outside the module tree so its duplicated flat
        # param views stay out of parameter discovery / DDP).
        self.train_makefx_compile = bool(train_makefx_compile)
        self.require_train_makefx_compile = bool(require_train_makefx_compile)
        self._makefx_max_slots = int(makefx_max_slots)
        self._makefx_disabled = False
        object.__setattr__(self, "_makefx_cache", None)

        self.ema_decay = float(ema_decay)
        if self.ema_decay < 0.0 or self.ema_decay >= 1.0:
            raise ValueError("ema_decay must satisfy 0 <= ema_decay < 1")
        self.ema_start_step = max(0, int(ema_start_step))
        self._ema_state = None

        self.swa_start_epoch = int(swa_start_epoch)
        self.swa_start_step = int(swa_start_step)
        if self.swa_start_epoch < -1:
            raise ValueError("swa_start_epoch must be -1 (disabled) or >= 0")
        if self.swa_start_step < -1:
            raise ValueError("swa_start_step must be -1 (disabled) or >= 0")
        self.stage_two_enabled = bool(
            stage_two_enabled or self.swa_start_epoch >= 0 or self.swa_start_step >= 0
        )
        if self.stage_two_enabled and self.swa_start_epoch < 0 and self.swa_start_step < 0:
            self.swa_start_epoch = max(1, self.epochs // 4 * 3)
        if self.stage_two_enabled and self.swa_start_epoch > self.epochs and self.max_steps is None:
            log.warning(
                "stage two start epoch %d is greater than total epochs %d; stage two will not activate",
                self.swa_start_epoch,
                self.epochs,
            )
        self.swa_lr = self.learning_rate if swa_lr is None else float(swa_lr)
        if self.swa_lr <= 0:
            raise ValueError("swa_lr must be positive")
        if self.stage_two_enabled and not (self.min_learning_rate <= self.swa_lr <= self.learning_rate):
            raise ValueError("swa_lr must satisfy min_learning_rate <= swa_lr <= learning_rate")
        self.stage_two_energy_weight = (
            self.a if swa_energy_weight is None else float(swa_energy_weight)
        )
        self.stage_two_force_weight = (
            self.b if swa_force_weight is None else float(swa_force_weight)
        )
        self.stage_two_stress_weight = (
            self.c if swa_stress_weight is None else float(swa_stress_weight)
        )
        self.swa_anneal_epochs = int(max(1, swa_anneal_epochs))
        self.swa_anneal_strategy = str(swa_anneal_strategy)
        if self.swa_anneal_strategy not in {"linear", "cos"}:
            raise ValueError("swa_anneal_strategy must be 'linear' or 'cos'")
        self._stage_two_active = False
        self._stage_two_activated_epoch = None
        self._stage_two_activated_step = None
        self._swa_scheduler = None
        self._swa_state = None
        self._swa_n = 0
        self.checkpoint_state_source = str(checkpoint_state_source or "auto").lower()
        if self.checkpoint_state_source not in {"auto", "raw", "ema", "swa"}:
            raise ValueError("checkpoint_state_source must be one of auto/raw/ema/swa")

        self._build_optimizer(self.optimizer_type, self.learning_rate, self.weight_decay, self.amsgrad)
        self._build_scheduler(
            self.lr_scheduler_kind, self.warmup_batches, self.warmup_start_ratio,
            self.lr_decay_step, self.lr_decay_factor, self.min_learning_rate,
        )

    # ------------------------------------------------------------------ setup
    def _build_optimizer(self, optimizer_type, lr, weight_decay, amsgrad):
        params = list(self.model.parameters())
        kind = str(optimizer_type).lower()
        common = dict(lr=lr, betas=(self.adam_beta1, self.adam_beta2), eps=self.adam_eps,
                      weight_decay=weight_decay, amsgrad=bool(amsgrad))
        if kind == "adam":
            self.optimizer = torch.optim.Adam(params, **common)
        else:
            self.optimizer = torch.optim.AdamW(params, **common)

    @staticmethod
    def _normalize_scheduler_kind(kind):
        raw = str(kind or "none").strip().lower().replace("_", "").replace("-", "")
        aliases = {
            "none": "none",
            "off": "none",
            "cosine": "cosine",
            "cosineannealinglr": "cosine",
            "exp": "exp",
            "exponential": "exp",
            "exponentiallr": "exp",
            "plateau": "plateau",
            "reducelronplateau": "plateau",
            "step": "step",
            "steplr": "step",
        }
        if raw not in aliases:
            raise ValueError("lr_scheduler must be one of none/cosine/exp/plateau/step")
        return aliases[raw]

    def _build_scheduler(self, kind, warmup_batches, warmup_start_ratio,
                         decay_step, decay_factor, min_lr):
        steps_per_epoch = max(1, len(self.train_loader))
        total_steps = max(1, self.max_steps or (self.epochs * steps_per_epoch))
        self._warmup_start_lr = max(min_lr, self.learning_rate * float(warmup_start_ratio))
        self._post_warmup_step = 0
        self._cosine_total_steps = max(1, total_steps - int(max(0, warmup_batches)))
        self._scheduler = None
        self._scheduler_kind = kind
        self._scheduler_interval = "none"
        if kind == "plateau":
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.lr_factor,
                patience=self.scheduler_patience,
                min_lr=min_lr,
            )
            self._scheduler_interval = "epoch"
        elif kind == "exp":
            self._scheduler_interval = "epoch"
        elif kind in {"cosine", "step"}:
            self._scheduler_interval = "step"
        if warmup_batches > 0:
            self._set_optimizer_lrs(self._warmup_start_lr)
        else:
            self._set_optimizer_lrs(self.learning_rate)

    def _set_optimizer_lrs(self, lr):
        lr = min(self.learning_rate, max(self.min_learning_rate, float(lr)))
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _clamp_optimizer_lrs(self):
        for group in self.optimizer.param_groups:
            group["lr"] = min(self.learning_rate, max(self.min_learning_rate, float(group["lr"])))

    def _step_scheduler_after_batch(self):
        if self._stage_two_active:
            return
        if self.warmup_batches > 0 and self.global_step <= self.warmup_batches:
            frac = min(float(self.global_step) / max(1, self.warmup_batches), 1.0)
            lr = self._warmup_start_lr + (self.learning_rate - self._warmup_start_lr) * frac
            self._set_optimizer_lrs(lr)
            return
        if self.warmup_batches > 0 and self.global_step == self.warmup_batches + 1:
            self._set_optimizer_lrs(self.learning_rate)
        if self._scheduler_kind == "cosine":
            self._post_warmup_step += 1
            t = min(self._post_warmup_step, self._cosine_total_steps)
            frac = 0.5 * (1.0 + math.cos(math.pi * t / self._cosine_total_steps))
            lr = self.min_learning_rate + (self.learning_rate - self.min_learning_rate) * frac
            self._set_optimizer_lrs(lr)
        elif self._scheduler_kind == "step" and self.lr_decay_step > 0:
            if self.global_step % self.lr_decay_step == 0:
                for group in self.optimizer.param_groups:
                    group["lr"] = float(group["lr"]) * float(self.lr_decay_factor)
                self._clamp_optimizer_lrs()

    def _step_scheduler_after_epoch(self, metric):
        if self._stage_two_active:
            if self._swa_scheduler is not None:
                self._swa_scheduler.step()
                self._clamp_optimizer_lrs()
            return
        if self._scheduler_kind == "plateau" and self._scheduler is not None:
            self._scheduler.step(metric)
            self._clamp_optimizer_lrs()
        elif self._scheduler_kind == "exp":
            for group in self.optimizer.param_groups:
                group["lr"] = float(group["lr"]) * self.lr_scheduler_gamma
            self._clamp_optimizer_lrs()

    def _maybe_activate_stage_two(self, epoch: int):
        if self._stage_two_active or not self.stage_two_enabled:
            return
        by_epoch = self.swa_start_epoch >= 0 and int(epoch) >= self.swa_start_epoch
        by_step = self.swa_start_step >= 0 and self.global_step >= self.swa_start_step
        if not (by_epoch or by_step):
            return
        self._stage_two_active = True
        self._stage_two_activated_epoch = int(epoch)
        self._stage_two_activated_step = int(self.global_step)
        self.a = self.stage_two_energy_weight
        self.b = self.stage_two_force_weight
        self.c = self.stage_two_stress_weight
        self._set_optimizer_lrs(self.swa_lr)
        self._swa_scheduler = SWALR(
            optimizer=self.optimizer,
            swa_lr=self.swa_lr,
            anneal_epochs=self.swa_anneal_epochs,
            anneal_strategy=self.swa_anneal_strategy,
        )
        log.info(
            "Stage Two/SWA activated at epoch=%d step=%d: weights E=%g F=%g S=%g lr=%g",
            epoch,
            self.global_step,
            self.a,
            self.b,
            self.c,
            self.swa_lr,
        )

    # ----------------------------------------------------------------- make_fx
    def _makefx_forward(self, pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell, strain=None):
        """make_fx-compiled ``(E_per_atom, dE/dpos)`` -- and, when ``strain`` is given,
        also ``dE/dstrain`` for the stress/virial loss (positions and cell deformed by
        ``I + sym(strain)``, traced at ``strain=0`` so the forward is undeformed).

        Compiled once per (training, input-shape) signature and cached outside the
        nn.Module tree. With a BucketBatchSampler the distinct shapes == the bucket
        count, so the compile count is bounded and each batch hits the cache."""
        from mace_ictd.training.makefx_compile import CompiledForceCache

        if getattr(self, "_makefx_cache", None) is None:
            object.__setattr__(self, "_makefx_cache",
                               CompiledForceCache(self.model, max_slots=self._makefx_max_slots))

        stress = strain is not None

        def _factory(model, *, training):
            if stress:
                def compute_fn(pos, strain, A, batch_idx, edge_src, edge_dst, edge_shifts, cell):
                    p = pos.detach().requires_grad_(True)
                    s = strain.detach().requires_grad_(True)
                    sym = 0.5 * (s + s.transpose(-1, -2))
                    defo = torch.eye(3, dtype=p.dtype, device=p.device) + sym  # [B,3,3]
                    pos_in = torch.einsum("ni,nij->nj", p, defo[batch_idx])
                    cell_in = torch.bmm(cell, defo)
                    e_atom = model(pos_in, A, batch_idx, edge_src, edge_dst, edge_shifts, cell_in)
                    if isinstance(e_atom, tuple):
                        e_atom = e_atom[0]
                    g = torch.autograd.grad(e_atom.sum(), [p, s], create_graph=training)
                    return e_atom, g[0], g[1]
                return compute_fn

            def compute_fn(pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell):
                p = pos.detach().requires_grad_(True)
                e_atom = model(p, A, batch_idx, edge_src, edge_dst, edge_shifts, cell)
                if isinstance(e_atom, tuple):
                    e_atom = e_atom[0]
                # E_per_atom.sum() == E_mean.sum() (the per-type offset is pos-independent),
                # so this gradient is the exact eager dE/dpos.
                grad = torch.autograd.grad(e_atom.sum(), p, create_graph=training)[0]
                return e_atom, grad
            return compute_fn

        if stress:
            example_inputs = (pos, strain, A, batch_idx, edge_src, edge_dst, edge_shifts, cell)
        else:
            example_inputs = (pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell)
        compiled = self._makefx_cache.get(
            example_inputs,
            training=bool(self.model.training),
            compute_fn_factory=_factory,
        )
        return compiled(*example_inputs)

    # --------------------------------------------------------------- per-batch
    def _unpack(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 11:
            (pos, A, batch_idx, force_ref, target_energies,
             edge_src, edge_dst, edge_shifts, cell, stress_ref, extras) = batch
        else:
            (pos, A, batch_idx, force_ref, target_energies,
             edge_src, edge_dst, edge_shifts, cell, stress_ref) = batch
            extras = {}
        fd, ld = self.dtype, torch.long
        dev = self.device
        return dict(
            pos=pos.to(dev, fd), A=A.to(dev, ld), batch_idx=batch_idx.to(dev, ld),
            force_ref=force_ref.to(dev, fd), target_energies=target_energies.to(dev, fd),
            edge_src=edge_src.to(dev, ld), edge_dst=edge_dst.to(dev, ld),
            edge_shifts=edge_shifts.to(dev, fd), cell=cell.to(dev, fd),
            stress_ref=stress_ref.to(dev, fd), extras=extras,
        )

    def _compute(self, batch, *, training):
        """Forward + force-autograd + energy/force loss. Mirrors the FSCETP baseline."""
        b = self._unpack(batch)
        pos, A, batch_idx = b["pos"], b["A"], b["batch_idx"]
        edge_src, edge_dst, edge_shifts, cell = b["edge_src"], b["edge_dst"], b["edge_shifts"], b["cell"]
        force_ref, target_energies = b["force_ref"], b["target_energies"]
        stress_ref, extras = b["stress_ref"], b["extras"]

        atom_mask = None
        if isinstance(extras, dict) and torch.is_tensor(extras.get("atom_mask", None)):
            atom_mask = extras["atom_mask"].to(device=self.device, dtype=pos.dtype).view(-1)

        def _mol_sum(x):
            if atom_mask is not None:
                x = x * atom_mask.view([-1] + [1] * (x.dim() - 1))
            return scatter(x, batch_idx, dim=0, reduce="sum")

        mapped_A = map_tensor_values(A, self.keys, self.values).to(self.device)
        E_offset_mol = _mol_sum(mapped_A)
        compute_stress = self.c > 0
        num_mol = cell.shape[0]

        use_makefx = (
            training
            and self.train_makefx_compile
            and not self._makefx_disabled
        )
        grad0 = grad_strain = None
        if use_makefx:
            try:
                if compute_stress:
                    strain0 = torch.zeros((num_mol, 3, 3), dtype=pos.dtype, device=self.device)
                    E_per_atom, grad0, grad_strain = self._makefx_forward(
                        pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell, strain=strain0)
                else:
                    E_per_atom, grad0 = self._makefx_forward(
                        pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell)
            except Exception as e:
                object.__setattr__(self, "_makefx_disabled", True)
                if self.require_train_makefx_compile:
                    raise RuntimeError(
                        "train_makefx_compile is required for this run, but tracing/compilation failed"
                    ) from e
                log.warning("train_makefx_compile failed (%s: %s); falling back to eager.",
                            type(e).__name__, e)
                use_makefx = False

        if not use_makefx:
            pos_leaf = pos.detach().requires_grad_(True)
            if compute_stress:
                # virial via the strain derivative: deform positions + cell by I + sym(strain),
                # evaluate at strain=0, read stress = dE/dstrain / V (FSCETP convention).
                strain = torch.zeros((num_mol, 3, 3), dtype=pos.dtype, device=self.device,
                                     requires_grad=True)
                sym = 0.5 * (strain + strain.transpose(-1, -2))
                defo = torch.eye(3, dtype=pos.dtype, device=self.device) + sym
                pos_in = torch.einsum("ni,nij->nj", pos_leaf, defo[batch_idx])
                cell_in = torch.bmm(cell, defo)
                grad_targets = [pos_leaf, strain]
            else:
                pos_in, cell_in, grad_targets = pos_leaf, cell, [pos_leaf]
            out = self.model(pos_in, A, batch_idx, edge_src, edge_dst, edge_shifts, cell_in)
            E_per_atom = out[0] if isinstance(out, tuple) else out
            E_conv_mol = _mol_sum(E_per_atom).squeeze(-1)
            E_mean = E_conv_mol + E_offset_mol
            grads = torch.autograd.grad(
                E_mean.sum(), grad_targets, create_graph=training, retain_graph=training)
            grad0 = grads[0]
            grad_strain = grads[1] if compute_stress else None
        else:
            E_conv_mol = _mol_sum(E_per_atom).squeeze(-1)
            E_mean = E_conv_mol + E_offset_mol

        f_pred = -grad0  # restore_force is identity in this pipeline
        force_ref_scaled = force_ref * self.force_shift_value

        if atom_mask is not None:
            mb = atom_mask.bool()
            f_pred_l, force_ref_l = f_pred[mb], force_ref_scaled[mb]
            num_atoms_per_mol = scatter(atom_mask, batch_idx, dim=0, reduce="sum")
        else:
            f_pred_l, force_ref_l = f_pred, force_ref_scaled
            num_atoms_per_mol = scatter(torch.ones_like(batch_idx), batch_idx, dim=0, reduce="sum")

        force_loss = self._loss_term(f_pred_l, force_ref_l)

        E_avg_pred = E_mean / num_atoms_per_mol
        target_energy_avg = target_energies / num_atoms_per_mol
        energy_loss = self._loss_term(E_avg_pred, target_energy_avg)

        total_loss = self.a * energy_loss + self.b * force_loss

        # Stress / virial loss: σ = dE/dε / V, SmoothL1 vs the reference stress (extra
        # term, total = a·E + b·F + c·σ). makefx path also produces dE/dε; the eager path
        # reads it from the second grad target.
        if compute_stress:
            volume = torch.abs(torch.det(cell)).clamp(min=1e-10)
            stress_pred = grad_strain / volume.view(-1, 1, 1)  # [B,3,3]
            stress_loss = self._loss_term(stress_pred, stress_ref)
            total_loss = total_loss + self.c * stress_loss
            with torch.no_grad():
                stress_rmse = torch.sqrt(self.criterion_2(stress_pred.reshape(-1), stress_ref.reshape(-1)))
        else:
            stress_loss = torch.zeros((), device=self.device)
            stress_rmse = torch.zeros((), device=self.device)

        with torch.no_grad():
            force_rmse = torch.sqrt(self.criterion_2(f_pred_l.reshape(-1), force_ref_l.reshape(-1)))
            energy_rmse_avg = torch.sqrt(self.criterion_2(E_avg_pred, target_energy_avg))
        return {
            "total_loss": total_loss,
            "energy_loss": energy_loss.detach(),
            "force_loss": force_loss.detach(),
            "stress_loss": stress_loss.detach(),
            "force_rmse": force_rmse,
            "energy_rmse_avg": energy_rmse_avg,
            "stress_rmse": stress_rmse,
        }

    def _loss_term(self, pred, target):
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        return smooth_l1_loss_stats(pred, target, beta=self.loss_beta)[0]

    @torch.no_grad()
    def _update_ema_state(self):
        state = self.model.state_dict()
        if self._ema_state is None:
            self._ema_state = {k: v.detach().clone() for k, v in state.items()}
            return
        decay = self.ema_decay
        one_minus = 1.0 - decay
        for key, val in state.items():
            cur = val.detach()
            old = self._ema_state.get(key)
            if old is None or old.shape != cur.shape:
                self._ema_state[key] = cur.clone()
                continue
            cur = cur.to(device=old.device, dtype=old.dtype)
            if torch.is_floating_point(old):
                old.mul_(decay).add_(cur, alpha=one_minus)
            else:
                old.copy_(cur)

    @torch.no_grad()
    def _update_swa_state(self):
        state = self.model.state_dict()
        if self._swa_state is None:
            self._swa_state = {k: v.detach().clone() for k, v in state.items()}
            self._swa_n = 1
            return
        self._swa_n += 1
        n = float(self._swa_n)
        for key, val in state.items():
            cur = val.detach()
            old = self._swa_state.get(key)
            if old is None or old.shape != cur.shape:
                self._swa_state[key] = cur.clone()
                continue
            cur = cur.to(device=old.device, dtype=old.dtype)
            if torch.is_floating_point(old):
                old.add_(cur - old, alpha=1.0 / n)
            else:
                old.copy_(cur)

    def _update_averaged_states(self, epoch: int):
        if self.ema_decay > 0.0 and self.global_step >= self.ema_start_step:
            self._update_ema_state()
        if self._stage_two_active:
            self._update_swa_state()

    def _reached_max_steps(self) -> bool:
        return self.max_steps is not None and self.global_step >= self.max_steps

    # ------------------------------------------------------------------- loops
    def train_epoch(self, epoch):
        if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(epoch)
        self.model.train()
        self._maybe_activate_stage_two(epoch)
        n_batches = len(self.train_loader)
        run = {"total_loss": 0.0, "energy_loss": 0.0, "force_loss": 0.0,
               "stress_loss": 0.0, "force_rmse": 0.0, "energy_rmse_avg": 0.0}
        t0 = time.time()
        seen = 0
        for i, batch in enumerate(self.train_loader):
            if self._reached_max_steps():
                break
            self._maybe_activate_stage_two(epoch)
            self.optimizer.zero_grad(set_to_none=True)
            out = self._compute(batch, training=True)
            loss = out["total_loss"]
            loss.backward()
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.global_step += 1
            self._step_scheduler_after_batch()
            self._update_averaged_states(epoch)
            for k in run:
                run[k] += float(out[k])
            seen += 1
            if self.log_interval and (i % self.log_interval == 0):
                lr = self.optimizer.param_groups[0]["lr"]
                s = f" S={float(out['stress_loss']):.4f}" if self.c > 0 else ""
                phase = "stage2" if self._stage_two_active else "stage1"
                log.info("epoch %d step %d batch %d/%d phase=%s loss=%.4f E=%.4f F=%.4f%s Frmse=%.4f lr=%.2e",
                         epoch, self.global_step, i, n_batches, phase, float(out["total_loss"]),
                         float(out["energy_loss"]), float(out["force_loss"]), s,
                         float(out["force_rmse"]), lr)
        seen = max(seen, 1)
        avg = {k: v / seen for k, v in run.items()}
        avg["time"] = time.time() - t0
        avg["steps"] = seen
        return avg

    @torch.no_grad()
    def _val_pass(self):
        # force needs grad of energy wrt pos even at eval -> enable grad locally.
        self.model.eval()
        run = {"total_loss": 0.0, "energy_loss": 0.0, "force_loss": 0.0,
               "stress_loss": 0.0, "force_rmse": 0.0, "energy_rmse_avg": 0.0}
        seen = 0
        for batch in self.val_loader:
            with torch.enable_grad():
                out = self._compute(batch, training=False)
            for k in run:
                run[k] += float(out[k])
            seen += 1
        seen = max(seen, 1)
        return {k: v / seen for k, v in run.items()}

    def fit(self, epochs=None):
        epochs = int(epochs) if epochs is not None else self.epochs
        best = math.inf
        for epoch in range(epochs):
            if self._reached_max_steps():
                break
            self._maybe_activate_stage_two(epoch)
            tr = self.train_epoch(epoch)
            sterm = f" S={tr['stress_loss']:.4f}" if self.c > 0 else ""
            phase = "stage2" if self._stage_two_active else "stage1"
            msg = (f"[epoch {epoch} step {self.global_step} {phase}] train loss={tr['total_loss']:.4f} "
                   f"E={tr['energy_loss']:.4f} F={tr['force_loss']:.4f}{sterm} "
                   f"Frmse={tr['force_rmse']:.4f} ({tr['time']:.1f}s)")
            if self.val_loader is not None:
                va = self._val_pass()
                msg += (f" | val loss={va['total_loss']:.4f} "
                        f"Frmse={va['force_rmse']:.4f} Ermse={va['energy_rmse_avg']:.4f}")
                cur = va["total_loss"]
            else:
                cur = tr["total_loss"]
            log.info(msg)
            print(msg, flush=True)
            self._step_scheduler_after_epoch(cur)
            if self.checkpoint_path is not None and cur < best:
                best = cur
                self.save_checkpoint(self.checkpoint_path, epoch=epoch)
            if self._reached_max_steps():
                break
        return best

    # -------------------------------------------------------------- checkpoint
    def _collect_arch_metadata(self):
        """Build the ``model_hyperparameters`` dict that the deploy-side
        ``LAMMPS_MLIAP_MFF.from_checkpoint`` reads to rebuild the architecture.

        Mirrors the FSCETP trainer's ``_collect_e3_arch_metadata`` for the keys that
        matter to the ictd-fix baseline: the ModelConfig snapshot plus the handful of
        live model attributes (num_interaction / avg_num_neighbors / ictd_fix_* etc.)
        whose defaults would otherwise be guessed wrong on reload."""
        base = self.model
        cfg = self.config
        meta = {}
        if cfg is not None:
            hp = {
                "dtype": str(getattr(cfg, "dtype", torch.float64)).replace("torch.", ""),
                "channel_in": int(getattr(cfg, "channel_in", 64)),
                "channel_in2": int(getattr(cfg, "channel_in2", 32)),
                "channel_in3": int(getattr(cfg, "channel_in3", 32)),
                "channel_in4": int(getattr(cfg, "channel_in4", 32)),
                "channel_in5": int(getattr(cfg, "channel_in5", 32)),
                "max_atomvalue": int(getattr(cfg, "max_atomvalue", 10)),
                "embedding_dim": int(getattr(cfg, "embedding_dim", 16)),
                "embed_size": list(getattr(cfg, "embed_size", [128, 128, 128])),
                "output_size": int(getattr(cfg, "output_size", 8)),
                "lmax": int(getattr(cfg, "lmax", 2)),
                "irreps_output_conv_channels": getattr(cfg, "irreps_output_conv_channels", None),
                "function_type": str(getattr(cfg, "function_type", "gaussian")),
                "max_radius": float(getattr(cfg, "max_radius", 5.0)),
                "max_radius_main": float(getattr(cfg, "max_radius_main", getattr(cfg, "max_radius", 5.0))),
                "number_of_basis": int(getattr(cfg, "number_of_basis", 8)),
                "number_of_basis_main": int(getattr(cfg, "number_of_basis_main", 8)),
                "num_layers": int(getattr(cfg, "num_layers", 1)),
                "main_hidden_sizes3": list(getattr(cfg, "main_hidden_sizes3", [64, 32])),
                "emb_number_main_2": list(getattr(cfg, "emb_number_main_2", [64, 64, 64])),
            }
            for attr in (
                "num_interaction", "invariant_channels",
                "ictd_tp_path_policy", "ictd_tp_max_rank_other", "ictd_save_tp_mode",
                "save_contraction_order", "save_multiple_mix_channels",
                "ictd_fix_route", "ictd_fix_product_backend",
                "ictd_fix_use_reduced_cg", "ictd_fix_edge_lmax",
                "ictd_fix_fusion_heads", "ictd_fix_fusion_head_weight_mode",
                "ictd_fix_interaction_attn_heads", "ictd_fix_interaction_scale",
                "ictd_fix_fusion_scale_init", "ictd_fix_gmix_gate_init",
                "ictd_fix_gmix_output_lmax", "avg_num_neighbors",
                "polynomial_cutoff_p", "long_range_mode", "angular_basis",
            ):
                if hasattr(base, attr):
                    val = getattr(base, attr)
                    if val is not None:
                        hp[attr] = val
            hp["angular_basis_folded_in_state_dict"] = bool(
                getattr(base, "angular_basis", "ictd") == "e3nn"
                and getattr(base, "_e3nn_folded", False)
            )
            hp["energy_output_scale_enabled"] = bool(
                getattr(base, "energy_output_scale_enabled", False)
            )
            scale_buf = getattr(base, "energy_output_scale", None)
            hp["energy_output_scale"] = (
                float(scale_buf.detach().cpu().item()) if torch.is_tensor(scale_buf) else 1.0
            )
            hp["energy_output_shift_enabled"] = bool(
                getattr(base, "energy_output_shift_enabled", False)
            )
            shift_buf = getattr(base, "energy_output_shift", None)
            hp["energy_output_shift"] = (
                float(shift_buf.detach().cpu().item()) if torch.is_tensor(shift_buf) else 0.0
            )
            # Explicit construction choices win (these are exactly what build_baseline_model
            # passed, so from_checkpoint rebuilds the same architecture).
            hp.update(self.extra_hparams)
            meta["model_hyperparameters"] = hp
        return meta

    def _training_metadata(self):
        return {
            "loss": self.loss_type,
            "loss_beta": self.loss_beta,
            "energy_weight": self.a,
            "force_weight": self.b,
            "stress_weight": self.c,
            "force_shift_value": self.force_shift_value,
            "optimizer": self.optimizer_type,
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "weight_decay": self.weight_decay,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_eps": self.adam_eps,
            "amsgrad": self.amsgrad,
            "lr_scheduler": self.lr_scheduler_kind,
            "warmup_batches": self.warmup_batches,
            "warmup_start_ratio": self.warmup_start_ratio,
            "warmup_start_lr": self._warmup_start_lr,
            "lr_factor": self.lr_factor,
            "scheduler_patience": self.scheduler_patience,
            "lr_scheduler_gamma": self.lr_scheduler_gamma,
            "lr_decay_step": self.lr_decay_step,
            "lr_decay_factor": self.lr_decay_factor,
            "epochs": self.epochs,
            "max_steps": self.max_steps,
            "max_grad_norm": self.max_grad_norm,
            "ema_decay": self.ema_decay,
            "ema_start_step": self.ema_start_step,
            "stage_two_enabled": self.stage_two_enabled,
            "stage_two_active": self._stage_two_active,
            "stage_two_activated_epoch": self._stage_two_activated_epoch,
            "stage_two_activated_step": self._stage_two_activated_step,
            "swa_start_epoch": self.swa_start_epoch,
            "swa_start_step": self.swa_start_step,
            "swa_lr": self.swa_lr,
            "swa_energy_weight": self.stage_two_energy_weight,
            "swa_force_weight": self.stage_two_force_weight,
            "swa_stress_weight": self.stage_two_stress_weight,
            "swa_anneal_epochs": self.swa_anneal_epochs,
            "swa_anneal_strategy": self.swa_anneal_strategy,
            "swa_n_averaged": self._swa_n,
            "checkpoint_state_source": self.checkpoint_state_source,
            "train_makefx_compile": self.train_makefx_compile,
            "makefx_max_slots": self._makefx_max_slots,
        }

    def _default_state_source(self):
        have_ema = isinstance(self._ema_state, dict)
        have_swa = isinstance(self._swa_state, dict)
        requested = self.checkpoint_state_source
        if requested == "auto":
            if have_ema:
                return "ema"
            if have_swa:
                return "swa"
            return "raw"
        if requested == "ema" and not have_ema:
            log.warning("checkpoint_state_source=ema requested but no EMA state exists; using raw")
            return "raw"
        if requested == "swa" and not have_swa:
            log.warning("checkpoint_state_source=swa requested but no SWA state exists; using raw")
            return "raw"
        return requested

    def save_checkpoint(self, path, *, epoch=0):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        default_state_source = self._default_state_source()
        ckpt = {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "e3trans_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "dtype": str(self.dtype).replace("torch.", ""),
            "max_radius": float(self.max_radius),
            "tensor_product_mode": self.tensor_product_mode,
            "atomic_energy_keys": self.keys.detach().cpu(),
            "atomic_energy_values": self.values.detach().cpu(),
            "a": self.a, "b": self.b, "c": self.c,
            "training_hyperparameters": self._training_metadata(),
            "default_state_source": default_state_source,
        }
        if isinstance(self._ema_state, dict):
            ckpt["e3trans_ema_state_dict"] = self._ema_state
        if isinstance(self._swa_state, dict):
            ckpt["e3trans_swa_state_dict"] = self._swa_state
            ckpt["swa_n_averaged"] = int(self._swa_n)
        ckpt.update(self._collect_arch_metadata())
        torch.save(ckpt, path)
        log.info("saved checkpoint -> %s (epoch %d step %d state=%s)",
                 path, epoch, self.global_step, default_state_source)
