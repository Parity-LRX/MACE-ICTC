"""Compact energy+force(+stress) trainer for MACE-ICTD (make_fx compile + size-bucketing).

This is the baseline energy+force training path extracted from the FSCETP trainer,
kept numerically faithful but stripped of the modes MACE-ICTD does not ship
(physical-tensor heads / multi-fidelity / external field / EMA / SWA / CUDA-graph /
compiled-autograd). What it preserves, verbatim in math:

  * energy:  ``E_mol = scatter_sum(E_per_atom, batch_idx) + scatter_sum(E0[A], batch_idx)``
             where ``E0`` is the per-type atomic-energy offset (the dataset stores
             energies already-baseline-subtracted; we add it back here).
  * forces:  ``f = -dE_mol/dpos`` via ``autograd.grad(E_mol.sum(), pos, create_graph=training)``.
             ``E0`` is pos-independent, so this equals ``-dE_conv/dpos``.
  * stress:  optional (``stress_weight > 0``) -- ``sigma = dE/dstrain / V`` via the
             strain-derivative trick: deform positions+cell by ``I + sym(strain)`` and
             differentiate w.r.t. ``strain`` at ``strain=0`` (FSCETP convention).
  * loss:    per-atom-normalized ``SmoothL1(beta=0.5)`` on energy + ``SmoothL1(beta=0.5)``
             on force (+ ``SmoothL1`` on stress), ``force_ref *= force_shift_value``;
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
        atomic_energy_keys=None,
        atomic_energy_values=None,
        # optimization
        learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-6,
        weight_decay: float = 0.0,
        optimizer_type: str = "adamw",
        amsgrad: bool = False,
        lr_scheduler: str = "cosine",
        warmup_batches: int = 0,
        warmup_start_ratio: float = 0.1,
        lr_decay_step: int = 1000,
        lr_decay_factor: float = 0.98,
        epochs: int = 1,
        max_grad_norm: float | None = None,
        # make_fx
        train_makefx_compile: bool = False,
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

        keys = atomic_energy_keys if atomic_energy_keys is not None else _DEFAULT_E0_KEYS
        values = atomic_energy_values if atomic_energy_values is not None else _DEFAULT_E0_VALUES
        self.keys = torch.as_tensor(list(keys), dtype=torch.long, device=self.device)
        self.values = torch.as_tensor(list(values), dtype=self.dtype, device=self.device)
        self.criterion_2 = RMSELoss()

        self.epochs = int(epochs)
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
        self._makefx_max_slots = int(makefx_max_slots)
        self._makefx_disabled = False
        object.__setattr__(self, "_makefx_cache", None)

        self._build_optimizer(optimizer_type, learning_rate, weight_decay, amsgrad)
        self._build_scheduler(
            lr_scheduler, warmup_batches, warmup_start_ratio,
            lr_decay_step, lr_decay_factor, min_learning_rate,
        )

    # ------------------------------------------------------------------ setup
    def _build_optimizer(self, optimizer_type, lr, weight_decay, amsgrad):
        params = list(self.model.parameters())
        kind = str(optimizer_type).lower()
        common = dict(lr=lr, betas=(0.9, 0.999), eps=1e-8,
                      weight_decay=weight_decay, amsgrad=bool(amsgrad))
        if kind == "adam":
            self.optimizer = torch.optim.Adam(params, **common)
        else:
            self.optimizer = torch.optim.AdamW(params, **common)

    def _build_scheduler(self, kind, warmup_batches, warmup_start_ratio,
                         decay_step, decay_factor, min_lr):
        kind = str(kind).lower()
        steps_per_epoch = max(1, len(self.train_loader))
        total_steps = max(1, self.epochs * steps_per_epoch)
        warmup_batches = int(max(0, warmup_batches))

        def _warmup_lambda(step):
            if warmup_batches <= 0:
                return 1.0
            return warmup_start_ratio + (1.0 - warmup_start_ratio) * min(step / warmup_batches, 1.0)

        if kind == "none":
            decay = None
        elif kind == "step":
            decay = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=int(decay_step), gamma=float(decay_factor))
        else:  # cosine (default)
            decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, total_steps - warmup_batches), eta_min=min_lr)

        if warmup_batches > 0:
            warmup = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _warmup_lambda)
            if decay is None:
                self.scheduler = warmup
            else:
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer, schedulers=[warmup, decay], milestones=[warmup_batches])
        else:
            self.scheduler = decay  # may be None

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

        force_loss = smooth_l1_loss_stats(f_pred_l, force_ref_l, beta=0.5)[0]

        E_avg_pred = E_mean / num_atoms_per_mol
        target_energy_avg = target_energies / num_atoms_per_mol
        energy_loss = smooth_l1_loss_stats(E_avg_pred, target_energy_avg, beta=0.5)[0]

        total_loss = self.a * energy_loss + self.b * force_loss

        # Stress / virial loss: σ = dE/dε / V, SmoothL1 vs the reference stress (extra
        # term, total = a·E + b·F + c·σ). makefx path also produces dE/dε; the eager path
        # reads it from the second grad target.
        if compute_stress:
            volume = torch.abs(torch.det(cell)).clamp(min=1e-10)
            stress_pred = grad_strain / volume.view(-1, 1, 1)  # [B,3,3]
            stress_loss = smooth_l1_loss_stats(stress_pred, stress_ref, beta=0.5)[0]
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

    # ------------------------------------------------------------------- loops
    def train_epoch(self, epoch):
        if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(epoch)
        self.model.train()
        n_batches = len(self.train_loader)
        run = {"total_loss": 0.0, "energy_loss": 0.0, "force_loss": 0.0,
               "stress_loss": 0.0, "force_rmse": 0.0, "energy_rmse_avg": 0.0}
        t0 = time.time()
        seen = 0
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)
            out = self._compute(batch, training=True)
            loss = out["total_loss"]
            loss.backward()
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            for k in run:
                run[k] += float(out[k])
            seen += 1
            if self.log_interval and (i % self.log_interval == 0):
                lr = self.optimizer.param_groups[0]["lr"]
                s = f" S={float(out['stress_loss']):.4f}" if self.c > 0 else ""
                log.info("epoch %d batch %d/%d loss=%.4f E=%.4f F=%.4f%s Frmse=%.4f lr=%.2e",
                         epoch, i, n_batches, float(out["total_loss"]),
                         float(out["energy_loss"]), float(out["force_loss"]), s,
                         float(out["force_rmse"]), lr)
        seen = max(seen, 1)
        avg = {k: v / seen for k, v in run.items()}
        avg["time"] = time.time() - t0
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
            tr = self.train_epoch(epoch)
            sterm = f" S={tr['stress_loss']:.4f}" if self.c > 0 else ""
            msg = (f"[epoch {epoch}] train loss={tr['total_loss']:.4f} "
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
            if self.checkpoint_path is not None and cur < best:
                best = cur
                self.save_checkpoint(self.checkpoint_path, epoch=epoch)
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
                "polynomial_cutoff_p", "long_range_mode",
            ):
                if hasattr(base, attr):
                    val = getattr(base, attr)
                    if val is not None:
                        hp[attr] = val
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

    def save_checkpoint(self, path, *, epoch=0):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        ckpt = {
            "epoch": int(epoch),
            "e3trans_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "dtype": str(self.dtype).replace("torch.", ""),
            "max_radius": float(self.max_radius),
            "tensor_product_mode": self.tensor_product_mode,
            "atomic_energy_keys": self.keys.detach().cpu(),
            "atomic_energy_values": self.values.detach().cpu(),
            "a": self.a, "b": self.b, "c": self.c,
        }
        ckpt.update(self._collect_arch_metadata())
        torch.save(ckpt, path)
        log.info("saved checkpoint -> %s (epoch %d)", path, epoch)
