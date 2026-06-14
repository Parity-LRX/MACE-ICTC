# MACE-ICTD 使用说明书

本文档说明 MACE-ICTD 代码库的整体结构、常用模式、训练、MACE 原生模型转换、AOTInductor 导出、LAMMPS 部署、benchmark 和测试方法。

English version: [USER_MANUAL.md](USER_MANUAL.md)

## 1. 这个仓库包含什么

MACE-ICTD 是一个独立的 MACE 实现：它把 MACE 的等变特征从原生 e3nn 球谐基表示改写到 ICTD（Irreducible Cartesian Tensor Decomposition，不可约笛卡尔张量分解）基中，并带有训练、转换、导出和 LAMMPS 部署栈。

核心技术点是 ICTD basis construction：用固定的 `Q`/`U` 算子把 MACE/e3nn 的角向代数重写成不可约笛卡尔张量块，同时保留原始 MACE 的 interaction/readout 语义。这个仓库重点解决 basis conversion、原生 `mace-torch` 数值对应和部署。

主要目录：

- `mace_ictd/models/`：`PureCartesianICTDFix` 主模型、ICTD irreps、tensor product、MACE-compatible symmetric contraction、radial basis、ZBL、long-range 等。
- `mace_ictd/training/`：energy/force/stress trainer，以及 `make_fx` + Inductor 的 force step 编译。
- `mace_ictd/data/`：extended XYZ 解析、H5 dataset、padding、bucket sampler、collate。
- `mace_ictd/cli/`：训练、MACE 转换、AOTI 导出、TorchScript 导出、LAMMPS helper。
- `mace_ictd/interfaces/`：checkpoint 加载、LAMMPS MLIAP wrapper、部署兼容逻辑。
- `mace_ictd/evaluation/`：ASE calculator。
- `mace_ictd/bench/`：MACE-ICTD 和原生 `mace-torch` 的 benchmark。
- `mace_ictd/test/`：数值一致性和 smoke tests。
- `lammps_user_mfftorch/`：LAMMPS `USER-MFFTORCH` C++ package。

核心模型 forward 接口：

```python
model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
```

含义：

- `pos`：原子坐标，形状 `[N, 3]`。
- `A`：原子序数，不是 species index。
- `batch`：每个原子属于哪个 graph，形状 `[N]`。
- `edge_src`, `edge_dst`：有向边索引。
- `edge_shifts`：周期性 image shift，形状 `[E, 3]`。
- `cell`：晶胞，形状 `[B, 3, 3]`。
- 返回值：逐原子 interaction energy，通常形状 `[N, 1]`。

E0 原子参考能通常由训练/导出 wrapper 在模型外部处理。

## 2. 安装

最小安装：

```bash
cd /path/to/MACE-ICTD
pip install -e .
```

可选依赖：

```bash
pip install -e ".[pyg]"   # torch-scatter / torch-cluster 加速
pip install -e ".[cue]"   # cuEquivariance backend
pip install -e ".[e0]"    # 读取 fitted E0 CSV 所需 pandas
pip install -e ".[full]"  # 所有可选依赖
```

版本建议：

- Python >= 3.9。
- PyTorch >= 2.4；`make_fx` 和 AOTInductor 路径建议 PyTorch >= 2.7。
- `e3nn < 0.6`，用于兼容当前 `mace-torch`。
- cuEq、AOTI benchmark 和真实 GPU 训练需要 CUDA。

可选 ICTD tensor-product C++/CUDA extension：

```bash
MFF_BUILD_ICTD_TP_EXT=1 pip install -e .
```

CUDA extension：

```bash
MFF_BUILD_ICTD_TP_EXT=1 MFF_BUILD_ICTD_TP_CUDA=1 pip install -e .
```

不编译 extension 时，Python fallback 仍然可用。

## 3. 命令行工具

安装后会有这些 console scripts：

| 命令 | Python 入口 | 用途 |
|---|---|---|
| `mff-convert-mace` | `mace_ictd.cli.convert_mace` | 把原生 `mace-torch` `ScaleShiftMACE` 转成 MACE-ICTD checkpoint。 |
| `mff-export-aoti` | `mace_ictd.cli.export_aoti_core` | 导出 AOTInductor `.pt2` core，用于 Python/C++/LAMMPS 部署。 |
| `mff-export-core` | `mace_ictd.cli.export_libtorch_core` | 导出 TorchScript core，主要用于旧的 LibTorch 部署。 |
| `mff-lammps` | `mace_ictd.cli.lammps_interface` | 生成 LAMMPS 部署 helper。 |

也可以直接用 module 方式：

```bash
python -m mace_ictd.cli.train --help
python -m mace_ictd.cli.convert_mace --help
python -m mace_ictd.cli.export_aoti_core --help
```

## 4. 核心概念

### 4.1 ICTD 基和 e3nn/MACE 基

原生 MACE 使用 e3nn 球谐基。MACE-ICTD 在内部把等变特征存成 ICTD 笛卡尔基。两者之间由固定的正交矩阵 `Q_l` 相连。

能量、力、virial 这类物理不变量不应该依赖基选择。等变中间特征如果需要转回原生 MACE/e3nn convention，可以用：

```python
model.to_mace_basis(x)
model.to_ictd_basis(x)
```

或者直接使用 `mace_ictd.mace_basis` 里的底层函数。

### 4.2 `angular_basis`

`angular_basis` 控制模型内部用哪个角向基计算：

| 值 | 含义 | 适用场景 |
|---|---|---|
| `ictd` | 默认 ICTD 内部基。 | parity 基准、bridge-U、最保守路径。 |
| `e3nn` | 把固定角向算子一次性 fold 到 e3nn/MACE 基。 | cuEq product 性能路径；AOTI 导出配合 `--cueq-product`。 |

限制：

- `ictd-bridge-u` 没有 e3nn fold path。直接对 bridge-U 请求 `angular_basis=e3nn` 时，导出逻辑会保留 `ictd` 并打印 warning。
- `cueq` product 支持 `angular_basis=e3nn`。
- 训练时如果使用 `angular_basis=e3nn`，固定 buffer 会在第一次 forward 后处于已 fold 状态；checkpoint reload 会恢复 Q blocks 和 product runtime flag，避免 double-fold。

### 4.3 Product backend

| Backend | 含义 | 推荐用途 |
|---|---|---|
| `ictd-bridge-u` | 使用 MACE/e3nn symmetric-contraction U，并把 ICTD/e3nn basis bridge fold 到 U 里。 | MACE parity、原生 MACE 转换、高 `max_ell` 稳妥路径。 |
| `cueq` | 用 cuEquivariance 做 product/symmetric contraction。 | 性能训练和推理，尤其配合 `--angular-basis e3nn`。 |
| `native-mace` | product block 内部直接调用 MACE 的 native symmetric contraction。 | debug/reference。 |
| `ictd-pure-u` | 直接用 ICTD 生成的 U。 | 诊断路径，不是当前高 `max_ell` 主生产路径。 |

### 4.4 `use_reduced_cg`

`--use-reduced-cg` 是 product/symmetric contraction 的结构选项，会改变 CG/path layout 和权重形状。

规则：

- 转换已有原生 MACE checkpoint 时，必须跟随原模型的 `mace_model.use_reduced_cg`，不要手动猜。
- 原生 `mace-torch` 训练也有这个选项，名字是 `--use_reduced_cg`。
- 从头训练 MACE-ICTD 时，只有在你明确要 reduced-CG 架构时才开。
- 它不是稳定训练吞吐的主要加速开关。最近 4090 上 `cueq + angular_basis=e3nn + make_fx` 的完整训练 step 测试里，稳定 step time 变化大约在 -1% 到 +2%，compile time 改善更明显。

## 5. 应该选择哪种模式

| 目标 | 推荐模式 |
|---|---|
| 原生 MACE 转换 / 数值 parity 基准 | `ictd-bridge-u`，`angular_basis=ictd`，通常 `dtype=float64`。 |
| 从头训练但希望模型语义贴近原生 MACE | `ictd-bridge-u`，`function-type=bessel`，保留 MACE-style ScaleShift。 |
| 高吞吐训练 | `cueq`，`angular_basis=e3nn`，`--train-makefx-compile`，配合 bucket。 |
| 从 ICTD checkpoint 导出高吞吐 AOTI | `mff-export-aoti --cueq-product --angular-basis e3nn`，前提是部署环境能加载 cuEq custom op。 |
| 最保守部署 | 不做 cuEq product replacement，保留 checkpoint 原本 basis 或 `ictd`。 |

完整 parity 训练命令：

```bash
python -m mace_ictd.cli.train \
  --data-dir DATA \
  --train-prefix train \
  --val-prefix val \
  --seed 123 \
  --channels 64 \
  --lmax 2 \
  --max-ell 2 \
  --num-interaction 2 \
  --correlation 2 \
  --function-type bessel \
  --product-backend ictd-bridge-u \
  --scaling rms_forces_scaling \
  --epochs 300 \
  --max-steps 200000 \
  --batch-size 4 \
  --loss smooth_l1 \
  --loss-beta 0.5 \
  --energy-weight 1.0 \
  --force-weight 10.0 \
  --stress-weight 0.0 \
  --optimizer adamw \
  --lr 0.001 \
  --min-lr 0.000001 \
  --weight-decay 0.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.999 \
  --adam-eps 1e-8 \
  --lr-scheduler plateau \
  --lr-factor 0.8 \
  --scheduler-patience 50 \
  --warmup-batches 1000 \
  --warmup-start-ratio 0.1 \
  --swa \
  --start-swa 225 \
  --swa-lr 0.0001 \
  --swa-energy-weight 1000.0 \
  --swa-force-weight 100.0 \
  --swa-stress-weight 0.0 \
  --ema-decay 0.0 \
  --checkpoint-state-source swa \
  --device cuda \
  --dtype float64 \
  --checkpoint model_bridge_u.pth
```

完整高性能训练命令：

```bash
python -m mace_ictd.cli.train \
  --data-dir DATA \
  --train-prefix train \
  --val-prefix val \
  --seed 123 \
  --channels 64 \
  --lmax 2 \
  --max-ell 2 \
  --num-interaction 2 \
  --correlation 2 \
  --function-type bessel \
  --product-backend cueq \
  --angular-basis e3nn \
  --train-makefx-compile \
  --makefx-buckets 6 \
  --makefx-max-slots 8 \
  --pad-nodes-to-max \
  --pad-edges-to-max \
  --scaling rms_forces_scaling \
  --epochs 300 \
  --max-steps 200000 \
  --batch-size 8 \
  --loss smooth_l1 \
  --loss-beta 0.5 \
  --energy-weight 1.0 \
  --force-weight 10.0 \
  --stress-weight 0.0 \
  --optimizer adamw \
  --lr 0.001 \
  --min-lr 0.000001 \
  --weight-decay 0.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.999 \
  --adam-eps 1e-8 \
  --lr-scheduler cosine \
  --warmup-batches 1000 \
  --warmup-start-ratio 0.1 \
  --swa \
  --start-swa 225 \
  --swa-lr 0.0001 \
  --swa-energy-weight 1000.0 \
  --swa-force-weight 100.0 \
  --swa-stress-weight 0.0 \
  --ema-decay 0.999 \
  --ema-start-step 1000 \
  --checkpoint-state-source swa \
  --device cuda \
  --dtype float32 \
  --checkpoint model_cueq_e3nn_makefx.pth
```

上面的数值是安全起点，不是对所有化学体系都最优的超参。要和原生 MACE
训练做严格对比，需要同时匹配 dataset split、seed、loss weight、optimizer、
scheduler、batch 构造、dtype、ScaleShift 和 E0 设置。

注意：两个 trainer 从随机初始化开始独立训练，不能因为函数形式一致就保证每一步 bitwise 一样；初始化、数据顺序、optimizer、CUDA kernel、dtype 和 loss 累加顺序都会影响训练轨迹。

## 6. 数据流程

训练读取预处理后的 H5：

```text
DATA/
  processed_train.h5
  processed_val.h5       # 可选
  processed_train.counts.npz / bucket sidecar
```

`mace_ictd.data.preprocessing` 支持 extended XYZ 风格数据：

- species / atomic number，
- 坐标，
- 力，
- 总能量，
- cell 和 PBC，
- 可选 stress 或 virial。

预处理是 Python API：

```python
from mace_ictd.data.preprocessing import save_to_h5_parallel

save_to_h5_parallel(
    prefix="train",
    max_radius=5.0,
    num_workers=8,
    data_dir="DATA",
)
```

训练 dataset 是 `mace_ictd.data.datasets.H5Dataset`，batching 使用 `mace_ictd.data.collate.collate_fn_h5`。

使用 `make_fx` 训练时推荐 bucket：

```bash
--train-makefx-compile --makefx-buckets 6
```

这样相近的 atom/edge count 会共用 compile，避免每个 shape 编译一次。

## 7. 训练细节

训练 CLI：

```bash
python -m mace_ictd.cli.train --help
```

关键结构参数：

| 参数 | 含义 |
|---|---|
| `--channels` | hidden channel 数。 |
| `--lmax` | hidden feature 的最高角动量。 |
| `--max-ell` | edge spherical harmonics cutoff；默认等于 `--lmax`。 |
| `--num-interaction` | MACE interaction/product block 数。 |
| `--correlation` | MACE product correlation order，也就是 `save_contraction_order`。 |
| `--function-type` | radial basis；要贴近 MACE 通常用 `bessel`。 |
| `--product-backend` | product backend，常用 `ictd-bridge-u` 或 `cueq`。 |
| `--angular-basis` | `ictd` 或 `e3nn`；性能路径中 `cueq` 可用 `e3nn`。 |
| `--use-reduced-cg` | reduced-CG product layout；转换原生 MACE 时必须跟原模型一致。 |
| `--long-range-mode` | 可选 learned scalar reciprocal correction；用 `reciprocal-spectral-v1` 打开，默认 `none`。 |

优化、step 和 seed：

| 参数 | 含义 |
|---|---|
| `--seed` | 设置 Python、NumPy、PyTorch、DataLoader shuffle 和 bucket sampler 的随机种子；这不等于强制 CUDA kernel 完全 deterministic。 |
| `--epochs` | 最大 epoch 数。 |
| `--max-steps` | 可选的 optimizer step 上限；达到后即使在 epoch 中间也会停止。 |
| `--batch-size` | 每个 batch 或 bucketed batch 的 graph 数。 |
| `--optimizer` | `adamw` 或 `adam`。 |
| `--lr`、`--min-lr`、`--weight-decay` | 初始/最大学习率、scheduler 的 LR 下限、AdamW weight decay。LR 会限制在 `[--min-lr, --lr]`。 |
| `--adam-beta1`、`--adam-beta2`、`--adam-eps`、`--amsgrad` | Adam/AdamW 的数值参数。 |
| `--lr-scheduler` | `plateau`、`exp`、`cosine`、`step` 或 `none`；也接受 `ReduceLROnPlateau` 和 `ExponentialLR` 别名。 |
| `--warmup-batches`、`--warmup-start-ratio` | linear warmup 长度和初始 LR 倍率。 |
| `--lr-factor`、`--scheduler-patience` | plateau scheduler 的 factor 和 patience。 |
| `--lr-scheduler-gamma` | exp scheduler 的 gamma。 |
| `--lr-decay-step`、`--lr-decay-factor` | legacy step scheduler 参数。 |
| `--max-grad-norm` | 可选 gradient clipping 阈值。 |

Loss：

```text
total = energy_weight * loss(E)
      + force_weight  * loss(F)
      + stress_weight * loss(stress)
```

| 参数 | 含义 |
|---|---|
| `--loss` | `smooth_l1` 默认，或者 `mse`。 |
| `--loss-beta` | `--loss smooth_l1` 时 energy/force/stress 共用的 SmoothL1 beta。 |
| `--energy-weight`、`--force-weight`、`--stress-weight` | 总 loss 里的权重；默认 `--stress-weight 0`，即不训练 stress。 |
| `--force-shift-value` | 进入 force loss 前乘到参考 force 上；除非复现 legacy run，否则保持 `1.0`。 |

打开 stress 后，stress 通过 strain derivative 计算。

MACE-style Stage Two / SWA：

| 参数 | 含义 |
|---|---|
| `--swa`、`--stage-two` | 开启 mace-torch 风格 Stage Two：切换 loss weights、降低 LR、保存平均权重。 |
| `--start-swa`、`--start-stage-two`、`--swa-start-epoch` | 从哪个 epoch 开始 Stage Two。如果只开 `--swa` 但不指定 start，默认 `max(1, epochs * 3 // 4)`，和 mace-torch 一致。 |
| `--swa-start-step` | 可选的 global step 触发。 |
| `--swa-lr`、`--stage-two-lr` | Stage Two/SWA LR；必须满足 `--min-lr <= --swa-lr <= --lr`。 |
| `--swa-energy-weight`、`--swa-force-weight`、`--swa-stress-weight` | Stage Two loss 权重。MACE-like energy/force 默认是 `1000` 和 `100`；stress 默认在 stress 关闭时为 `0`，stress 打开时为 `10`。 |
| `--swa-anneal-epochs`、`--swa-anneal-strategy` | SWALR annealing 控制。 |
| `--ema-decay` | 大于 `0` 时开启 EMA，例如 `0.999`；保存为 `e3trans_ema_state_dict`。 |
| `--ema-start-step` | 从哪个 global optimizer step 开始更新 EMA。 |
| `--checkpoint-state-source` | `auto`、`raw`、`ema` 或 `swa`。部署加载器读取 `default_state_source`；`auto` 优先 EMA，其次 SWA，最后 raw。 |

现在这里复刻的是和本 trainer 相关的 mace-torch Stage Two 行为：到 Stage Two
边界后切换 loss weights，暂停主 LR scheduler，把 optimizer LR 切到 `--swa-lr`，
并把平均权重保存为 `e3trans_swa_state_dict`。

ScaleShift：

- 默认 `--scaling rms_forces_scaling`。
- 也支持 `std_scaling`、`no_scaling`。
- 可用 `--atomic-inter-scale` 和 `--atomic-inter-shift` 显式覆盖。
- `--no-atomic-inter-shift` 表示保留 scale，但 interaction energy shift 设为 0。

E0：

- 用 `--atomic-energy-keys` 和 `--atomic-energy-values` 显式传入。
- 如果省略，训练 CLI 使用内置 H/C/N/O 默认值。
- 部署时如果需要绝对能量，AOTI export 用 `--embed-e0`。

Long-range correction：

```bash
python -m mace_ictd.cli.train \
  --data-dir DATA \
  --channels 64 --lmax 2 --num-interaction 2 \
  --long-range-mode reciprocal-spectral-v1 \
  --long-range-boundary periodic \
  --long-range-reciprocal-backend direct_kspace \
  --long-range-kmax 4 \
  --long-range-source-channels 1 \
  --checkpoint model_lr.pth
```

这会打开当前支持的 long-range 模块 `reciprocal-spectral-v1`。它从最终逐原子的 invariant
descriptor 预测 learned latent scalar sources，并添加 reciprocal-space energy term。贡献项接近
zero-init，所以开启后初始行为接近 short-range baseline，再通过同一个 energy/force/stress loss 学到修正。
它不是固定电荷的解析 Ewald 项，也不需要显式 charge label。

常用选项：

| 参数 | 含义 |
|---|---|
| `--long-range-boundary periodic` | 完全周期 reciprocal solve；`direct_kspace` 必须用这个。 |
| `--long-range-boundary slab` | slab 边界；需要配合 `--long-range-reciprocal-backend mesh_fft`。 |
| `--long-range-reciprocal-backend direct_kspace` | 直接 k-space sum，保留在模型/导出 core 内；适合小 `kmax` 测试。 |
| `--long-range-reciprocal-backend mesh_fft` | FFT mesh 路径，适合更大 periodic/slab 系统。 |
| `--long-range-kmax` | `direct_kspace` 的整数 k-lattice cutoff。 |
| `--long-range-mesh-size` | `mesh_fft` 的 mesh resolution。 |
| `--long-range-source-channels` | latent scalar source channel 数。 |
| `--no-long-range-neutralize` | 关闭每个 graph 的 source neutralization；通常保持默认开启。 |
| `--long-range-include-k0` | 包含 k=0 mode；neutralized source 下通常保持关闭。 |
| `--long-range-green-mode` | `poisson` 或 `learned_poisson`。 |

checkpoint 会把 long-range 超参数写进 `model_hyperparameters`，所以
`LAMMPS_MLIAP_MFF.from_checkpoint` 和 `mff-export-aoti --checkpoint model_lr.pth ...` 会重建同一个架构。
原生 MACE 转换不会给已经训练好的 MACE checkpoint 自动加 long-range module；需要这个修正时，应在
MACE-ICTD 里用 `--long-range-mode reciprocal-spectral-v1` 训练或 fine-tune。

## 8. 原生 MACE 模型转换

用原生 `mace-torch` 训练好模型后，走这个转换路径。输入必须是 torch 保存的
`ScaleShiftMACE` 对象，不只是 raw `state_dict`：

```bash
mff-convert-mace \
  --mace-model mace.model \
  --out mace_ictd.pth \
  --product-backend ictd-bridge-u \
  --dtype float64 \
  --device cpu
```

推荐的数值对应路径：

- 转换时使用 `--product-backend ictd-bridge-u`。
- 用 `--dtype float64` 做最严格的 energy/force 对比。
- 导出时不替换 product backend。
- 如果部署的 `.pt2` 需要直接返回绝对能量，用 `--embed-e0`。

```bash
mff-export-aoti \
  --checkpoint mace_ictd.pth \
  --elements H,C,N,O \
  --out mace_ictd.pt2 \
  --dynamic \
  --embed-e0
```

如果要在推理端替换成 cuEq product 并做 e3nn fold：

```bash
mff-export-aoti \
  --checkpoint mace_ictd.pth \
  --elements H,C,N,O \
  --out mace_ictd_cueq_e3nn.pt2 \
  --dynamic \
  --cueq-product \
  --angular-basis e3nn
```

cuEq 导出路径会替换 product block，并把兼容的角向算子 fold 到 e3nn convention。部署时需要注册
cuEquivariance custom ops。如果当前 exporter/runtime 不能把 `--embed-e0` 和 `--cueq-product`
同时使用，就把 E0 留在 core 外部处理。

支持的版本和模型变体：

- converter 面向本仓库测试和 benchmark 使用的 `mace-torch` `ScaleShiftMACE` 对象结构；验证环境是
  `mace==0.3.16` 和 `e3nn<0.6`。
- 更新的 `mace-torch` 版本如果保存对象结构和 `extract_config_mace_model` 输出仍兼容，可能可以工作，
  但这里不自动承诺覆盖。
- checkpoint 加载本身受 Python pickle 兼容性影响。很老的 MACE 预训练模型可能需要对应历史版本的
  `mace-torch`/`e3nn` 环境先成功 load；只要模型对象能 load，后续是否能转换由下面的结构检查决定。
- 这不是任意 native-MACE-like 实现、raw state dict 或自定义 research fork 的通用转换器；它要求保存出来的是兼容的 `ScaleShiftMACE`。

转换支持范围是故意收紧的；不支持的原生 MACE 变体会报错，而不是静默转换成错误模型。当前主要假设包括：

- `ScaleShiftMACE`，
- Bessel radial basis，
- `radial_MLP=[64,64,64]`，
- hidden irreps 使用 MACE parity，`l=0..L` 连续，且各 `l` channel 数一致，
- 每层 correlation 一致，
- 无 pair repulsion / distance transform，
- SiLU MACE-style scalar readout，`MLP_irreps=16x0e`，
- 第一层 interaction 是 `RealAgnosticInteractionBlock` 或 `RealAgnosticResidualInteractionBlock`，
- 后续 interaction 是 `RealAgnosticResidualInteractionBlock`，
- `max_ell >= hidden_irreps.lmax`，
- `num_interactions >= 2`。

backend 差异：

- `ictd-bridge-u`：推荐转换 backend；通过 ICTD/e3nn basis bridge fold MACE/e3nn 的 U convention，是主数值对应路径。
- `native-mace`：debug/reference backend，用于诊断 MACE 侧 contraction 行为。
- `cueq`：性能 product backend，尤其适合导出时配合 `--cueq-product --angular-basis e3nn`。
- `ictd-pure-u`：直接用 ICTD 生成 U 的诊断路径，不是原生 MACE exact conversion 主路径。

converter 会读取 `mace_model.use_reduced_cg` 并用同样配置重建 MACE-ICTD。用户转换原生 MACE 模型时不应该手动猜这个选项。
转换会保留原生 MACE 架构，不会自动添加 learned long-range module。

## 9. AOTInductor 导出

基础导出：

```bash
mff-export-aoti \
  --checkpoint model.pth \
  --elements H,C,N,O \
  --out model.pt2 \
  --dynamic \
  --embed-e0
```

性能导出：

```bash
mff-export-aoti \
  --checkpoint model.pth \
  --elements H,C,N,O \
  --out model_cueq_e3nn.pt2 \
  --dynamic \
  --cueq-product \
  --angular-basis e3nn \
  --assume-cutoff-edges \
  --preserve-edge-order \
  --fuse-selector-message-linear \
  --inductor-max-autotune
```

性能导出的 cuEq 路径里，如果当前 exporter/runtime 不能把 `--embed-e0` 和 product replacement
组合使用，就把 atomic E0 留在导出 core 外处理。需要 E0-embedded 绝对能量 `.pt2` 时，最简单稳妥的是使用上面的 bridge-U 保守导出。

常用导出选项：

| 选项 | 含义 |
|---|---|
| `--dynamic` | 尽量导出动态 atom/edge 维度。 |
| `--static-n` | 固定 atom count，适合固定 N 的 MD 或 dynamic export 不稳定时。 |
| `--embed-e0` | 把 E0 原子参考能嵌进导出 core。 |
| `--cueq-product` | 导出时把 product block 替换成 cuEq product。 |
| `--angular-basis e3nn` | 对可 fold 的 product backend 做 e3nn basis fold。 |
| `--assume-cutoff-edges` | 假设调用方已经筛好 cutoff 内边，跳过模型内部 edge mask。 |
| `--preserve-edge-order` | 假设调用方 edge order 稳定，跳过模型内部按 `edge_dst` 排序。 |
| `--fuse-selector-message-linear` | 在支持的位置融合 selector/message linear。 |
| `--inductor-max-autotune` | 编译更慢，可能得到更快 kernel；必须 benchmark 后再用于生产。 |

关于 `strict=False`：

- exporter 会优先尝试 strict export。
- 如果 strict 失败但属于 exporter 限制，会 fallback 到 non-strict。
- non-strict 不等于默认正确；导出流程仍然会 compile/load `.pt2` 并做数值比较。

## 10. ASE 和 Python 推理

ASE wrapper 是 `mace_ictd.evaluation.calculator.MyE3NNCalculator`。

示例：

```python
import torch
from mace_ictd.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
from mace_ictd.evaluation.calculator import MyE3NNCalculator

wrapper = LAMMPS_MLIAP_MFF.from_checkpoint(
    "model.pth",
    element_types=["H", "C", "N", "O"],
    device="cuda",
)

atoms.calc = MyE3NNCalculator(
    model=wrapper.wrapper.model,
    atomic_energies_dict={1: 0.0, 6: 0.0, 7: 0.0, 8: 0.0},
    device=torch.device("cuda"),
    max_radius=5.0,
)
```

真实 MD 生产建议先导出 AOTI/LAMMPS，并做数值验证后再跑长程模拟。

## 11. LAMMPS 部署

LAMMPS package 位于：

```text
lammps_user_mfftorch/
```

阅读：

- `lammps_user_mfftorch/README.md`
- `lammps_user_mfftorch/docs/BUILD_AND_RUN.md`

提供：

- `pair_style mff/torch`
- `pair_style mff/torch/kk`
- `compute ... mff/torch/phys`

一般流程：

1. 训练或转换 checkpoint。
2. 根据目标部署方式导出 AOTI `.pt2` 或 TorchScript core。
3. 编译带 `USER-MFFTORCH` 和 LibTorch 的 LAMMPS。
4. 在 LAMMPS input 中使用导出的模型。

简化示例：

```lammps
units metal
atom_style atomic
boundary p p p

read_data system.data
neighbor 1.0 bin

pair_style mff/torch/kk 5.0 cuda
pair_coeff * * /path/to/model.pt2 H C N O

fix 1 all nve
run 100
```

`pair_coeff` 里的元素顺序必须和导出/加载时一致。

## 12. Benchmark

主 benchmark：

```bash
python -m mace_ictd.bench.bench_mace_ictd_vs_mace \
  --device cuda \
  --dtype float32 \
  --channels 64 \
  --atoms-list 256,1024,4096 \
  --configs 1:1,2:2,2:3 \
  --train-iters 5 \
  --infer-iters 20 \
  --out-dir /tmp/mace_ictd_bench
```

这个 benchmark 是 backend/kernel throughput harness，不是化学精度验证。

建议分开比较：

- 原生 `mace-torch` e3nn backend。
- 原生 `mace-torch` cuEq backend。
- MACE-ICTD bridge-U eager/make_fx/AOTI。
- MACE-ICTD cuEq product eager/make_fx/AOTI。
- 可选 pure-U 诊断路径。

报告时不要混在一起：

- 首次 compile 时间，
- 稳态 step time，
- ASE/Python overhead，
- neighbor-list overhead，
- LAMMPS 端吞吐。

## 13. 测试和验证

基础 smoke tests：

```bash
python -m mace_ictd.test.test_training_smoke
python -m pytest mace_ictd/test/test_angular_basis.py -q
python -m pytest mace_ictd/test/test_export_aoti_core.py -q
```

MACE converter 验证：

```bash
python -m mace_ictd.test.test_mace_converter
```

cuEq product 测试：

```bash
python -m pytest mace_ictd/test/test_cueq_product_backend.py -q
python -m mace_ictd.test.test_cueq_makefx_training
```

cuEq 和 make_fx 测试需要 CUDA 环境。

数值精度预期：

- float64 bridge-U conversion 应该接近机器精度。
- float32 cuEq 路径按 float32 容差判断。
- AOTI 和 make_fx 路径必须和 eager 输出做 compile/load 后数值比较。

## 14. 常见坑

### Bridge-U 和 `angular_basis=e3nn`

Bridge-U 没有 e3nn fold path。bridge-U parity 路径应使用：

```bash
--product-backend ictd-bridge-u --angular-basis ictd
```

如果要 e3nn-folded product 推理，用：

```bash
--cueq-product --angular-basis e3nn
```

### cuEq product replacement

从 bridge-U product 替换到 cuEq product 时，只能拷 learnable MACE contraction weights，不能拷 bridge-U 的固定 `U_matrix_*` buffer。bridge-U 的 U 已经包含 ICTD/e3nn basis fold，直接拷到 cuEq 会错。当前 export path 已处理这个问题。

### `use_reduced_cg`

它是结构选项，不是无害的速度开关。转换原生 MACE 时跟随原 checkpoint；从头训练时要主动决定，并保证 checkpoint metadata 记录一致。

### ScaleShift 和 E0

MACE-style ScaleShift 作用在 interaction energy 上；E0 原子参考能单独加。要和原生 MACE 或部署绝对能量一致，必须检查：

- atomic energy keys/values，
- scale/shift，
- `avg_num_neighbors`，
- export 是否用了 `--embed-e0`。

### `max_ell` 和 `lmax`

- `lmax`：hidden feature angular cutoff。
- `max_ell`：edge spherical-harmonics cutoff。

原生 MACE 常见配置允许 `max_ell >= hidden_lmax`。更高 `max_ell` 会明显增加 product contraction 和 force training 成本。

### 动态 shape

Dynamic AOTI 和 make_fx 对 PyTorch/Inductor 版本敏感。如果 dynamic export 失败，可以尝试：

- `--static-n` 固定 atom count，
- 减少 dynamic 维度，
- 更小/更少的 bucket，
- PyTorch 2.7+，
- 关闭 fusion/autotune 等可选优化。

## 15. 开发说明

建议遵守：

- 修改角向 basis 或 product backend 前，先跑 MACE parity 测试。
- smoke test 通过不等于 MACE parity 已证明；parity 要跑 MACE-vs-ICTD converter test。
- 修改 checkpoint metadata 后，必须验证 `LAMMPS_MLIAP_MFF.from_checkpoint` strict reload。
- 修改 `angular_basis=e3nn` 后，必须验证 eager forward 和 checkpoint reload，避免 fixed buffer double-fold。

重要文件：

| 文件 | 作用 |
|---|---|
| `mace_ictd/models/pure_cartesian_ictd_fix.py` | 主模型和 product backends。 |
| `mace_ictd/mace_basis.py` | ICTD/e3nn 正交 basis 转换。 |
| `mace_ictd/interfaces/mace_converter.py` | 原生 MACE 到 MACE-ICTD 的权重转换。 |
| `mace_ictd/cli/export_aoti_core.py` | AOTI 导出、cuEq product replacement、export-time angular basis 逻辑。 |
| `mace_ictd/training/makefx_compile.py` | `make_fx` force-step 编译。 |
| `mace_ictd/training/train_loop.py` | Trainer、checkpoint metadata、ScaleShift/E0 loss 逻辑。 |
| `mace_ictd/interfaces/lammps_mliap.py` | 部署端 checkpoint reload 和 wrapper 逻辑。 |
| `mace_ictd/test/test_mace_converter.py` | 原生 MACE 到 MACE-ICTD 的转换 parity 测试。 |
