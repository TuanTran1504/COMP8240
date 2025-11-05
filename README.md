# 🧠 Mixup Reproduction & Extension

This project reproduces and extends the results of **Mixup: Beyond Empirical Risk Minimization** (Zhang et al., 2018).  
It replicates the original experiments on **CIFAR-10**, **CIFAR-100**, and **Speech Commands**, and further tests Mixup on **Fashion-MNIST** to evaluate its generalization performance on unseen data.

---

## 🚀 Features

- Full reproduction of Mixup experiments for image and audio classification  
- Implementations of **ResNet-18 (PreAct)**, **Wide-ResNet-28-10**, and **VGG11**  
- Additional experiments on **Fashion-MNIST** (grayscale)  
- Spectrogram-level Mixup for audio data  
- Unified training pipeline with Mixup and ERM baselines  
- Reproducible results with GPU/CPU support  

---

## 🧱 Project Structure

```
mixup/
├── train2.py                 # Unified trainer for classification + Mixup
├── models/
│   └── resnet.py             # ResNet18_* architectures with PreAct blocks
├── command_dataset.py        # Speech Commands loader (16 kHz → spectrograms)
├── utils.py                  # Progress bar and helper utilities
├── results/                  # CSV logs (auto-generated)
├── checkpoint/               # Model checkpoints (auto-generated)
├── THIRD_PARTY_NOTICES.md
├── LICENSE
└── README.md
```

---

## 🛠️ Tech Stack

| Component      | Library/Framework                    |
|----------------|--------------------------------------|
| Deep Learning  | PyTorch 2.8.0 + CUDA 12.6            |
| Datasets       | Torchvision (CIFAR, Fashion-MNIST)   |
| Audio Handling | Torchaudio + Spectrogram Transform   |
| Models         | ResNet-18, WideResNet-28-10, VGG11   |
| Augmentation   | Mixup (linear interpolation)         |
| Logging        | CSV output + progress bar            |

---

## ⚙️ Setup

### 🧩 Environment Setup

```bash
conda create -n mixup python=3.10 -y
conda activate mixup
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install numpy tqdm matplotlib pyyaml
```

---

## 📦 Datasets

- **CIFAR-10 / CIFAR-100**: Auto-downloaded via `torchvision.datasets`
- **Fashion-MNIST**: Drop-in grayscale benchmark (fixed 1-channel)
- **Speech Commands**: Converted to spectrograms (160×101) at 16 kHz  
  *(see `command_dataset.py` for path configuration)*

Default data root: `~/data`  
Override with `--data-root /path/to/data`

---

## ▶️ Quickstart

### 🧩 CIFAR-10 / CIFAR-100
Baseline (ERM, no Mixup):
```bash
python -u train2.py --model ResNet18_CIFAR10 --dataset CIFAR10   --adjustlr 100-150 --lr 0.1 --alpha 0 --seed 42 --name ResNet_C10 --amp
```

Mixup (α = 1.0):
```bash
python -u train2.py --model ResNet18_CIFAR10 --dataset CIFAR10   --adjustlr 100-150 --lr 0.1 --alpha 1.0 --seed 42 --name ResNet_C10_Mixup --amp
```

CIFAR-100:
```bash
python -u train2.py --model ResNet18_CIFAR100 --dataset CIFAR100   --adjustlr 100-150 --lr 0.1 --alpha 1.0 --seed 42 --name ResNet_C100_Mixup --amp
```

### 👕 Fashion-MNIST
> FMNIST is fixed to single-channel input (1 ch).

```bash
python -u train2.py --model ResNet18_FMNIST --dataset FASHIONMNIST   --adjustlr 100-150 --lr 0.1 --alpha 1.0 --seed 42 --name ResNet_FMNIST --amp
```

### 🎧 Speech Commands
VGG11 (spectrogram-level Mixup):
```bash
python -u train2.py --model VGG11 --dataset COMMAND   --lr 0.001 --alpha 1.0 --epoch 30 --batch-size 128 --seed 42   --name VGG11_Commands --amp
```

---

## 📊 Results Summary

| Dataset | Model | ERM (Test Err %) | Mixup (Test Err %) | Observation |
|----------|--------|------------------|--------------------|--------------|
| CIFAR-10 | ResNet-18 | 5.2 | 4.3 | Matches original trend |
| CIFAR-100 | ResNet-18 | 24.4 | 22.5 | Slight improvement |
| FMNIST | ResNet-18 | 7.3 | 6.4 | Improved generalization |
| Commands | VGG11 | 4.1 | 3.8 | Stable across seeds |

Results align closely with the Mixup paper; small deviations arise from hardware and random seed differences.

---

## 🔁 Reproducibility Notes

- Deterministic training with `--seed ≠ 0`
  - Seeds set for Python, NumPy, and PyTorch
  - `cudnn.deterministic = True`, `cudnn.benchmark = False`
- Mixed precision uses:
  ```python
  with torch.amp.autocast(device_type="cuda", enabled=args.amp):
  ```
- LR schedule: milestones controlled via `--adjustlr` (e.g., `100-150`)

---

## 📁 Logs & Checkpoints

| Output | Path |
|---------|------|
| Training logs | `results/log_<Model>_<RunName>_<Seed>.csv` |
| Checkpoints | `checkpoint/ckpt_<RunName>_{best|last}_<Seed>.pt` |

---

## 📘 Acknowledgments & Licenses

- **Mixup:** Zhang *et al.* (ICLR 2018)  
- **CIFAR, Fashion-MNIST, Speech Commands:** via `torchvision`  
- PreAct ResNet structure adapted from open-source examples (MIT/BSD)  
- See `THIRD_PARTY_NOTICES.md` and `LICENSE` for full details  

---

## 🧾 Citation

If you use this project, please cite:

```bibtex
@inproceedings{zhang2018mixup,
  title={mixup: Beyond Empirical Risk Minimization},
  author={Zhang, Hongyi and Cisse, Moustapha and Dauphin, Yann N. and Lopez-Paz, David},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2018}
}
```

---

## 🧩 Troubleshooting

| Issue | Solution |
|--------|-----------|
| **Windows DataLoader freeze** | use `--num-workers 0` |
| **CUDA not detected** | verify PyTorch CUDA build matches driver |
| **FMNIST channel mismatch** | repo fixed to 1-channel; use `ResNet18_FMNIST` |
| **No output** | run with `-u` flag and check `results/` for CSV logs |

---

## 📄 License

MIT License © 2025 Dinh Tuan Tran
