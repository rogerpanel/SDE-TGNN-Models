# SDE-TGNN: Stochastic Differential Equation Temporal Graph Neural Network

**Multi-Domain Network Intrusion Detection with Principled Uncertainty Quantification**

## Overview

SDE-TGNN is a novel deep learning framework for network intrusion detection that combines **Temporal Graph Neural Networks** with **Stochastic Differential Equations (SDEs)** and **Fokker-Planck-based analytical uncertainty propagation**. The model achieves state-of-the-art detection performance across six heterogeneous network security datasets while providing calibrated uncertainty estimates and certified adversarial robustness guarantees.

### Key Contributions

1. **SDE-Based Temporal Dynamics**: Models the continuous-time evolution of network state through learned drift and diffusion functions, capturing both deterministic trends and stochastic variability in traffic patterns.

2. **Fokker-Planck Uncertainty Propagation**: Instead of expensive Monte Carlo sampling, propagates the first two moments (mean and covariance) of the state distribution analytically, providing efficient O(d^2) uncertainty quantification.

3. **Multi-Scale Temporal Fusion**: Integrates SDE trajectories at multiple temporal scales, enabling the model to capture both short-term anomalies and long-term attack campaigns.

4. **Multi-Domain Generalization**: A unified architecture evaluated across six diverse network security domains (cloud, IIoT, container, enterprise, IoT, hybrid) without domain-specific modifications.

5. **Certified Robustness**: Leverages the stochastic nature of the SDE framework to provide certified accuracy guarantees via randomized smoothing.

## Architecture

```
Input Features --> Feature Embedding --> Graph Attention (x4 layers)
                                              |
                                    State Projection (D -> d)
                                              |
                              +---------------+---------------+
                              |               |               |
                        SDE Scale 1     SDE Scale 2     SDE Scale 3
                        (short-term)    (medium-term)   (long-term)
                              |               |               |
                              +-------+-------+-------+-------+
                                      |               |
                              Multi-Scale Fusion   Fokker-Planck
                                      |           (mu, Sigma)
                                      v               |
                              Classification Head <---+
                                      |
                              Logits + Uncertainty
```

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- CUDA >= 11.7 (recommended for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/sde-tgnn/sde-tgnn.git
cd sde-tgnn

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Dataset Preparation

SDE-TGNN is evaluated on six network intrusion detection datasets:

| Dataset | Domain | Samples | Classes | Source |
|---------|--------|---------|---------|--------|
| Microsoft Azure Cloud | Cloud Security | ~1M | 2 | [Kaggle](https://www.kaggle.com/c/microsoft-malware-prediction) |
| Edge-IIoTset | Industrial IoT | ~157K | 15 | [IEEE DataPort](https://ieee-dataport.org/documents/edge-iiotset) |
| Kubernetes vs Docker | Container Security | ~50K | 5 | [Custom Collection] |
| CSE-CIC-IDS2018 | Enterprise Network | ~16M | 15 | [UNB](https://www.unb.ca/cic/datasets/ids-2018.html) |
| UNB-CIC-IoT2023 | IoT Network | ~33M | 34 | [UNB](https://www.unb.ca/cic/datasets/iotdataset-2023.html) |
| UNSW-NB15 | Hybrid Network | ~2.5M | 10 | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |

### Directory Structure for Raw Data

```
data/raw/
    microsoft_cloud/
        microsoft_cloud_malware.csv
    edge_iiot/
        edge_iiot.csv
    kubernetes_docker/
        kubernetes_docker.csv
    cic_ids2018/
        cic_ids2018.csv
    cic_iot2023/
        cic_iot2023.csv
    unsw_nb15/
        unsw_nb15.csv
```

### Preprocessing

```bash
# Preprocess all datasets
python scripts/preprocess_data.py --data_root data/raw --output_dir data/processed

# Preprocess a single dataset with graph construction
python scripts/preprocess_data.py --dataset cic_ids2018 --build_graphs --graph_k 10
```

## Training

### Quick Start

```bash
# Train on all datasets with default configuration
python scripts/train.py --dataset all --config config/default_config.yaml --output_dir outputs/all

# Train on a specific dataset
python scripts/train.py --dataset cic_ids2018 --output_dir outputs/cic_ids2018
```

### Experiment Configurations

```bash
# ICS3D: Industrial Control System domains (Cloud + Container + IIoT)
python scripts/train.py --config experiments/configs/ics3d_config.yaml --output_dir outputs/ics3d

# IIS3D: Internet/IoT Security domains (CIC-IDS2018 + CIC-IoT2023 + UNSW-NB15)
python scripts/train.py --config experiments/configs/iis3d_config.yaml --output_dir outputs/iis3d
```

### Resume Training

```bash
python scripts/train.py --resume outputs/all/checkpoints/epoch_50.pt --output_dir outputs/all
```

### Training Configuration

Key hyperparameters in `config/default_config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.hidden_dim` | 256 | Graph attention hidden dimension |
| `model.state_dim` | 64 | SDE state dimension |
| `model.num_layers` | 4 | Number of graph attention layers |
| `model.num_heads` | 8 | Attention heads |
| `model.num_scales` | 3 | Temporal scales |
| `training.lr` | 2e-4 | Learning rate |
| `training.epochs` | 100 | Maximum training epochs |
| `training.patience` | 15 | Early stopping patience |
| `sde.solver` | euler_maruyama | SDE solver type |
| `sde.dt` | 0.01 | Integration time step |
| `fokker_planck.moment_order` | 2 | FP moment propagation order |

## Evaluation

### Full Evaluation Pipeline

```bash
# Run complete evaluation (metrics + calibration + adversarial)
python scripts/evaluate.py \
    --checkpoint outputs/all/checkpoints/best_model.pt \
    --dataset all \
    --adversarial \
    --output_dir evaluation_results
```

### Evaluation Outputs

The evaluation script produces:
- **Detection metrics**: Accuracy, Precision, Recall, F1, AUC-ROC per dataset
- **Calibration analysis**: ECE, MCE, Brier score, reliability diagrams
- **Adversarial robustness**: FGSM, PGD, certified accuracy at multiple epsilon values
- **Uncertainty analysis**: Predictive entropy distributions, epistemic/aleatoric decomposition
- **Visualizations**: Publication-quality figures saved to `evaluation_results/figures/`

## Repository Structure

```
SDE_TGNN_Models/
    config/
        default_config.yaml           # Default hyperparameters
    experiments/
        configs/
            ics3d_config.yaml          # ICS 3-domain experiment
            iis3d_config.yaml          # IIS 3-domain experiment
    scripts/
        train.py                       # Training script
        evaluate.py                    # Evaluation script
        preprocess_data.py             # Data preprocessing
    src/
        __init__.py
        data/
            __init__.py
            preprocessing.py           # Dataset preprocessing pipelines
            dataset.py                 # PyTorch Dataset/DataLoader classes
            feature_engineering.py     # Feature harmonization, graph construction
        models/
            __init__.py
            sde_tgnn.py                # Main SDE-TGNN architecture
            drift_network.py           # Deterministic drift f_theta(h, G, t)
            diffusion_network.py       # Learned diffusion sigma_phi(h, t)
            graph_attention.py         # Temporal graph attention layers
            fokker_planck.py           # FP moment propagation solver
            sde_solver.py              # SDE numerical solvers (EM, Milstein, Adaptive)
            baselines.py               # 8 baseline models
        training/
            __init__.py
            trainer.py                 # Training loop with early stopping
            losses.py                  # ELBO, calibration, and combined losses
        evaluation/
            __init__.py
            metrics.py                 # Detection metrics (Accuracy, F1, AUC-ROC)
            calibration.py             # ECE, Brier, temperature scaling
            adversarial.py             # PGD, FGSM, certified accuracy
            visualization.py           # Publication-quality plots
    requirements.txt
    setup.py
    README.md
```

## Baseline Models

SDE-TGNN is compared against eight baselines:

| Model | Category | Uncertainty |
|-------|----------|-------------|
| Random Forest | Classical ML | No |
| XGBoost | Classical ML | No |
| BiLSTM | Deep Learning | No |
| CNN-BiLSTM | Deep Learning | No |
| GraphSAGE | Graph Neural Network | No |
| Neural ODE | Continuous Depth | No |
| MC Dropout | Bayesian Approximation | Yes |
| Deep Ensemble | Ensemble | Yes |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{sdetgnn2025,
  title={SDE-TGNN: Stochastic Differential Equation Temporal Graph Neural Network
         for Multi-Domain Network Intrusion Detection},
  author={SDE-TGNN Authors},
  journal={},
  year={2025},
  note={Under Review}
}
```

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2025 SDE-TGNN Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
