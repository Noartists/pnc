"""
In-Context Dynamics Learning for Parafoil Systems.

Modules:
  - data_generation: Domain randomization, control policies, dataset generation
  - dataset: PyTorch dataset and normalization
  - model: Transformer dynamics model + MLP baseline
  - trainer: Training loop with curriculum learning
  - evaluate: OOD evaluation, ablation, figure generation
  - mpc_controller: MPPI controller with learned dynamics
"""
