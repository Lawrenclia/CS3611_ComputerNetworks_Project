# Project Artifacts

This file records where generated outputs and model weights live.

## Model Weights

Active weights used by default:

- Q-Learning: `artifacts/models/active/q_table.json`
- DQN: `artifacts/models/active/dqn_model.pt`

Backups:

- `artifacts/models/backups/`

Candidate weights produced by curriculum training before installation:

- `artifacts/models/candidates/`

Root-level `q_table.json`, `q_table_good.json`, `q_table_curriculum.json`, and `dqn_model.pt` are legacy compatibility copies. New training and demos should use `artifacts/models/active/`.

## Images

- Report and poster figures: `report/figures/`
- One-click demo outputs: `artifacts/demo_results/<timestamp>/`
- Ad-hoc comparison plots: `artifacts/plots/`
- Legacy collected outputs from earlier runs: `artifacts/legacy/`

## Reports

- Final report PDF: `report/output/cs3611_project5_report.pdf`
- Final poster PPTX: `report/output/cs3611_project5_poster.pptx`
- LaTeX report source: `report/cs3611_project5_report.tex`

## Training Outputs

- Training metrics and summaries: `artifacts/training/`
- Q-Learning checkpoints: `artifacts/checkpoints/qlearning/`
- Curriculum Q-Learning checkpoints: `artifacts/checkpoints/q_curriculum_<timestamp>/`
- DQN checkpoints: `artifacts/checkpoints/dqn/`
