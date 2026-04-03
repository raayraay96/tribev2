# TRIBE v2 — Cognitive Scaffolding BCI System

## What This Is

A closed-loop brain-computer interface backend that uses Meta's TRIBE v2 foundation model (177.2M parameters) to predict prefrontal cortex activity from naturalistic stimuli and drive real-time cognitive scaffolding interventions for ADHD. The system extracts PFC stability scores from fMRI-level cortical predictions and uses a hysteresis state machine to trigger software-based interventions (visual vignetting, adaptive pacing, phasic alerting) that mimic the pharmacological effects of Guanfacine.

## Core Value

Accurate, neurologically consistent PFC stability scoring from TRIBE v2 predictions — the foundation that makes closed-loop scaffolding interventions meaningful rather than arbitrary.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. Inferred from existing codebase. -->

- ✓ TRIBE v2 model loading and inference via TribeModel.from_pretrained() — existing
- ✓ Multimodal feature extraction pipeline (text/audio/video) — existing
- ✓ Text-to-speech → WhisperX transcription pipeline for text inputs — existing
- ✓ Cortical surface prediction on fsaverage5 mesh (~20,484 vertices) — existing
- ✓ PyTorch Lightning training loop with SLURM support — existing
- ✓ Multi-study data loading (Algonauts2025, Lahner2024, Lebel2023, Wen2017) — existing
- ✓ Brain surface visualization (nilearn + PyVista backends) — existing
- ✓ VRAM-aware sequential feature extraction with explicit cleanup — existing
- ✓ Grid search configuration and experiment management — existing
- ✓ Conda environment on Scholar cluster (Python 3.11, PyTorch 2.5.1, CUDA 12.1) — existing

### Active

<!-- Current scope. Building toward these. -->

- [ ] PFC vertex extraction accuracy — validate vertex ranges against Brodmann areas 9/10/46
- [ ] PFC stability scoring function — calibrate inverse_variance, activation_ratio, and combined methods
- [ ] Hysteresis state machine — implement SCAFFOLDING/EDGE/STABLE transitions with 3-second sustain recovery
- [ ] Stimulus design optimization — maximize cognitive state differentiability across stimulus types
- [ ] Temporal smoothing — implement and calibrate per-timestep noise reduction
- [ ] Karpathy loop automation — iterative experiment loop with automated parameter search
- [ ] VARIANCE_SCALE calibration — determine optimal sensitivity for inverse_variance scoring
- [ ] Threshold calibration — validate 0.40/0.55 thresholds against ground-truth cognitive states
- [ ] Real-time inference pipeline — sub-second latency from stimulus to stability score
- [ ] Closed-loop intervention system — trigger vignetting/pacing/alerting based on state machine output

### Out of Scope

- EEG/HRV hardware integration — requires Muse 2 and Polar H10 physical setup (hardware layer)
- IRB protocol submission — institutional process, not a coding task
- Multi-GPU distributed training — model is 177M params, fits single GPU
- Custom feature extractor training — feature extractors are frozen by design
- Production deployment to cloud — this is a research system on Scholar cluster

## Context

**Technical Environment:**
- Purdue Scholar GPU cluster with V100 16GB and A30 24GB GPUs via SLURM
- Home directory ~1GB quota — all caches must go to /scratch/scholar/
- LLaMA 3.2-3B requires HuggingFace gated access (accepted, HF_TOKEN configured)
- Meta's neuralset/neuraltrain ecosystem pinned at 0.0.2

**Prior Work:**
- TRIBE v2 paper and pretrained weights from Meta Research (facebook/tribev2)
- CLAUDE.md documents the scaffolding pipeline architecture and Karpathy loop targets
- scaffolding_pipeline.py in parent directory implements initial PFC scoring (under development)
- Previous sessions resolved PyTorch 2.5+ API compatibility and VRAM contention issues

**Research Protocol:**
- Karpathy loop: iterative refinement of PFC vertex ranges, scoring parameters, state machine thresholds
- 10-minute budget experiments via train_experiment.py
- Maximize val_pearson correlation between predicted and measured cortical activity

## Constraints

- **GPU Memory**: V100 16GB minimum — sequential feature extraction mandatory, no parallel model loading
- **Disk Quota**: Home dir ~1GB — all caches redirect to /scratch/scholar/edraymon/.cache/
- **Dependencies**: neuralset==0.0.2, neuraltrain==0.0.2 pinned exactly — Meta internal packages
- **PyTorch**: ≥2.5.1, <2.7 — model uses torch.compiler APIs
- **License**: CC-BY-NC-4.0 — non-commercial research only
- **Reproducibility**: seed=33 across all experiments for result consistency
- **SLURM**: All training/inference jobs run via SLURM scheduler, not interactive
- **fsaverage5**: Cortical mesh locked at ~20,484 vertex resolution (project standard)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use inverse_variance as default scoring method | Lower PFC variance correlates with stability; simplest interpretable metric | — Pending |
| Sequential feature extraction with VRAM cleanup | V100 16GB can't hold all 3 frozen models simultaneously | ✓ Good |
| fsaverage5 over fsaverage6 | Balanced resolution vs compute; sufficient for PFC region extraction | — Pending |
| Hysteresis state machine with 3-second sustain | Prevent oscillation between SCAFFOLDING and STABLE states | — Pending |
| PFC vertices: LH 0-3000, RH 10242-13242 | Approximate mapping to prefrontal cortex on fsaverage5 mesh | ⚠️ Revisit — needs validation against Brodmann atlas |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-02 after initialization*
