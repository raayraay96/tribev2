# Requirements — TRIBE v2 Cognitive Scaffolding BCI

## Milestone: v1.0 — PFC Stability Pipeline

### Must-Haves

| ID | Requirement | Source | UAT Criteria |
|----|-------------|--------|-------------|
| R1 | Atlas-based PFC vertex extraction using Destrieux/Schaefer atlas on fsaverage5 | Research, CONCERNS.md | Given TRIBE v2 predictions, when PFC ROI extracted, then vertices correspond to documented PFC labels (not hardcoded indices) |
| R2 | PFC stability scoring with 3 methods (inverse_variance, activation_ratio, combined) | PROJECT.md Active | Given PFC vertex predictions, when scoring applied, then output is float ∈ [0,1] that differentiates cognitive states |
| R3 | Per-session baseline calibration for scoring thresholds | Research PITFALLS | Given a new session, when baseline established, then thresholds are relative to session mean/std, not absolute |
| R4 | Gaussian temporal smoothing with configurable kernel | PROJECT.md Active | Given raw per-timestep scores, when smoothing applied, then noise reduced without masking acute state changes |
| R5 | Hysteresis state machine (STABLE/EDGE/SCAFFOLDING) with 3s sustain timer | PROJECT.md Active | Given score stream, when state transitions occur, then no oscillation faster than 3 seconds |
| R6 | Automated Karpathy loop for parameter search | PROJECT.md Active | Given parameter space, when loop runs, then best config by val_pearson saved with reproducible seed |
| R7 | Stimulus-differentiated scoring validation | Research PITFALLS | Given 5 stimulus types, when scores computed, then neutral > cognitive_load and relaxation > emotional_arousal |
| R8 | End-to-end scaffolding pipeline (stimuli → score → state) | PROJECT.md Active | Given text input, when pipeline runs, then produces state machine decisions with <5s latency |

### Should-Haves

| ID | Requirement | Source | UAT Criteria |
|----|-------------|--------|-------------|
| S1 | Sub-regional PFC decomposition (dorsolateral, anterior, ventromedial) | Research ARCHITECTURE | Given PFC predictions, when decomposed, then each sub-region scored independently |
| S2 | Cross-stimulus validation (held-out stimulus evaluation) | Research PITFALLS | Given scored model, when novel stimulus tested, then scores are neurologically plausible |
| S3 | Session-level summary statistics and visualization | Research FEATURES | Given completed session, when summary generated, then includes score timeline, state distribution, intervention count |
| S4 | W&B integration for scaffolding experiment tracking | STACK.md | Given experiment run, when completed, then metrics logged to W&B dashboard |

### Won't-Haves (This Milestone)

| ID | Exclusion | Reason |
|----|-----------|--------|
| W1 | EEG/HRV hardware integration | Hardware layer — separate project scope |
| W2 | Desktop GUI for interventions | Actuation layer — future milestone |
| W3 | Multi-subject training | Using pretrained average-subject model |
| W4 | Cloud deployment | Research system on Scholar cluster |
| W5 | Real-time streaming interface | Batch inference sufficient for v1.0 |

---
*Requirements defined: 2026-04-02*
