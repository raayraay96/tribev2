# Roadmap — TRIBE v2 Cognitive Scaffolding BCI

## Milestone: v1.0 — PFC Stability Pipeline

### Phase 1: PFC Atlas ROI Module
**Goal:** Replace hardcoded vertex ranges with atlas-based PFC extraction
**Requirements:** R1
**Deliverables:**
- `tribev2/pfc_roi.py` — Atlas-based PFC vertex extraction module
- Functions: `load_pfc_atlas()`, `extract_pfc_vertices()`, `get_pfc_labels()`
- Uses nilearn Destrieux atlas on fsaverage5 for anatomically accurate PFC parcellation
- Returns labeled PFC sub-regions (dorsolateral, anterior, ventromedial)
- Unit tests validating vertex counts and label correspondence
**Estimate:** Small (1-2 plans)

---

### Phase 2: PFC Stability Scoring
**Goal:** Implement and validate scoring functions that produce meaningful stability metrics
**Requirements:** R2, R3, R7
**Deliverables:**
- `tribev2/scoring.py` — Three scoring methods + baseline calibration
- `inverse_variance(pfc_vertices)` → scaled score ∈ [0,1]
- `activation_ratio(pfc_vertices, whole_brain)` → ratio score ∈ [0,1]
- `combined(pfc_vertices, whole_brain)` → weighted composite ∈ [0,1]
- Per-session baseline: z-score normalization using session mean/std
- Validation script comparing scores across 5 stimulus types
**Estimate:** Medium (2-3 plans)
**Depends on:** Phase 1

---

### Phase 3: Temporal Smoothing
**Goal:** Reduce per-timestep noise while preserving acute state changes
**Requirements:** R4
**Deliverables:**
- Add `temporal_smooth()` to `tribev2/scoring.py`
- Gaussian kernel with configurable size (default: 5 timesteps)
- Asymmetric option: smooth increases, preserve rapid drops
- Comparison plots: raw vs smoothed scores for each stimulus type
**Estimate:** Small (1 plan)
**Depends on:** Phase 2

---

### Phase 4: Hysteresis State Machine
**Goal:** Implement closed-loop state controller with oscillation prevention
**Requirements:** R5, R8
**Deliverables:**
- `tribev2/state_machine.py` — Hysteresis state machine module
- States: STABLE, EDGE, SCAFFOLDING (Python Enum)
- Configurable thresholds (default: SCAFFOLDING < 0.40, STABLE > 0.55)
- 3-second sustain timer for recovery
- Event emission (state_changed callback)
- End-to-end pipeline: text → model → PFC → score → smooth → state
**Estimate:** Medium (2 plans)
**Depends on:** Phase 3

---

### Phase 5: Karpathy Loop Automation
**Goal:** Automated parameter search to find optimal scoring configuration
**Requirements:** R6
**Deliverables:**
- `tribev2/karpathy_loop.py` — Automated experiment runner
- Parameter space: VARIANCE_SCALE, thresholds, smoothing kernel, scoring method
- SLURM job submission for each configuration
- Results aggregation with val_pearson ranking
- Best config export to YAML
- 10-minute budget per experiment
**Estimate:** Medium (2-3 plans)
**Depends on:** Phase 4

---

### Phase 6: Integration & Validation
**Goal:** End-to-end validation and documentation
**Requirements:** R7, R8, S1, S2, S3
**Deliverables:**
- Integration test: full pipeline from text input to state machine output
- Cross-stimulus validation: hold out 1 stimulus, validate generalization
- Sub-regional PFC analysis (S1)
- Session summary reports with visualization (S3)
- Updated CLAUDE.md with final architecture
- Updated README with scaffolding pipeline docs
**Estimate:** Medium (2-3 plans)
**Depends on:** Phase 5

---

## Phase Summary

| Phase | Name | Plans (est) | Dependencies | Status |
|-------|------|-------------|--------------|--------|
| 1 | PFC Atlas ROI Module | 1-2 | None | Not started |
| 2 | PFC Stability Scoring | 2-3 | Phase 1 | Not started |
| 3 | Temporal Smoothing | 1 | Phase 2 | Not started |
| 4 | Hysteresis State Machine | 2 | Phase 3 | Not started |
| 5 | Karpathy Loop Automation | 2-3 | Phase 4 | Not started |
| 6 | Integration & Validation | 2-3 | Phase 5 | Not started |

**Total estimated plans:** 10-15
**Build order:** Sequential — each phase depends on the previous

---
*Roadmap created: 2026-04-02*
