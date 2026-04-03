# Research Summary — BCI Cognitive Scaffolding

**Synthesis Date:** 2026-04-02

## Key Findings

### 1. PFC Vertex Mapping Is The #1 Priority Fix
The current hardcoded vertex ranges (LH: 0-3000, RH: 10242-13242) have no anatomical basis on fsaverage5. **The entire scoring pipeline is built on potentially wrong brain regions.** Use nilearn's Destrieux or Schaefer atlas for accurate PFC parcellation. This is foundational — everything downstream depends on it.

### 2. Absolute Thresholds Won't Work
fMRI prediction (and by extension, TRIBE v2 encoding predictions) has poor test-retest reliability at the individual level. The hardcoded thresholds (0.40 scaffolding, 0.55 stable) need per-session baseline calibration. Use relative stability changes (z-scores within session) instead of absolute values.

### 3. Stack Is Solid, One Critical Gap
The existing tech stack (TRIBE v2 + PyTorch 2.5 + PyTorch Lightning + nilearn) is appropriate and well-configured. The one gap is the nilearn atlas integration for PFC ROI extraction — pure Python/numpy, no new dependencies needed.

### 4. Build Order Is Clear
```
Phase 1: PFC Atlas ROI Module (foundation)
Phase 2: Scoring Functions + Calibration (core metric)
Phase 3: Temporal Smoothing (noise reduction)
Phase 4: State Machine (closed-loop control)
Phase 5: Karpathy Loop (automated refinement)
```

### 5. Three Scoring Methods Need Head-to-Head Comparison
- `inverse_variance`: Simple but assumption may not hold (low variance ≠ stability)
- `activation_ratio`: Better grounded but sensitive to whole-brain baseline
- `combined`: Most robust but needs weight tuning
→ Karpathy loop should test all three across stimulus types

## Architecture Recommendation

```
tribev2/
├── pfc_roi.py          # NEW — Atlas-based PFC extraction
├── scoring.py          # NEW — Stability scoring functions
├── state_machine.py    # NEW — Hysteresis state machine
└── karpathy_loop.py    # NEW — Automated experiment search
```

## Risk Mitigation

| Risk | Mitigation | Owner |
|------|-----------|-------|
| Wrong PFC vertices | Atlas-based extraction | Phase 1 |
| Unreliable thresholds | Per-session baseline calibration | Phase 2 |
| Smoothing masks signal | Asymmetric kernel, tune carefully | Phase 3 |
| State oscillation | 3-second sustain hysteresis | Phase 4 |
| Overfitting to stimuli | Cross-validate across stimulus categories | Phase 5 |

## Confidence Assessment

| Area | Confidence | Notes |
|------|-----------|-------|
| Stack choices | ★★★★★ | Already validated, right tools |
| PFC atlas approach | ★★★★☆ | Standard neuroscience practice |
| Scoring methods | ★★★☆☆ | Need empirical validation |
| State machine design | ★★★★☆ | Standard control theory |
| Threshold values | ★★☆☆☆ | Pure hypothesis until calibrated |

---
*Research synthesis: 2026-04-02*
