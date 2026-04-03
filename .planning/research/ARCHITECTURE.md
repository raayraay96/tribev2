# Architecture Research — BCI Cognitive Scaffolding

**Research Date:** 2026-04-02
**Domain:** Closed-loop BCI for ADHD Cognitive Scaffolding

## System Architecture

```
┌───────────────────────────────────────────────────────────┐
│                  TRIBE V2 MODEL LAYER                     │
│  TribeModel.from_pretrained("facebook/tribev2")           │
│  Input: text/audio/video → Output: (T, 20484) predictions │
└─────────────────────────┬─────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────┐
│              PFC EXTRACTION LAYER (NEW)                    │
│  Atlas-based ROI extraction (Destrieux/Schaefer)          │
│  PFC vertices → (T, n_pfc_vertices) subset                │
└─────────────────────────┬─────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────┐
│              SCORING LAYER (NEW)                           │
│  inverse_variance: 1/(1 + var*100)                        │
│  activation_ratio: PFC_activation / whole_brain           │
│  combined: 0.6*inv_var + 0.4*temporal_stability           │
│  Optional: Gaussian temporal smoothing                     │
│  Output: scalar stability score ∈ [0, 1]                  │
└─────────────────────────┬─────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────┐
│           STATE MACHINE LAYER (NEW)                        │
│  States: STABLE(>0.55) | EDGE(0.40-0.55) | SCAFFOLDING   │
│  Hysteresis: sustain recovery 3s before STABLE transition │
│  Event emission: state_changed, intervention_needed       │
└─────────────────────────┬─────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────┐
│          INTERVENTION LAYER (FUTURE)                       │
│  A. Visual Vignetting (peripheral darkening)              │
│  B. Adaptive ISI Pacing (1.0s → 1.6s)                    │
│  C. Phasic Alerting (400Hz pip + green highlight)         │
└───────────────────────────────────────────────────────────┘
```

## Component Boundaries

1. **Model Layer** — Read-only. TribeModel inference only. No modifications to TRIBE v2 weights.
2. **PFC Extraction** — New module. Depends on nilearn atlas. Pure numpy operations.
3. **Scoring Layer** — New module. Pure numpy/scipy. Stateless per-timestep computation.
4. **State Machine** — New module. Stateful. Tracks transitions and sustain timers.
5. **Intervention Layer** — Future. Desktop GUI integration. Out of scope for v1.

## Data Flow

1. Stimulus text → `model.get_events_dataframe(text_path=...)` → events DataFrame
2. Events → `model.predict(events)` → `(n_timesteps, 20484)` cortical predictions
3. Predictions → PFC ROI extraction → `(n_timesteps, n_pfc_vertices)` subset
4. PFC subset → scoring function → `(n_timesteps,)` stability scores
5. Scores → state machine → state transitions + intervention triggers

## Build Order (Dependencies)

| Phase | Component | Depends On |
|-------|-----------|------------|
| 1 | PFC Atlas ROI Module | nilearn, fsaverage5 |
| 2 | Scoring Functions | PFC ROI Module |
| 3 | Temporal Smoothing | Scoring Functions |
| 4 | State Machine | Scoring + Smoothing |
| 5 | Karpathy Loop | All above + SLURM |
| 6 | Intervention System | State Machine (future) |

## Key Architecture Decision: Atlas vs Hardcoded Vertices

**Current:** Hardcoded PFC vertex ranges (LH: 0-3000, RH: 10242-13242)
**Problem:** Vertex indices on fsaverage5 don't linearly map to anatomical regions. The first 3000 vertices are NOT necessarily prefrontal cortex.
**Solution:** Use nilearn surface atlas (Destrieux or Schaefer) to extract accurate PFC labels

```python
# Correct approach:
from nilearn import datasets
destrieux = datasets.fetch_atlas_surf_destrieux()
# Filter for PFC-relevant labels (superior frontal, middle frontal, etc.)
pfc_labels = [label for label in destrieux['labels'] if 'front' in label.lower()]
pfc_mask = np.isin(destrieux['map_left'], pfc_label_indices)
pfc_vertices = predictions[:, pfc_mask]
```

---
*Architecture research: 2026-04-02*
