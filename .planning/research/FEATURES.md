# Features Research — BCI Cognitive Scaffolding

**Research Date:** 2026-04-02
**Domain:** Closed-loop BCI for ADHD Cognitive Scaffolding

## Table Stakes (Must Have)

| Feature | Complexity | Dependencies | Notes |
|---------|-----------|--------------|-------|
| Accurate PFC vertex extraction | Medium | nilearn atlas | Currently hardcoded; needs atlas-based ROI |
| PFC stability scoring | Medium | Model inference | Core metric driving all interventions |
| State machine (SCAFFOLD/EDGE/STABLE) | Low | Scoring function | Hysteresis prevents oscillation |
| Multi-stimulus inference | Low | Already implemented | Text, audio, video inputs |
| Reproducible experiments | Low | seed=33, config YAML | Already in place |

## Differentiators (Competitive Advantage)

| Feature | Complexity | Dependencies | Notes |
|---------|-----------|--------------|-------|
| Guanfacine-mimicking interventions | High | State machine + actuation | Unique: software replicating pharmacological effect |
| Karpathy loop automation | Medium | SLURM + parameter search | Automated architecture search |
| Combined scoring method | Medium | Multiple metrics | 60% inverse_variance + 40% temporal stability |
| Temporal smoothing | Low | Gaussian conv | Noise reduction in per-timestep scores |
| Stimulus optimization | High | Experiment design | Maximize cognitive state differentiation |

## Anti-Features (Deliberately NOT Building)

| Feature | Reason |
|---------|--------|
| EEG signal processing | Hardware layer — separate project |
| Real-time fMRI acquisition | Using predicted cortical activity, not measured |
| Clinical diagnostic tool | Research prototype; no clinical validation |
| Custom model training from scratch | Using pre-trained TRIBE v2 weights |
| Mobile/web interface | Desktop research tool only |

## Feature Dependencies

```
PFC Atlas ROI → PFC Stability Score → State Machine → Interventions
                                    ↗
Temporal Smoothing ────────────────┘
                                    
Karpathy Loop → Parameter Search → Best Config → Production Pipeline
```

## Key Feature: Neurofeedback Validation

Research shows neurofeedback requires **consistency and multiple sessions** for neuroplastic changes. The system must:
1. Track stability scores across sessions (not just within)
2. Detect adaptation/learning effects over time
3. Provide session-level summary statistics

**Warning:** fMRI prediction has poor test-retest reliability in individuals. The system should use relative (within-session) stability changes, not absolute thresholds.

---
*Features research: 2026-04-02*
