# Pitfalls Research — BCI Cognitive Scaffolding

**Research Date:** 2026-04-02
**Domain:** Closed-loop BCI for ADHD Cognitive Scaffolding

## Critical Pitfalls

### 1. Hardcoded PFC Vertex Ranges Are Anatomically Wrong
- **Warning signs:** Stability scores don't differentiate between cognitive states
- **Prevention:** Use nilearn Destrieux/Schaefer atlas for vertex selection; validate against known PFC parcellation
- **Phase mapping:** Phase 1 (PFC Atlas Module)
- **Severity:** HIGH — entire scoring pipeline built on wrong vertices

### 2. fMRI Prediction Has Poor Test-Retest Reliability
- **Warning signs:** Same stimulus gives wildly different scores across runs
- **Prevention:** Use relative stability (within-session changes) not absolute thresholds. Calibrate baseline per session.
- **Phase mapping:** Phase 2 (Scoring Calibration)
- **Severity:** HIGH — absolute thresholds (0.40/0.55) may be meaningless without per-session baseline

### 3. Overfitting Scoring Parameters to Training Stimuli
- **Warning signs:** Scoring works perfectly on 5 stimulus types but fails on novel content
- **Prevention:** Hold out stimulus types during calibration. Cross-validate across stimulus categories.
- **Phase mapping:** Phase 5 (Karpathy Loop)
- **Severity:** MEDIUM — could produce a system that only works on specific texts

### 4. State Machine Oscillation
- **Warning signs:** Rapid STABLE↔SCAFFOLDING transitions within seconds
- **Prevention:** Hysteresis with sustain timer (3 seconds minimum in recovery state). Debounce state changes.
- **Phase mapping:** Phase 4 (State Machine)
- **Severity:** MEDIUM — causes jarring user experience with flickering interventions

### 5. VRAM OOM During Extended Inference
- **Warning signs:** Jobs crash after processing many stimuli without cleanup
- **Prevention:** Already addressed with `_free_extractor_model()`. Ensure inference pipeline also cleans up between batches.
- **Phase mapping:** All phases (infrastructure concern)
- **Severity:** LOW — already mitigated

### 6. Temporal Smoothing Destroys Sharp Transitions
- **Warning signs:** State machine never enters SCAFFOLDING because smoothing blurs rapid drops
- **Prevention:** Tune kernel size carefully. Use asymmetric smoothing (smooth increases, preserve drops).
- **Phase mapping:** Phase 3 (Temporal Smoothing)
- **Severity:** MEDIUM — could mask the signal you're trying to detect

### 7. Variance-Based Scoring Is Not Neurologically Grounded
- **Warning signs:** Low PFC variance might indicate disengagement, not stability
- **Prevention:** Validate against known cognitive states. Compare inverse_variance with activation_ratio and combined methods.
- **Phase mapping:** Phase 2 (Scoring Calibration)
- **Severity:** MEDIUM — inverse assumption may not hold universally

### 8. The "Area" Fallacy — PFC Is Not Homogeneous
- **Warning signs:** Averaging over entire PFC region masks sub-regional differences
- **Prevention:** Decomppose PFC into dorsolateral (BA 9/46), anterior (BA 10), ventromedial. Score sub-regions separately.
- **Phase mapping:** Phase 1 (PFC Atlas Module)
- **Severity:** LOW to MEDIUM — depends on how fine-grained scaffolding needs to be

## Domain-Specific Warnings

1. **Neurofeedback requires many sessions** — Single-session results don't prove the system works. Design for longitudinal tracking.
2. **Guanfacine comparison is metaphorical** — The system mimics the cognitive effects, not the pharmacology. Don't overclaim.
3. **CC-BY-NC license** — TRIBE v2 is non-commercial only. Can't productize without Meta's permission.
4. **Student research context** — Scholar cluster access is time-limited. Design experiments to run within SLURM job limits (3-day max).

---
*Pitfalls research: 2026-04-02*
