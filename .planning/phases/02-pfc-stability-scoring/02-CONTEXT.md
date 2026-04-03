# Phase 2: PFC Stability Scoring - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — discuss skipped per autonomous workflow)

<domain>
## Phase Boundary

Implement three PFC stability scoring methods (inverse_variance, activation_ratio, combined) with per-session baseline calibration. Each method takes PFC vertex predictions and produces a scalar stability score ∈ [0, 1]. Include session baseline normalization using z-scores to handle fMRI prediction unreliability.

</domain>

<decisions>
## Implementation Decisions

### Agent's Discretion
- inverse_variance: `1 / (1 + variance * VARIANCE_SCALE)` where VARIANCE_SCALE is configurable
- activation_ratio: `mean(PFC) / mean(whole_brain)` normalized to [0,1]
- combined: `alpha * inverse_variance + (1 - alpha) * temporal_stability` with configurable alpha
- Per-session baseline: z-score normalization using first N timesteps as calibration window
- Follow Pydantic config pattern from existing codebase
- Module should integrate seamlessly with Phase 1 pfc_roi.py output

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tribev2/pfc_roi.py`: extract_pfc_vertices() returns (T, n_pfc) arrays, get_pfc_mask() returns boolean masks
- `tribev2/model.py`: FmriEncoder forward pass produces per-vertex predictions
- numpy/scipy for statistical operations

### Integration Points
- Input: output of pfc_roi.extract_pfc_vertices() → (n_timesteps, n_pfc_vertices)
- Output: (n_timesteps,) array of stability scores ∈ [0, 1]
- Downstream: Phase 4 state machine consumes these scores

</code_context>

<specifics>
## Specific Ideas
- VARIANCE_SCALE default should be 100 (from CLAUDE.md scaffolding pipeline)
- Combined scoring should support configurable weights
- All scoring functions should be stateless (baseline is a separate transform)

</specifics>

<deferred>
## Deferred Ideas
None.
</deferred>
