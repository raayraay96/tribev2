# Phase 1 Plan: PFC Atlas ROI Module

## Plan 1: Create `tribev2/pfc_roi.py`

### Goal
Replace hardcoded PFC vertex ranges with anatomically accurate atlas-based extraction.

### Implementation
1. Create `tribev2/pfc_roi.py` with Destrieux atlas integration
2. PFC labels confirmed from nilearn: 16 frontal labels covering ~1905 vertices/hemisphere
3. Support sub-regional decomposition (dorsolateral, ventrolateral, anterior, orbital/ventromedial)
4. Follow existing Pydantic + logging conventions

### Files to Create/Modify
- **[NEW]** `tribev2/pfc_roi.py` — Atlas-based PFC extraction module
- **[NEW]** `tests/test_pfc_roi.py` — Unit tests for PFC extraction

### Verification
- [ ] Module imports without errors
- [ ] `extract_pfc_vertices()` returns correct shape from (T, 20484) input
- [ ] PFC vertex count matches expected ~3810 (1905 per hemisphere)
- [ ] Sub-regional decomposition produces non-overlapping regions
- [ ] All tests pass

---
*Plan created: 2026-04-02*
