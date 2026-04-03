# Phase 1: PFC Atlas ROI Module - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — discuss skipped per autonomous workflow)

<domain>
## Phase Boundary

Replace hardcoded PFC vertex ranges (LH: 0-3000, RH: 10242-13242) with atlas-based PFC extraction using nilearn's Destrieux atlas on fsaverage5 surface. Create `tribev2/pfc_roi.py` module with functions for loading atlas, extracting PFC vertices by anatomical label, and supporting sub-regional decomposition (dorsolateral BA 9/46, anterior BA 10, ventromedial).

</domain>

<decisions>
## Implementation Decisions

### Agent's Discretion
All implementation choices are at the agent's discretion — infrastructure phase. Key guidance from research:
- Use nilearn `fetch_atlas_surf_destrieux()` for surface-based parcellation
- Target frontal labels containing 'front' in the Destrieux naming convention
- Return both a boolean mask and explicit vertex index arrays
- Follow existing codebase Pydantic config conventions
- Module should be importable as `from tribev2.pfc_roi import extract_pfc_vertices`
- Include docstrings in NumPy/Google hybrid style matching existing code

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tribev2/utils_fmri.py`: TribeSurfaceProjector already uses nilearn, FSAVERAGE_SIZES dict, load_mni_mesh()
- `tribev2/model.py`: FmriEncoder produces (B, T, n_vertices) predictions
- `tribev2/demo_utils.py`: TribeModel.predict() returns (n_timesteps, 20484) numpy array

### Established Patterns
- Pydantic BaseModel for all configuration classes with `extra="forbid"`
- Module-level LOGGER = logging.getLogger(__name__)
- Type hints throughout (Python 3.10+ union syntax)
- black formatting (88 char line length)

### Integration Points
- Output of TribeModel.predict() → input to pfc_roi.extract_pfc_vertices()
- fsaverage5 mesh: 10,242 vertices per hemisphere, 20,484 total
- Left hemisphere: vertices 0-10241, Right hemisphere: vertices 10242-20483

</code_context>

<specifics>
## Specific Ideas

- Destrieux atlas labels for PFC should include: superior frontal, middle frontal, inferior frontal (pars opercularis, triangularis, orbitalis), frontomarginal, orbital gyri
- Function should accept a list of target labels or default to all PFC labels
- Return type should include both mask and vertex indices for flexible downstream use

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
