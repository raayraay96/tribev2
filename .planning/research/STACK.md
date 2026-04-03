# Stack Research — BCI Cognitive Scaffolding

**Research Date:** 2026-04-02
**Domain:** Closed-loop Brain-Computer Interface for ADHD Cognitive Scaffolding

## Recommended Stack (Already In Place)

| Component | Choice | Version | Rationale | Confidence |
|-----------|--------|---------|-----------|------------|
| Brain Encoding Model | Meta TRIBE v2 | 177.2M params | State-of-the-art multimodal brain encoding; pre-trained on fsaverage5 | ★★★★★ |
| ML Framework | PyTorch | 2.5.1+cu121 | Model requirement; torch.compiler APIs | ★★★★★ |
| Training | PyTorch Lightning | latest | Handles SLURM, checkpointing, metrics | ★★★★☆ |
| Cortical Mesh | fsaverage5 | ~20,484 vertices | TRIBE v2 standard; sufficient PFC resolution | ★★★★☆ |
| PFC Atlas | Destrieux or Schaefer | nilearn built-in | Surface-based, better than hardcoded vertex ranges | ★★★★★ |
| Brain Viz | nilearn + PyVista | latest | Already integrated; standard neuroscience tools | ★★★★★ |
| Experiment Config | Pydantic + exca | 0.0.2 | Meta's framework; already embedded | ★★★★☆ |

## Critical Stack Addition: PFC Atlas Mapping

**Current approach (fragile):** Hardcoded vertex indices LH 0-3000, RH 10242-13242
**Recommended approach:** Use nilearn's Destrieux or Schaefer atlas in fsaverage5 space

```python
from nilearn import datasets
destrieux = datasets.fetch_atlas_surf_destrieux()
# Extract PFC labels corresponding to BA 9/10/46
# Use label indices instead of hardcoded vertex ranges
```

**Why Schaefer over Destrieux for this project:**
- Schaefer atlas is organized by functional networks (frontoparietal, default mode)
- Better correspondence to the cognitive control regions relevant to ADHD
- Available in fsaverage5 space directly via nilearn

## What NOT to Use

| Technology | Reason |
|-----------|--------|
| Real-time fMRI (rtfMRI) | TRIBE v2 predicts from stimuli, not from actual fMRI scans |
| BCI2000 / OpenViBE | Designed for EEG BCIs, not encoding model pipelines |
| Multi-GPU training | Model is 177M params; overkill and complicates SLURM setup |
| Cloud deployment | Research system on Scholar; unnecessary complexity |

## Stack Gaps to Fill

1. **PFC ROI definition** — nilearn atlas integration (not yet implemented)
2. **Temporal smoothing validation** — scipy.signal or custom Gaussian convolution
3. **State machine framework** — Python enum + dataclass (lightweight, no external dep)
4. **Experiment logging** — W&B already integrated but not used for scaffolding runs

---
*Stack research: 2026-04-02*
