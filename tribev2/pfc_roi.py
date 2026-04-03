# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Atlas-based prefrontal cortex (PFC) vertex extraction for fsaverage5 cortical predictions.

Uses nilearn's Destrieux atlas to extract anatomically accurate PFC vertices from
TRIBE v2 cortical predictions, replacing hardcoded vertex ranges with proper parcellation.

PFC sub-regions are defined following the Destrieux atlas labeling convention:
- Dorsolateral PFC (dlPFC): superior and middle frontal gyri/sulci (BA 9/46)
- Ventrolateral PFC (vlPFC): inferior frontal gyrus — opercular, triangular, orbital (BA 44/45/47)
- Anterior PFC (aPFC): frontomarginal and transverse frontopolar (BA 10)
- Orbital/Ventromedial PFC (vmPFC): orbital, rectus, suborbital regions (BA 11/12/25)
"""

import logging
import typing as tp
from enum import Enum
from functools import lru_cache

import numpy as np

LOGGER = logging.getLogger(__name__)

# Number of vertices per hemisphere in fsaverage5
_FSAVERAGE5_VERTICES_PER_HEMI = 10242

# Destrieux atlas label indices for PFC sub-regions.
# These map to the integer values in the atlas parcellation arrays.
# Labels verified against nilearn.datasets.fetch_atlas_surf_destrieux() output.

# Dorsolateral PFC (dlPFC) — BA 9/46: superior frontal, middle frontal
_DLPFC_LABELS: list[str] = [
    "G_front_sup",       # label 16 — superior frontal gyrus
    "G_front_middle",    # label 15 — middle frontal gyrus
    "S_front_sup",       # label 55 — superior frontal sulcus
    "S_front_middle",    # label 54 — middle frontal sulcus
]

# Ventrolateral PFC (vlPFC) — BA 44/45/47: inferior frontal gyrus parts
_VLPFC_LABELS: list[str] = [
    "G_front_inf-Opercular",  # label 12 — pars opercularis (BA 44)
    "G_front_inf-Triangul",   # label 14 — pars triangularis (BA 45)
    "G_front_inf-Orbital",    # label 13 — pars orbitalis (BA 47)
    "S_front_inf",            # label 53 — inferior frontal sulcus
]

# Anterior PFC (aPFC) — BA 10: frontopolar
_APFC_LABELS: list[str] = [
    "G_and_S_frontomargin",       # label 1 — frontomarginal gyrus and sulcus
    "G_and_S_transv_frontopol",   # label 5 — transverse frontopolar
]

# Orbital / Ventromedial PFC (vmPFC) — BA 11/12/25: orbital and rectus
_VMPFC_LABELS: list[str] = [
    "G_orbital",           # label 24 — orbital gyri
    "G_rectus",            # label 31 — gyrus rectus
    "S_orbital_lateral",   # label 63 — lateral orbital sulcus
    "S_orbital_med-olfact",  # label 64 — medial orbital / olfactory sulcus
    "S_orbital-H_Shaped",  # label 65 — H-shaped orbital sulcus
    "S_suborbital",        # label 71 — suborbital sulcus
]

# All PFC labels combined
_ALL_PFC_LABELS: list[str] = _DLPFC_LABELS + _VLPFC_LABELS + _APFC_LABELS + _VMPFC_LABELS


class PFCRegion(Enum):
    """PFC sub-regions based on Destrieux atlas parcellation."""
    DORSOLATERAL = "dlPFC"
    VENTROLATERAL = "vlPFC"
    ANTERIOR = "aPFC"
    ORBITAL_VENTROMEDIAL = "vmPFC"
    ALL = "all_PFC"


# Map region enum to label lists
_REGION_LABELS: dict[PFCRegion, list[str]] = {
    PFCRegion.DORSOLATERAL: _DLPFC_LABELS,
    PFCRegion.VENTROLATERAL: _VLPFC_LABELS,
    PFCRegion.ANTERIOR: _APFC_LABELS,
    PFCRegion.ORBITAL_VENTROMEDIAL: _VMPFC_LABELS,
    PFCRegion.ALL: _ALL_PFC_LABELS,
}


@lru_cache(maxsize=1)
def _load_destrieux_atlas() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load Destrieux atlas parcellation for fsaverage5 surface.

    Returns
    -------
    map_left : np.ndarray of shape (10242,)
        Label index per vertex for left hemisphere.
    map_right : np.ndarray of shape (10242,)
        Label index per vertex for right hemisphere.
    labels : list[str]
        Human-readable label names indexed by label integer.
    """
    from nilearn import datasets

    destrieux = datasets.fetch_atlas_surf_destrieux()
    labels = [
        l.decode() if isinstance(l, bytes) else l
        for l in destrieux["labels"]
    ]
    map_left = np.asarray(destrieux["map_left"])
    map_right = np.asarray(destrieux["map_right"])

    assert map_left.shape == (_FSAVERAGE5_VERTICES_PER_HEMI,), (
        f"Expected {_FSAVERAGE5_VERTICES_PER_HEMI} vertices in left hemisphere, "
        f"got {map_left.shape[0]}"
    )
    assert map_right.shape == (_FSAVERAGE5_VERTICES_PER_HEMI,), (
        f"Expected {_FSAVERAGE5_VERTICES_PER_HEMI} vertices in right hemisphere, "
        f"got {map_right.shape[0]}"
    )

    LOGGER.info(
        "Loaded Destrieux atlas: %d labels, %d vertices/hemisphere",
        len(labels),
        _FSAVERAGE5_VERTICES_PER_HEMI,
    )
    return map_left, map_right, labels


def _label_names_to_indices(
    target_labels: list[str], all_labels: list[str]
) -> list[int]:
    """Convert human-readable label names to integer indices.

    Parameters
    ----------
    target_labels : list[str]
        Label names to look up.
    all_labels : list[str]
        Complete list of atlas labels (index = label integer).

    Returns
    -------
    indices : list[int]
        Integer indices corresponding to target_labels.

    Raises
    ------
    ValueError
        If any target label is not found in the atlas.
    """
    indices = []
    for name in target_labels:
        try:
            idx = all_labels.index(name)
        except ValueError:
            raise ValueError(
                f"Label '{name}' not found in Destrieux atlas. "
                f"Available labels: {all_labels}"
            )
        indices.append(idx)
    return indices


def get_pfc_mask(
    region: PFCRegion = PFCRegion.ALL,
    custom_labels: list[str] | None = None,
) -> np.ndarray:
    """Get a boolean mask for PFC vertices on the fsaverage5 mesh.

    The mask covers both hemispheres concatenated (left then right),
    matching the TRIBE v2 prediction output format of shape (T, 20484).

    Parameters
    ----------
    region : PFCRegion
        Which PFC sub-region to extract. Default: all PFC.
    custom_labels : list[str] or None
        Override with custom Destrieux label names. If provided, ``region``
        is ignored.

    Returns
    -------
    mask : np.ndarray of shape (20484,), dtype=bool
        True for vertices belonging to the requested PFC region.
    """
    map_left, map_right, labels = _load_destrieux_atlas()

    if custom_labels is not None:
        target_labels = custom_labels
    else:
        target_labels = _REGION_LABELS[region]

    label_indices = _label_names_to_indices(target_labels, labels)

    mask_left = np.isin(map_left, label_indices)
    mask_right = np.isin(map_right, label_indices)
    mask = np.concatenate([mask_left, mask_right])

    n_pfc = int(mask.sum())
    LOGGER.info(
        "PFC mask (%s): %d vertices (LH=%d, RH=%d) out of %d total",
        region.value if custom_labels is None else "custom",
        n_pfc,
        int(mask_left.sum()),
        int(mask_right.sum()),
        len(mask),
    )
    return mask


def get_pfc_vertex_indices(
    region: PFCRegion = PFCRegion.ALL,
) -> np.ndarray:
    """Get the integer indices of PFC vertices on the fsaverage5 mesh.

    Parameters
    ----------
    region : PFCRegion
        Which PFC sub-region. Default: all PFC.

    Returns
    -------
    indices : np.ndarray of shape (n_pfc,), dtype=int
        Vertex indices (0-based) into the (20484,) cortical prediction array.
    """
    mask = get_pfc_mask(region=region)
    return np.where(mask)[0]


def extract_pfc_vertices(
    predictions: np.ndarray,
    region: PFCRegion = PFCRegion.ALL,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Extract PFC vertex predictions from full cortical prediction array.

    Parameters
    ----------
    predictions : np.ndarray of shape (n_timesteps, 20484)
        Full cortical surface predictions from TRIBE v2.
    region : PFCRegion
        Which PFC sub-region to extract. Ignored if ``mask`` is provided.
    mask : np.ndarray of shape (20484,) or None
        Pre-computed boolean mask. If provided, ``region`` is ignored.
        Use this when calling repeatedly to avoid re-computing the mask.

    Returns
    -------
    pfc_predictions : np.ndarray of shape (n_timesteps, n_pfc_vertices)
        Predictions for the requested PFC vertices only.

    Raises
    ------
    ValueError
        If predictions don't have the expected shape.
    """
    if predictions.ndim != 2:
        raise ValueError(
            f"Expected 2D predictions (n_timesteps, n_vertices), "
            f"got shape {predictions.shape}"
        )
    n_expected = 2 * _FSAVERAGE5_VERTICES_PER_HEMI
    if predictions.shape[1] != n_expected:
        raise ValueError(
            f"Expected {n_expected} vertices in predictions, "
            f"got {predictions.shape[1]}"
        )

    if mask is None:
        mask = get_pfc_mask(region=region)

    return predictions[:, mask]


def get_pfc_labels_for_region(region: PFCRegion) -> list[str]:
    """Get the Destrieux label names for a PFC sub-region.

    Parameters
    ----------
    region : PFCRegion
        The sub-region to query.

    Returns
    -------
    labels : list[str]
        Destrieux atlas label names in this sub-region.
    """
    return list(_REGION_LABELS[region])


def get_all_region_masks() -> dict[PFCRegion, np.ndarray]:
    """Get masks for all PFC sub-regions.

    Returns
    -------
    masks : dict[PFCRegion, np.ndarray]
        Mapping from sub-region enum to boolean mask of shape (20484,).
        Does not include PFCRegion.ALL (use get_pfc_mask() for that).
    """
    regions = [r for r in PFCRegion if r != PFCRegion.ALL]
    masks = {region: get_pfc_mask(region=region) for region in regions}

    # Verify non-overlapping
    total = sum(m.sum() for m in masks.values())
    all_mask = get_pfc_mask(region=PFCRegion.ALL)
    if total != all_mask.sum():
        LOGGER.warning(
            "Sub-region masks overlap or miss vertices: "
            "sum=%d vs all=%d",
            total,
            int(all_mask.sum()),
        )

    return masks


def summarize_pfc_regions() -> dict[str, tp.Any]:
    """Get a summary of all PFC regions and their vertex counts.

    Returns
    -------
    summary : dict
        Keys are region names, values are dicts with 'n_vertices',
        'labels', 'pct_of_cortex'.
    """
    n_total = 2 * _FSAVERAGE5_VERTICES_PER_HEMI
    summary = {}
    for region in PFCRegion:
        mask = get_pfc_mask(region=region)
        n = int(mask.sum())
        summary[region.value] = {
            "n_vertices": n,
            "labels": get_pfc_labels_for_region(region),
            "pct_of_cortex": round(100.0 * n / n_total, 2),
        }
    return summary
