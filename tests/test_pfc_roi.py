# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for PFC atlas ROI extraction module."""

import numpy as np
import pytest

from tribev2.pfc_roi import (
    PFCRegion,
    extract_pfc_vertices,
    get_all_region_masks,
    get_pfc_labels_for_region,
    get_pfc_mask,
    get_pfc_vertex_indices,
    summarize_pfc_regions,
)

# fsaverage5 constants
N_VERTICES_PER_HEMI = 10242
N_TOTAL_VERTICES = 2 * N_VERTICES_PER_HEMI


class TestGetPFCMask:
    """Tests for get_pfc_mask()."""

    def test_mask_shape(self):
        mask = get_pfc_mask()
        assert mask.shape == (N_TOTAL_VERTICES,)
        assert mask.dtype == bool

    def test_mask_has_pfc_vertices(self):
        mask = get_pfc_mask()
        n_pfc = mask.sum()
        assert n_pfc > 0, "PFC mask should contain at least some vertices"
        # PFC should be roughly 15-25% of cortex
        pct = 100.0 * n_pfc / N_TOTAL_VERTICES
        assert 5.0 < pct < 40.0, f"PFC is {pct:.1f}% of cortex — seems wrong"

    def test_mask_bilateral(self):
        mask = get_pfc_mask()
        left = mask[:N_VERTICES_PER_HEMI].sum()
        right = mask[N_VERTICES_PER_HEMI:].sum()
        assert left > 0, "Left hemisphere should have PFC vertices"
        assert right > 0, "Right hemisphere should have PFC vertices"
        # Hemispheres should be roughly symmetric
        ratio = min(left, right) / max(left, right)
        assert ratio > 0.7, f"Hemispheres very asymmetric: {left} vs {right}"

    def test_subregion_masks(self):
        for region in PFCRegion:
            mask = get_pfc_mask(region=region)
            assert mask.shape == (N_TOTAL_VERTICES,)
            assert mask.sum() > 0, f"Region {region.value} has no vertices"

    def test_custom_labels(self):
        mask = get_pfc_mask(custom_labels=["G_front_sup"])
        assert mask.shape == (N_TOTAL_VERTICES,)
        assert mask.sum() > 0

    def test_invalid_custom_label_raises(self):
        with pytest.raises(ValueError, match="not found"):
            get_pfc_mask(custom_labels=["NONEXISTENT_LABEL"])


class TestGetPFCVertexIndices:
    """Tests for get_pfc_vertex_indices()."""

    def test_indices_valid(self):
        indices = get_pfc_vertex_indices()
        assert indices.ndim == 1
        assert np.all(indices >= 0)
        assert np.all(indices < N_TOTAL_VERTICES)

    def test_indices_match_mask(self):
        mask = get_pfc_mask()
        indices = get_pfc_vertex_indices()
        assert len(indices) == mask.sum()
        assert np.array_equal(indices, np.where(mask)[0])


class TestExtractPFCVertices:
    """Tests for extract_pfc_vertices()."""

    def test_extraction_shape(self):
        n_timesteps = 50
        predictions = np.random.randn(n_timesteps, N_TOTAL_VERTICES).astype(
            np.float32
        )
        pfc = extract_pfc_vertices(predictions)
        assert pfc.ndim == 2
        assert pfc.shape[0] == n_timesteps
        assert pfc.shape[1] > 0
        assert pfc.shape[1] < N_TOTAL_VERTICES

    def test_extraction_with_premask(self):
        n_timesteps = 10
        predictions = np.random.randn(n_timesteps, N_TOTAL_VERTICES).astype(
            np.float32
        )
        mask = get_pfc_mask()
        pfc = extract_pfc_vertices(predictions, mask=mask)
        assert pfc.shape == (n_timesteps, mask.sum())

    def test_extraction_with_region(self):
        predictions = np.random.randn(5, N_TOTAL_VERTICES).astype(np.float32)
        dlpfc = extract_pfc_vertices(predictions, region=PFCRegion.DORSOLATERAL)
        all_pfc = extract_pfc_vertices(predictions, region=PFCRegion.ALL)
        assert dlpfc.shape[1] < all_pfc.shape[1], (
            "dlPFC should have fewer vertices than all PFC"
        )

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="2D predictions"):
            extract_pfc_vertices(np.random.randn(100))

    def test_wrong_vertex_count_raises(self):
        with pytest.raises(ValueError, match="vertices"):
            extract_pfc_vertices(np.random.randn(10, 1000))


class TestGetAllRegionMasks:
    """Tests for get_all_region_masks()."""

    def test_non_overlapping(self):
        masks = get_all_region_masks()
        assert PFCRegion.ALL not in masks

        # Check no vertex belongs to more than one sub-region
        combined = np.zeros(N_TOTAL_VERTICES, dtype=int)
        for region, mask in masks.items():
            combined += mask.astype(int)
        assert np.all(combined <= 1), (
            f"Overlapping vertices found: max overlap = {combined.max()}"
        )

    def test_union_equals_all(self):
        masks = get_all_region_masks()
        union = np.zeros(N_TOTAL_VERTICES, dtype=bool)
        for mask in masks.values():
            union |= mask
        all_mask = get_pfc_mask(region=PFCRegion.ALL)
        assert np.array_equal(union, all_mask), (
            "Union of sub-regions should equal ALL PFC mask"
        )


class TestGetPFCLabels:
    """Tests for get_pfc_labels_for_region()."""

    def test_all_regions_have_labels(self):
        for region in PFCRegion:
            labels = get_pfc_labels_for_region(region)
            assert len(labels) > 0

    def test_all_is_superset(self):
        all_labels = set(get_pfc_labels_for_region(PFCRegion.ALL))
        for region in PFCRegion:
            if region == PFCRegion.ALL:
                continue
            region_labels = set(get_pfc_labels_for_region(region))
            assert region_labels.issubset(all_labels)


class TestSummarizePFCRegions:
    """Tests for summarize_pfc_regions()."""

    def test_summary_completeness(self):
        summary = summarize_pfc_regions()
        for region in PFCRegion:
            assert region.value in summary
            assert "n_vertices" in summary[region.value]
            assert "labels" in summary[region.value]
            assert "pct_of_cortex" in summary[region.value]
            assert summary[region.value]["n_vertices"] > 0
