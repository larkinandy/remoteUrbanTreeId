#!/usr/bin/env python3
"""Exercise the shared raster, temporal, and vector encoders on small tensors."""

from __future__ import annotations

import unittest

from model import model_components as components


@unittest.skipIf(components.TORCH_IMPORT_ERROR is not None, "PyTorch is not installed")
class ModelComponentSmokeTests(unittest.TestCase):
    """Verify output shapes and both supported raster normalization modes."""

    def test_image_encoder_batch_normalization(self) -> None:
        """Batch-normalized raster encoding returns one vector per record."""
        encoder = components.ImageEncoder(5, 16, 0.0, normalization="batch")
        output = encoder(components.torch.randn(2, 5, 28, 28))
        self.assertEqual(tuple(output.shape), (2, 16))

    def test_image_encoder_group_normalization(self) -> None:
        """Final-model group normalization supports nonstandard hidden widths."""
        encoder = components.ImageEncoder(3, 18, 0.0, normalization="group")
        output = encoder(components.torch.randn(2, 3, 20, 20))
        self.assertEqual(tuple(output.shape), (2, 18))
        final_norm = encoder.net[9]
        self.assertEqual(final_norm.num_channels, 18)
        self.assertEqual(final_norm.num_groups, 6)

    def test_masked_sequence_encoder(self) -> None:
        """Masked temporal encoding ignores padded positions and preserves shape."""
        encoder = components.MaskedSequenceEncoder(7, 16, 0.0)
        values = components.torch.randn(3, 6, 7)
        mask = components.torch.tensor(
            [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]],
            dtype=components.torch.float32,
        )
        output = encoder(values, mask)
        self.assertEqual(tuple(output.shape), (3, 16))
        self.assertTrue(components.torch.isfinite(output).all().item())

    def test_phenology_encoder(self) -> None:
        """Vector encoding returns the configured hidden dimension."""
        encoder = components.PhenologyEncoder(12, 16, 0.0)
        output = encoder(components.torch.randn(4, 12))
        self.assertEqual(tuple(output.shape), (4, 16))

    def test_unknown_normalization_is_rejected(self) -> None:
        """Misspelled normalization modes fail before training begins."""
        with self.assertRaises(ValueError):
            components.make_spatial_normalization(32, "instance")


if __name__ == "__main__":
    unittest.main()
