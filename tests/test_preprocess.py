"""Unit tests for the data pre-processing functions (M3 requirement)."""
from __future__ import annotations

import pytest
from PIL import Image

from src.data.preprocess import is_valid_image, resize_image, split_indices


class TestResizeImage:
    def test_resizes_to_square_target(self, sample_image):
        out = resize_image(sample_image, 224)
        assert out.size == (224, 224)

    def test_forces_rgb_mode(self):
        grayscale = Image.new("L", (64, 64), 128)
        out = resize_image(grayscale, 224)
        assert out.mode == "RGB"
        assert len(out.getpixel((0, 0))) == 3

    def test_handles_rgba_with_transparency(self):
        rgba = Image.new("RGBA", (50, 80), (255, 0, 0, 128))
        out = resize_image(rgba, 224)
        assert out.mode == "RGB"
        assert out.size == (224, 224)

    @pytest.mark.parametrize("size", [32, 64, 128, 224])
    def test_arbitrary_sizes(self, sample_image, size):
        assert resize_image(sample_image, size).size == (size, size)

    @pytest.mark.parametrize("bad", [0, -1, -224])
    def test_rejects_non_positive_size(self, sample_image, bad):
        with pytest.raises(ValueError):
            resize_image(sample_image, bad)


class TestSplitIndices:
    def test_partitions_are_disjoint_and_complete(self):
        train, val, test = split_indices(1000, 0.8, 0.1, 0.1, seed=42)
        assert len(train) + len(val) + len(test) == 1000
        assert set(train) | set(val) | set(test) == set(range(1000))
        assert not (set(train) & set(val))
        assert not (set(train) & set(test))
        assert not (set(val) & set(test))

    def test_respects_requested_ratios(self):
        train, val, test = split_indices(1000, 0.8, 0.1, 0.1, seed=42)
        assert len(train) == 800
        assert len(val) == 100
        assert len(test) == 100

    def test_is_deterministic_for_a_given_seed(self):
        assert split_indices(500, 0.8, 0.1, 0.1, seed=7) == split_indices(500, 0.8, 0.1, 0.1, seed=7)

    def test_different_seeds_shuffle_differently(self):
        a, _, _ = split_indices(500, 0.8, 0.1, 0.1, seed=1)
        b, _, _ = split_indices(500, 0.8, 0.1, 0.1, seed=2)
        assert a != b

    def test_rejects_ratios_that_do_not_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_indices(100, 0.7, 0.1, 0.1)

    def test_handles_empty_dataset(self):
        assert split_indices(0, 0.8, 0.1, 0.1) == ([], [], [])

    def test_no_index_is_dropped_on_rounding_remainder(self):
        # 7 * 0.8 = 5.6 -> rounding must not lose the remainder.
        train, val, test = split_indices(7, 0.8, 0.1, 0.1, seed=3)
        assert sorted(train + val + test) == list(range(7))


class TestIsValidImage:
    def test_accepts_a_real_jpeg(self, tmp_path, sample_image):
        path = tmp_path / "ok.jpg"
        sample_image.save(path, "JPEG")
        assert is_valid_image(path) is True

    def test_rejects_zero_byte_file(self, tmp_path):
        path = tmp_path / "empty.jpg"
        path.write_bytes(b"")
        assert is_valid_image(path) is False

    def test_rejects_non_image_bytes(self, tmp_path):
        path = tmp_path / "junk.jpg"
        path.write_bytes(b"this is definitely not a jpeg")
        assert is_valid_image(path) is False

    def test_rejects_truncated_jpeg(self, tmp_path, sample_image):
        good = tmp_path / "good.jpg"
        sample_image.save(good, "JPEG")
        truncated = tmp_path / "truncated.jpg"
        truncated.write_bytes(good.read_bytes()[:40])
        assert is_valid_image(truncated) is False

    def test_rejects_missing_path(self, tmp_path):
        assert is_valid_image(tmp_path / "nope.jpg") is False
