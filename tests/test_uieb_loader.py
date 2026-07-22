"""
Tests for UIEB dataset loader.
"""

import pytest
import torch
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from src.data.load_uieb import UIEBDataset, load_uieb_dataset


@pytest.fixture
def mock_uieb_dataset():
    """Create a mock UIEB dataset for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create directory structure
        raw_dir = tmpdir / "raw"
        ref_dir = tmpdir / "reference"
        raw_dir.mkdir()
        ref_dir.mkdir()
        
        # Create 10 mock image pairs
        for i in range(10):
            # Create degraded image (with blue cast)
            img_deg = Image.new("RGB", (256, 256))
            arr_deg = np.array(img_deg)
            arr_deg[:, :, 2] += 100  # Add blue cast
            arr_deg = np.clip(arr_deg, 0, 255).astype(np.uint8)
            Image.fromarray(arr_deg).save(raw_dir / f"image_{i:03d}.jpg")
            
            # Create reference image (normal)
            img_ref = Image.new("RGB", (256, 256), color=(100, 100, 100))
            img_ref.save(ref_dir / f"image_{i:03d}.jpg")
        
        yield tmpdir


def test_uieb_dataset_creation(mock_uieb_dataset):
    """Test creating UIEB dataset."""
    dataset = UIEBDataset(
        root_path=str(mock_uieb_dataset),
        image_size=128,
        split="train",
        seed=42,
    )
    assert len(dataset) > 0


def test_uieb_dataset_deterministic_split(mock_uieb_dataset):
    """Test deterministic splits."""
    dataset1 = UIEBDataset(
        root_path=str(mock_uieb_dataset),
        image_size=128,
        split="train",
        seed=42,
    )
    
    dataset2 = UIEBDataset(
        root_path=str(mock_uieb_dataset),
        image_size=128,
        split="train",
        seed=42,
    )
    
    assert dataset1.indices == dataset2.indices


def test_uieb_dataset_different_splits(mock_uieb_dataset):
    """Test different splits have different images."""
    train_dataset = UIEBDataset(
        root_path=str(mock_uieb_dataset),
        image_size=128,
        split="train",
        seed=42,
    )
    
    val_dataset = UIEBDataset(
        root_path=str(mock_uieb_dataset),
        image_size=128,
        split="val",
        seed=42,
    )
    
    # Train and val should have different images (non-overlapping)
    train_indices = set(train_dataset.indices)
    val_indices = set(val_dataset.indices)
    assert len(train_indices & val_indices) == 0  # No overlap


def test_uieb_dataset_getitem(mock_uieb_dataset):
    """Test getting a single item."""
    dataset = UIEBDataset(
        root_path=str(mock_uieb_dataset),
        image_size=128,
        split="train",
        seed=42,
    )
    
    img_deg, img_ref, img_id = dataset[0]
    
    assert isinstance(img_deg, torch.Tensor)
    assert isinstance(img_ref, torch.Tensor)
    assert isinstance(img_id, str)
    assert img_deg.shape == (3, 128, 128)
    assert img_ref.shape == (3, 128, 128)
    assert img_deg.min() >= 0 and img_deg.max() <= 1
    assert img_ref.min() >= 0 and img_ref.max() <= 1


def test_uieb_dataset_subset(mock_uieb_dataset):
    """Test subset size."""
    subset_size = 3
    dataset = UIEBDataset(
        root_path=str(mock_uieb_dataset),
        image_size=128,
        split="train",
        subset_size=subset_size,
        seed=42,
    )
    
    assert len(dataset) == subset_size


def test_load_uieb_dataset_function(mock_uieb_dataset):
    """Test load_uieb_dataset convenience function."""
    images_deg, images_ref, image_ids = load_uieb_dataset(
        image_size=128,
        subset_size=5,
        split="train",
        seed=42,
        root_path=str(mock_uieb_dataset),
    )
    
    assert images_deg.shape[0] == 5  # 5 images
    assert images_deg.shape == (5, 3, 128, 128)
    assert images_ref.shape == (5, 3, 128, 128)
    assert len(image_ids) == 5


def test_load_uieb_dataset_deterministic(mock_uieb_dataset):
    """Test load_uieb_dataset is deterministic."""
    images_deg1, images_ref1, ids1 = load_uieb_dataset(
        image_size=128,
        subset_size=3,
        split="train",
        seed=42,
        root_path=str(mock_uieb_dataset),
    )
    
    images_deg2, images_ref2, ids2 = load_uieb_dataset(
        image_size=128,
        subset_size=3,
        split="train",
        seed=42,
        root_path=str(mock_uieb_dataset),
    )
    
    assert torch.allclose(images_deg1, images_deg2)
    assert torch.allclose(images_ref1, images_ref2)
    assert ids1 == ids2


def test_uieb_dataset_invalid_path():
    """Test error handling for invalid path."""
    with pytest.raises(FileNotFoundError):
        UIEBDataset(
            root_path="/nonexistent/path",
            image_size=128,
            split="train",
        )


def test_uieb_dataset_eval_split_no_augmentation(mock_uieb_dataset):
    """Test eval split has consistent crops (center crop, no flip)."""
    dataset = UIEBDataset(
        root_path=str(mock_uieb_dataset),
        image_size=128,
        split="val",
        seed=42,
    )
    
    # Get same image twice
    img1_deg, img1_ref, _ = dataset[0]
    img2_deg, img2_ref, _ = dataset[0]
    
    # Should be identical (no augmentation on eval)
    assert torch.allclose(img1_deg, img2_deg)
    assert torch.allclose(img1_ref, img2_ref)
