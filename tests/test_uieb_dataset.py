"""
Test suite for UIEB dataset loader.
Verifies underwater image dataset functionality.
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image
import torch
from src.data.uieb_dataset import UIEBDataset


def create_test_uieb_structure(tmp_dir: Path) -> Path:
    """Create minimal test UIEB structure with dummy images."""
    uieb_dir = tmp_dir / "UIEB"
    raw_dir = uieb_dir / "raw-890"
    ref_dir = uieb_dir / "reference-890"
    raw_dir.mkdir(parents=True)
    ref_dir.mkdir(parents=True)
    
    # Create 10 test image pairs
    for i in range(10):
        raw_img = Image.new("RGB", (64, 64), color=(100, 150, 200))  # bluish (underwater)
        ref_img = Image.new("RGB", (64, 64), color=(200, 200, 200))  # neutral (enhanced)
        
        raw_img.save(raw_dir / f"{i:03d}.png")
        ref_img.save(ref_dir / f"{i:03d}.png")
    
    return uieb_dir


def test_uieb_dataset_initialization():
    """Test UIEBDataset initialization with valid paths."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        uieb_dir = create_test_uieb_structure(Path(tmp_dir))
        
        dataset = UIEBDataset(root=uieb_dir.parent, split="train")
        assert dataset is not None
        assert len(dataset) > 0


def test_uieb_dataset_split_sizes():
    """Test that train/val/test splits have expected sizes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        uieb_dir = create_test_uieb_structure(Path(tmp_dir))
        
        train_ds = UIEBDataset(root=uieb_dir.parent, split="train", train_ratio=0.8)
        val_ds = UIEBDataset(root=uieb_dir.parent, split="val", train_ratio=0.8)
        test_ds = UIEBDataset(root=uieb_dir.parent, split="test", train_ratio=0.8)
        
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == 10, "Total samples should be 10"
        assert len(train_ds) == 8, "Train should have 80% = 8 samples"
        assert len(val_ds) == 1, "Val should have 10% = 1 sample"
        assert len(test_ds) == 1, "Test should have 10% = 1 sample"


def test_uieb_dataset_getitem():
    """Test that __getitem__ returns valid (raw, reference) tuple."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        uieb_dir = create_test_uieb_structure(Path(tmp_dir))
        
        dataset = UIEBDataset(root=uieb_dir.parent, split="train")
        
        raw_img, ref_img = dataset[0]
        
        assert isinstance(raw_img, Image.Image), "raw_img should be PIL Image"
        assert isinstance(ref_img, Image.Image), "ref_img should be PIL Image"
        assert raw_img.mode == "RGB", "raw_img should be RGB"
        assert ref_img.mode == "RGB", "ref_img should be RGB"


def test_uieb_dataset_reproducible_split():
    """Test that split is reproducible with same seed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        uieb_dir = create_test_uieb_structure(Path(tmp_dir))
        
        ds1 = UIEBDataset(root=uieb_dir.parent, split="train", seed=42)
        ds2 = UIEBDataset(root=uieb_dir.parent, split="train", seed=42)
        
        # Both should have same length (reproducible split)
        assert len(ds1) == len(ds2), "Same seed should produce same split"
        
        # Get same index from both and compare
        raw1, ref1 = ds1[0]
        raw2, ref2 = ds2[0]
        
        # Compare image data (should be from same source file)
        assert raw1.tobytes() == raw2.tobytes(), "Same seed should select same images"


def test_uieb_dataset_missing_directory():
    """Test that UIEBDataset raises error for missing directories."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        empty_uieb = tmp_path / "UIEB"
        empty_uieb.mkdir()
        
        with pytest.raises(FileNotFoundError):
            UIEBDataset(root=tmp_path, split="train")


def test_uieb_dataset_torch_compatibility():
    """Test that dataset works with PyTorch DataLoader."""
    from torch.utils.data import DataLoader
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        uieb_dir = create_test_uieb_structure(Path(tmp_dir))
        dataset = UIEBDataset(root=uieb_dir.parent, split="train")
        
        # Create DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Iterate once
        for batch_idx, batch_data in enumerate(loader):
            if batch_idx == 0:
                assert len(batch_data) == 2, "Batch should have raw and ref"
                # Note: DataLoader doesn't automatically batch PIL Images,
                # so we just check the data structure
                break
