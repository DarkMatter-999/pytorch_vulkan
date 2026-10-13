"""Verified, bounded MNIST IDX parsing and deterministic subset loading."""

from __future__ import annotations

import gzip
import hashlib
import io
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


MAX_COMPRESSED_BYTES = 16 * 1024 * 1024
DEFAULT_DATA_ROOT = Path("/tmp/opencode/pytorch-vulkan-mnist")
MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"


@dataclass(frozen=True)
class Resource:
    filename: str
    md5: str
    count: int
    kind: str

    @property
    def idx_size(self) -> int:
        if self.kind == "images":
            return 16 + self.count * 28 * 28
        return 8 + self.count


# Canonical torchvision v0.19.0 MNIST identity; MD5 is compatibility identity.
RESOURCE_TABLE = {
    "train-images-idx3-ubyte.gz": Resource(
        "train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873", 60000, "images"
    ),
    "train-labels-idx1-ubyte.gz": Resource(
        "train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432", 60000, "labels"
    ),
    "t10k-images-idx3-ubyte.gz": Resource(
        "t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3", 10000, "images"
    ),
    "t10k-labels-idx1-ubyte.gz": Resource(
        "t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c", 10000, "labels"
    ),
}
SPLITS = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", 257),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", 129),
}


def external_data_root(path: Path, repo_root: Path) -> Path:
    """Resolve a data location and reject the worktree or any alias into it."""
    resolved = Path(path).expanduser().resolve()
    repository = Path(repo_root).expanduser().resolve()
    if resolved.is_relative_to(repository):
        raise ValueError("MNIST data root must be outside the worktree")
    return resolved


def read_idx_images(payload: bytes, *, expected_count: int) -> torch.Tensor:
    if len(payload) < 16:
        raise ValueError("invalid image IDX header: header is truncated")
    magic, count, rows, cols = struct.unpack(">IIII", payload[:16])
    if (magic, count, rows, cols) != (2051, expected_count, 28, 28):
        raise ValueError("invalid image IDX header")
    if len(payload) != 16 + count * 28 * 28:
        raise ValueError("invalid image IDX length")
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8, offset=16).reshape(
        count, 1, 28, 28
    )


def read_idx_labels(payload: bytes, *, expected_count: int) -> torch.Tensor:
    if len(payload) < 8:
        raise ValueError("invalid label IDX header: header is truncated")
    magic, count = struct.unpack(">II", payload[:8])
    if magic != 2049 or count != expected_count:
        raise ValueError("invalid label IDX header")
    if len(payload) != 8 + count:
        raise ValueError("invalid label IDX length")
    labels = torch.frombuffer(bytearray(payload), dtype=torch.uint8, offset=8).to(
        dtype=torch.int64
    )
    if bool(((labels < 0) | (labels > 9)).any()):
        raise ValueError("invalid label IDX label value; expected labels in [0, 9]")
    return labels


def _read_resource(path: Path, resource: Resource) -> tuple[bytes, dict]:
    with path.open("rb") as stream:
        compressed = stream.read(MAX_COMPRESSED_BYTES + 1)
    if len(compressed) > MAX_COMPRESSED_BYTES:
        raise ValueError(f"MNIST compressed resource exceeds size bound: {path.name}")
    try:
        return validate_compressed_resource(compressed, resource)
    except ValueError as error:
        # Preserve canonical file context in diagnostics from the common validator.
        if resource.filename in str(error):
            raise
        raise ValueError(f"invalid MNIST resource {path.name}: {error}") from error


class _SelectedMNIST(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def _selected_subset(images: torch.Tensor, labels: torch.Tensor, split: str):
    _, _, selection_count = SPLITS[split]
    if images.shape[0] != labels.shape[0] or images.shape[0] < selection_count:
        raise ValueError(f"insufficient or mismatched parsed MNIST items for {split}")
    selected_images = images[:selection_count].to(dtype=torch.float32).div_(255.0)
    selected_labels = labels[:selection_count].clone()
    histogram = torch.bincount(selected_labels, minlength=10).tolist()
    input_hash = hashlib.sha256()
    input_hash.update(images[:selection_count].numpy().tobytes())
    input_hash.update(selected_labels.numpy().tobytes())
    metadata = {
        "dataset": "MNIST",
        "split": split,
        "count": selection_count,
        "indices": list(range(selection_count)),
        "label_histogram": {str(label): count for label, count in enumerate(histogram)},
        "input_digest_sha256": input_hash.hexdigest(),
    }
    return _SelectedMNIST(selected_images, selected_labels), metadata


def load_mnist_subset(root: Path, *, split: str, repo_root: Path) -> tuple[Dataset, dict]:
    if split not in SPLITS:
        raise ValueError("split must be 'train' or 'test'")
    resolved_root = external_data_root(root, repo_root)
    image_name, label_name, _ = SPLITS[split]
    payloads = {}
    identities = {}
    # Verify the complete canonical dataset, not merely the selected split.
    for name, resource in RESOURCE_TABLE.items():
        path = resolved_root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing canonical MNIST resource: {path}")
        payloads[name], identities[name] = _read_resource(path, resource)

    images = read_idx_images(payloads[image_name], expected_count=RESOURCE_TABLE[image_name].count)
    labels = read_idx_labels(payloads[label_name], expected_count=RESOURCE_TABLE[label_name].count)
    dataset, metadata = _selected_subset(images, labels, split)
    metadata["data_identity"] = {
        "canonical_resource_table": "torchvision-v0.19.0-MNIST",
        "resources": identities,
    }
    return dataset, metadata


def validate_compressed_resource(compressed: bytes, resource: Resource) -> tuple[bytes, dict]:
    """Check a downloaded resource with the same limits and identity as a cache file."""
    if len(compressed) > MAX_COMPRESSED_BYTES:
        raise ValueError("MNIST compressed resource exceeds size bound")
    md5 = hashlib.md5(compressed).hexdigest()
    if md5 != resource.md5:
        raise ValueError(f"MNIST checksum mismatch for {resource.filename}")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as stream:
            payload = stream.read(resource.idx_size + 1)
    except (OSError, EOFError) as error:
        raise ValueError(f"invalid gzip resource {resource.filename}: {error}") from error
    if len(payload) > resource.idx_size:
        raise ValueError(f"MNIST decompressed resource exceeds size bound: {resource.filename}")
    if resource.kind == "images":
        read_idx_images(payload, expected_count=resource.count)
    else:
        read_idx_labels(payload, expected_count=resource.count)
    return payload, {
        "md5": md5,
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "compressed_bytes": len(compressed),
    }


def mnist_loader(dataset: Dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
