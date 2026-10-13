import gzip
import hashlib
import struct
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import SequentialSampler, TensorDataset

from tools.mnist_data import (
    RESOURCE_TABLE,
    _selected_subset,
    external_data_root,
    load_mnist_subset,
    mnist_loader,
    read_idx_images,
    read_idx_labels,
)
from tools.prepare_mnist_data import prepare


def _image_idx(count=2, rows=28, cols=28, pixels=None):
    pixels = bytes(count * rows * cols) if pixels is None else pixels
    return struct.pack(">IIII", 2051, count, rows, cols) + pixels


def _label_idx(labels):
    return struct.pack(">II", 2049, len(labels)) + bytes(labels)


def test_idx_parsers_accept_exact_tiny_fixtures_and_normalize_dtype():
    images = read_idx_images(_image_idx(), expected_count=2)
    labels = read_idx_labels(_label_idx([0, 9]), expected_count=2)

    assert images.shape == (2, 1, 28, 28)
    assert images.dtype is torch.uint8 and images.device.type == "cpu"
    assert labels.tolist() == [0, 9]
    assert labels.dtype is torch.int64 and labels.device.type == "cpu"


def test_canonical_table_keeps_mnist_filename_count_and_md5_identities():
    assert {
        name: (resource.count, resource.md5)
        for name, resource in RESOURCE_TABLE.items()
    } == {
        "train-images-idx3-ubyte.gz": (60000, "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        "train-labels-idx1-ubyte.gz": (60000, "d53e105ee54ea40749a09fcbcd1e9432"),
        "t10k-images-idx3-ubyte.gz": (10000, "9fb629c4189551a2d022fa330f9573f3"),
        "t10k-labels-idx1-ubyte.gz": (10000, "ec29112dd5afa0611ce80d1b7f02629c"),
    }


@pytest.mark.parametrize(
    "payload, expected_count, message",
    [
        (b"\0" * 16, 1, "header"),
        (struct.pack(">IIII", 2051, 2, 28, 28), 2, "length"),
        (_image_idx(count=1), 2, "header"),
        (_image_idx(rows=27), 2, "header"),
        (_image_idx() + b"x", 2, "length"),
        (b"short", 1, "header"),
    ],
)
def test_image_idx_rejects_invalid_magic_count_geometry_and_length(
    payload, expected_count, message
):
    with pytest.raises(ValueError, match=message):
        read_idx_images(payload, expected_count=expected_count)


@pytest.mark.parametrize(
    "payload, expected_count, message",
    [
        (b"\0" * 8, 1, "header"),
        (struct.pack(">II", 2049, 2) + b"\0", 2, "length"),
        (_label_idx([1]), 2, "header"),
        (_label_idx([0]) + b"x", 1, "length"),
        (_label_idx([10]), 1, "label"),
        (_label_idx([255]), 1, "label"),
    ],
)
def test_label_idx_rejects_invalid_magic_count_length_and_range(
    payload, expected_count, message
):
    with pytest.raises(ValueError, match=message):
        read_idx_labels(payload, expected_count=expected_count)


def test_external_root_rejects_worktree_and_symlink_alias_before_creating_it(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    nested = repo / "cache"
    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)

    for candidate in (nested, alias / "nested"):
        with pytest.raises(ValueError, match="outside the worktree"):
            external_data_root(candidate, repo)
    assert not nested.exists()
    assert not (repo / "nested").exists()


def test_external_root_resolves_external_symlink_without_creating_it(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = tmp_path / "external-target"
    link = tmp_path / "external-link"
    link.symlink_to(target, target_is_directory=True)

    resolved = external_data_root(link, repo)
    assert resolved == target.resolve()
    assert not resolved.exists()


def test_loader_contract_keeps_final_partial_batches():
    train = mnist_loader(
        TensorDataset(torch.zeros(257, 1), torch.zeros(257, dtype=torch.int64))
    )
    test = mnist_loader(
        TensorDataset(torch.zeros(129, 1), torch.zeros(129, dtype=torch.int64))
    )

    assert [len(batch[0]) for batch in train] == [16] * 16 + [1]
    assert [len(batch[0]) for batch in test] == [16] * 8 + [1]
    assert train.batch_size == test.batch_size == 16
    assert not train.drop_last and isinstance(train.sampler, SequentialSampler)
    assert train.num_workers == 0 and not train.pin_memory


@pytest.mark.parametrize(("split", "count"), [("train", 257), ("test", 129)])
def test_selection_keeps_ordered_prefix_and_scales_only_selected_images(split, count):
    # Synthetic helper fixture tests selection semantics only; it never reaches
    # or bypasses canonical-file verification in load_mnist_subset.
    pixels = torch.arange(count + 1).remainder(255).to(torch.uint8).reshape(
        count + 1, 1, 1, 1
    )
    images = pixels.expand(-1, 1, 28, 28).clone()
    labels = torch.arange(count + 1, dtype=torch.int64) % 10

    dataset, metadata = _selected_subset(images, labels, split)

    assert len(dataset) == count
    assert metadata["indices"] == list(range(count))
    assert dataset[0][0].shape == (1, 28, 28)
    assert dataset[0][0].dtype is torch.float32
    assert dataset[0][0][0, 0, 0].item() == 0.0
    assert dataset[count - 1][0][0, 0, 0].item() == pytest.approx(
        (count - 1) % 255 / 255.0
    )
    assert dataset[count - 1][1].item() == (count - 1) % 10
    assert "images" not in metadata and "labels" not in metadata


def test_missing_data_fails_without_network_or_creating_any_files(tmp_path, monkeypatch):
    import urllib.request

    def unexpected_network(*args, **kwargs):
        raise AssertionError("load_mnist_subset must never acquire data")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected_network)
    root = tmp_path / "mnist"

    with pytest.raises(FileNotFoundError):
        load_mnist_subset(root, split="train", repo_root=tmp_path / "repo")
    assert not root.exists()


@pytest.mark.parametrize("split", ["train", "test"])
def test_load_rejects_invalid_canonical_bytes_without_rewriting_them(tmp_path, split):
    root = tmp_path / "mnist"
    root.mkdir()
    resource = RESOURCE_TABLE["train-images-idx3-ubyte.gz"]
    corrupted = b"not canonical MNIST"
    path = root / resource.filename
    path.write_bytes(corrupted)

    with pytest.raises(ValueError, match="checksum"):
        load_mnist_subset(root, split=split, repo_root=tmp_path / "repo")
    assert path.read_bytes() == corrupted


def test_prepare_bad_compressed_checksum_does_not_publish_or_clobber(tmp_path, monkeypatch):
    import tools.prepare_mnist_data as prepare_module

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            return gzip.compress(b"not the canonical IDX")

    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return Response()

    monkeypatch.setattr(prepare_module, "urlopen", fake_urlopen)
    root = tmp_path / "mnist"

    with pytest.raises(ValueError, match="checksum"):
        prepare(root, repo_root=tmp_path / "repo")
    assert calls == [
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
            30,
        )
    ]
    assert not root.exists()


def test_prepare_rejects_worktree_root_before_network_or_mkdir(tmp_path, monkeypatch):
    import tools.prepare_mnist_data as prepare_module

    repo = tmp_path / "repo"
    root = repo / "cache"
    calls = []
    monkeypatch.setattr(
        prepare_module,
        "urlopen",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="outside the worktree"):
        prepare(root, repo_root=repo)
    assert calls == []
    assert not root.exists()


def test_prepare_corrupt_gzip_does_not_publish(tmp_path, monkeypatch):
    import tools.prepare_mnist_data as prepare_module

    resource = RESOURCE_TABLE["train-images-idx3-ubyte.gz"]
    compressed = b"broken gzip"
    # Use the production identity only to let this external-boundary fixture reach
    # bounded gzip/IDX validation; this is not canonical-data evidence.
    monkeypatch.setitem(
        prepare_module.RESOURCE_TABLE,
        resource.filename,
        replace(resource, md5=hashlib.md5(compressed).hexdigest()),
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            assert size == 16 * 1024 * 1024 + 1
            return compressed

    monkeypatch.setattr(prepare_module, "urlopen", lambda url, timeout: Response())
    root = tmp_path / "mnist"
    with pytest.raises(ValueError, match="invalid gzip"):
        prepare(root, repo_root=tmp_path / "repo")
    assert not root.exists()


def test_prepare_existing_invalid_file_is_preserved_without_network(tmp_path, monkeypatch):
    import tools.prepare_mnist_data as prepare_module

    resource = RESOURCE_TABLE["train-images-idx3-ubyte.gz"]
    root = tmp_path / "mnist"
    root.mkdir()
    path = root / resource.filename
    path.write_bytes(b"invalid existing bytes")
    monkeypatch.setattr(
        prepare_module,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid existing data must fail before network")
        ),
    )

    with pytest.raises(ValueError, match="checksum"):
        prepare(root, repo_root=tmp_path / "repo")
    assert path.read_bytes() == b"invalid existing bytes"
