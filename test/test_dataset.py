"""Run with python -m unittest discover -s test -p test_dataset.py.

Load the standalone NumPy dataset module without application/network setup.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

_path = Path(__file__).resolve().parents[1] / "exo" / "train" / "dataset.py"
_spec = importlib.util.spec_from_file_location("exo_dataset", _path)
dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dataset)


class TestBatchTargetLengths(unittest.TestCase):
  def test_unequal_sequences(self):
    inputs, targets, lengths = dataset.batch_with_lengths([[1, 2, 3], [4, 5]])
    np.testing.assert_array_equal(inputs, [[1, 2], [4, 5]])
    np.testing.assert_array_equal(targets, [[2, 3], [5, 0]])
    np.testing.assert_array_equal(lengths, [2, 1])

  def test_equal_sequences(self):
    inputs, targets, lengths = dataset.batch_with_lengths([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(lengths, [2, 2])
    self.assertEqual(int(lengths.sum()), targets.size)
    self.assertEqual(inputs.dtype, np.int32)
    self.assertEqual(targets.dtype, np.int32)

  def test_explicit_padding_width(self):
    inputs, targets, lengths = dataset.batch_with_lengths([[1, 2, 3], [4, 5]], maxlen=6)
    self.assertEqual(inputs.shape, (2, 5))
    np.testing.assert_array_equal(targets, [[2, 3, 0, 0, 0], [5, 0, 0, 0, 0]])
    np.testing.assert_array_equal(lengths, [2, 1])

  def test_no_targets_for_short_sequences(self):
    inputs, targets, lengths = dataset.batch_with_lengths([[], [7], [1, 2]])
    np.testing.assert_array_equal(lengths, [0, 0, 1])
    mask = np.arange(inputs.shape[1])[None, :] < lengths[:, None]
    np.testing.assert_array_equal(targets[mask], [2])

  def test_all_short_sequences_have_zero_targets(self):
    for tokens in ([[], []], [[1], [2]]):
      with self.subTest(tokens=tokens):
        inputs, targets, lengths = dataset.batch_with_lengths(tokens)
        self.assertEqual(inputs.shape, (2, 0))
        self.assertEqual(targets.shape, (2, 0))
        np.testing.assert_array_equal(lengths, [0, 0])

  def test_zero_token_ids_are_valid_targets(self):
    _, targets, lengths = dataset.batch_with_lengths([[1, 0, 2], [3, 0]])
    mask = np.arange(targets.shape[1])[None, :] < lengths[:, None]
    np.testing.assert_array_equal(targets[mask], [0, 2, 0])

  def test_padding_does_not_contribute_to_masked_loss(self):
    # This is the length-mask contract used by the MLX loss, not an MLX run.
    inputs, _, lengths = dataset.batch_with_lengths([[1, 2, 3], [4, 5]])
    mask = np.arange(inputs.shape[1])[None, :] < lengths[:, None]
    for padding_loss in (0.0, 100.0):
      per_token_loss = np.array([[1.0, 2.0], [3.0, padding_loss]])
      self.assertEqual(float((per_token_loss * mask).sum() / mask.sum()), 2.0)

  def test_dataset_iteration_preserves_target_counts(self):
    rows = [[1, 2, 3, 4], [5, 6], [7, 8, 9], [10, 11]]
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "test.jsonl"
      path.write_text("".join(json.dumps({"tokens": row}) + "\n" for row in rows))
      data = dataset.Dataset(path, preprocess=lambda item: item["tokens"])
      for uniform_length in (True, False):
        for batch_size in (1, 2):
          with self.subTest(uniform_length=uniform_length, batch_size=batch_size):
            batches = list(dataset.iterate_batches(data, batch_size, uniform_length=uniform_length))
            self.assertEqual(sum(int(lengths.sum()) for _, _, lengths in batches), 7)
            np.testing.assert_array_equal(np.concatenate([lengths for _, _, lengths in batches]), [3, 1, 2, 1])


if __name__ == "__main__":
  unittest.main()
