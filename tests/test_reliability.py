import tempfile
import os
from contextlib import contextmanager
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from src.local_ocr import LocalOCRRecognizer
from src.model import HandwrittenCNN
from src.corrector import HandwrittenCorrector, char_to_idx
from src.utils import split_train_validation, get_dataloaders


@contextmanager
def retained_test_directory():
    root = Path(os.environ.get('OCR_TEST_TMP', Path.home() / '.codex/cache/tasks/ocr-tests'))
    root.mkdir(parents=True, exist_ok=True)
    yield tempfile.mkdtemp(dir=root)


class FakeEMNIST:
    def __init__(self, root, split, train, download, transform): self.transform = transform
    def __len__(self): return 100
    def __getitem__(self, index):
        return self.transform(Image.new('L', (28, 28), color=index)), index


class ReliabilityTests(unittest.TestCase):
    def test_missing_checkpoint_refuses_inference(self):
        with retained_test_directory() as d:
            with self.assertRaises(FileNotFoundError):
                LocalOCRRecognizer(Path(d)/'missing.pth', device='cpu', warmup=False)

    def test_corrupt_checkpoint_refuses_inference(self):
        with retained_test_directory() as d:
            p = Path(d)/'broken.pth'; p.write_bytes(b'not a state dictionary')
            with self.assertRaises(RuntimeError): LocalOCRRecognizer(p, device='cpu', warmup=False)

    def test_compatible_state_dict_loads_in_eval_mode(self):
        # Random fixture tests loading/schema only, not recognition accuracy.
        with retained_test_directory() as d:
            p = Path(d)/'fixture.pth'; torch.save(HandwrittenCNN(62).state_dict(), p)
            r = LocalOCRRecognizer(p, device='cpu', warmup=False)
            self.assertFalse(r.model.training)

    def test_split_is_disjoint_complete_and_reproducible(self):
        a, b = split_train_validation(list(range(100)), list(range(100)), seed=42)
        a2, b2 = split_train_validation(list(range(100)), list(range(100)), seed=42)
        self.assertFalse(set(a.indices) & set(b.indices))
        self.assertEqual(set(a.indices) | set(b.indices), set(range(100)))
        self.assertEqual(a.indices, a2.indices); self.assertEqual(b.indices, b2.indices)

    def test_validation_has_clean_separate_transform(self):
        with patch('src.utils.datasets.EMNIST', FakeEMNIST):
            train, val, test = get_dataloaders(num_workers=0, download=False)
        self.assertIsNot(train.dataset.dataset, val.dataset.dataset)
        self.assertEqual(len(val.dataset.dataset.transform.transforms), 3)
        self.assertTrue(torch.equal(val.dataset[0][0], val.dataset[0][0]))

    def test_incompatible_dataset_views_rejected(self):
        with self.assertRaises(ValueError): split_train_validation([1], [1, 2])

    def test_dictionary_pruning_is_used_for_scoring(self):
        c = HandwrittenCorrector(); c.words_by_len = {2:['aa','zz']}; c.dictionary = {'aa','zz'}
        c.word_compatibility_score = lambda raw, word: 0 if word == 'aa' else -9999
        p = torch.full((62,), .001); p[char_to_idx['a']] = .15; p[char_to_idx['z']] = .65
        result, _ = c.joint_probability_decode([p,p], raw_str='aa')
        self.assertEqual(result.lower(), 'aa')


if __name__ == '__main__': unittest.main()
