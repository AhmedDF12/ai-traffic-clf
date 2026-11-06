import os
import random

import numpy as np

from traffic_clf.seeding import seed_everything


def test_numpy_and_python_repeat_the_same_draws() -> None:
    seed_everything(123)
    numpy_a = np.random.rand(4)
    python_a = [random.random() for _ in range(3)]
    seed_everything(123)
    numpy_b = np.random.rand(4)
    python_b = [random.random() for _ in range(3)]
    assert np.array_equal(numpy_a, numpy_b)
    assert python_a == python_b


def test_different_seeds_change_the_draw_and_export_hash_seed() -> None:
    seed_everything(1)
    first = np.random.rand(2)
    seed_everything(2)
    second = np.random.rand(2)
    assert not np.array_equal(first, second)
    assert os.environ["PYTHONHASHSEED"] == "2"
