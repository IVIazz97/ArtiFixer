# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from flashsplat.solver import multi_instance_opt


class MultiInstanceOptTest(unittest.TestCase):
    def test_assigns_gaussian_to_its_dominant_object(self):
        # 3 Gaussians: the first contributes only to background, the second only to
        # object 1, the third only to object 2.
        contrib = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # background
                [0.0, 1.0, 0.0],  # object 1
                [0.0, 0.0, 1.0],  # object 2
            ]
        )
        labels = multi_instance_opt(contrib, gamma=0.0)

        self.assertTrue(labels[0].tolist() == [True, False, False])
        self.assertTrue(labels[1].tolist() == [False, True, False])
        self.assertTrue(labels[2].tolist() == [False, False, True])

    def test_ties_and_majority(self):
        # Gaussian 0 leans to object 1, Gaussian 1 leans to background.
        contrib = torch.tensor([[1.0, 9.0], [9.0, 1.0]])
        labels = multi_instance_opt(contrib, gamma=0.0)
        self.assertEqual(labels[1].tolist(), [True, False])

    def test_gamma_is_monotone_towards_background(self):
        torch.manual_seed(0)
        contrib = torch.rand(4, 256)

        previous = multi_instance_opt(contrib, gamma=-1.0)
        for gamma in (-0.5, 0.0, 0.5, 1.0):
            current = multi_instance_opt(contrib, gamma=gamma)
            # Raising gamma biases towards background, so the set of Gaussians claimed by
            # each object may only shrink.
            self.assertTrue(
                bool((current[1:] <= previous[1:]).all()),
                f"gamma={gamma} grew an object relative to the previous gamma",
            )
            previous = current

    def test_unseen_object_gets_no_gaussians(self):
        contrib = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        labels = multi_instance_opt(contrib, gamma=0.0)
        self.assertEqual(labels[1].tolist(), [False, False])

    def test_rejects_wrong_rank(self):
        with self.assertRaises(ValueError):
            multi_instance_opt(torch.zeros(5), gamma=0.0)


if __name__ == "__main__":
    unittest.main()
