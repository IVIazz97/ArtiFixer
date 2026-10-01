# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import unittest


class Artifixer3DDistillationTests(unittest.TestCase):
    def test_release_distillation_defaults_to_scratch30k(self):
        run_source = Path("data_processing/run_artifixer3d.py").read_text()
        artifixer3d_source = Path("data_processing/artifixer3d.py").read_text()

        self.assertIn("--artifixer3d_steps", run_source)
        self.assertIn("default=30000", run_source)
        self.assertNotIn("resume_mode", run_source + artifixer3d_source)
        self.assertNotIn("image_path_override_fallback_to_original", run_source + artifixer3d_source)

    def test_resume_requires_explicit_base_checkpoint(self):
        artifixer3d_source = Path("data_processing/artifixer3d.py").read_text()

        self.assertIn("base_checkpoint: Path | None", artifixer3d_source)
        self.assertIn("if base_checkpoint is not None:", artifixer3d_source)
        self.assertIn('overrides.append(f"resume={base_checkpoint}")', artifixer3d_source)

    def test_distillation_passes_selected_indices_and_image_override(self):
        artifixer3d_source = Path("data_processing/artifixer3d.py").read_text()

        self.assertIn('f"selected_indices_file={paths.distillation_selected_indices_path}"', artifixer3d_source)
        self.assertIn('f"image_path_override={paths.override_image_dir.name}"', artifixer3d_source)
        self.assertIn('f"n_iterations={steps}"', artifixer3d_source)

    def test_frames_with_the_same_intrinsics_share_one_camera(self):
        import ast
        from types import SimpleNamespace

        import numpy as np

        source = Path("data_processing/artifixer3d.py").read_text()
        tree = ast.parse(source)
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "shared_camera_id")
        namespace = {"np": np, "Camera": object}
        exec(compile(ast.Module([helper], []), "artifixer3d.py", "exec"), namespace)  # threedgrut is not importable here
        cameras, ids = [], {}

        def camera(params, width=1600):
            return SimpleNamespace(id=len(cameras) + 1, model="OPENCV", width=width, height=900, params=np.array(params))

        photo = [1200.0, 1200.0, 800.0, 450.0, 0, 0, 0, 0]
        got = [namespace["shared_camera_id"](cameras, ids, camera(p, w)) for p, w in
               [(photo, 1600)] * 3 + [([1200.0 + 1e-9] + photo[1:], 1600), ([1100.0] + photo[1:], 1600), (photo, 1599)]]
        self.assertEqual(got, [1, 1, 1, 1, 2, 3])
        self.assertEqual([c.id for c in cameras], [1, 2, 3])
        self.assertEqual(source.count("shared_camera_id(\n"), 2)  # both the photos and the generated frames use it

    def test_continuation_replays_the_schedule_in_its_window(self):
        import ast
        import math

        source = Path("data_processing/artifixer3d.py").read_text()
        tree = ast.parse(source)
        helper = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "continuation_schedule_overrides"
        )
        namespace = {"math": math}
        exec(compile(ast.Module([helper], []), "artifixer3d.py", "exec"), namespace)  # threedgrut is not importable here
        replay = namespace["continuation_schedule_overrides"]
        try:
            from hydra import compose, initialize_config_dir
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("hydra is not installed")
        for name, fn in (("div", lambda a, b: a / b), ("int_list", lambda values: [int(v) for v in values])):
            if not OmegaConf.has_resolver(name):
                OmegaConf.register_new_resolver(name, fn)
        configs = Path("thirdparty/3DGRUT-ArtiFixer/configs").resolve()
        with initialize_config_dir(config_dir=str(configs), version_base=None):
            conf = compose(config_name="apps/colmap_3dgut_sparse_mcmc_lpips")
        strategy = OmegaConf.to_container(conf.strategy, resolve=True)
        scheduler = OmegaConf.to_container(conf.scheduler, resolve=True)
        self.assertEqual(conf.n_iterations, 30000)

        got = dict(o.split("=", 1) for o in replay(strategy, scheduler, 30000, 30000, 35000))
        # relocate/add 500..25000 and perturb 0..27500 of 30000 steps -> the same share of 30000..35000
        self.assertEqual(got["strategy.relocate.start_iteration"], "30083")
        self.assertEqual(got["strategy.relocate.end_iteration"], "34167")
        self.assertEqual(got["strategy.add.end_iteration"], "34167")
        self.assertEqual(got["strategy.perturb.start_iteration"], "30000")
        self.assertEqual(got["strategy.perturb.end_iteration"], "34583")
        self.assertNotIn("scheduler.density.max_steps", got)  # type skip
        self.assertEqual(got["scheduler.positions.max_steps"], "35000")

        def lr(step, lr_init, max_steps, lr_final=scheduler["positions"]["lr_final"]):
            t = min(max(step / max_steps, 0), 1)  # 3DGRUT exponential_scheduler
            return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)

        first, final = scheduler["positions"]["lr_init"], scheduler["positions"]["lr_final"]
        new = (float(got["scheduler.positions.lr_init"]), 35000)
        for step, fresh in ((30000, 0), (32500, 15000), (35000, 30000), (40000, 30000)):
            self.assertAlmostEqual(lr(step, *new) / lr(fresh, first, 30000), 1, places=9)
        self.assertGreater(lr(30000, first, 30000) / final, 0.99)  # what plain resume would use: the final LR

        half = dict(o.split("=", 1) for o in replay(strategy, scheduler, 30000, 50000, 55000, lr_scale=0.1))
        self.assertAlmostEqual(lr(50000, float(half["scheduler.positions.lr_init"]), 55000) / (0.1 * first), 1, places=9)
        self.assertEqual(half["strategy.add.start_iteration"], "50083")


if __name__ == "__main__":
    unittest.main()
