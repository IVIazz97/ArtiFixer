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


if __name__ == "__main__":
    unittest.main()
