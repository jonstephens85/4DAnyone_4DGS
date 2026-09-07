"""Selection and malformed-input checks for the optional results viewer."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("view_rerun", Path(__file__).parents[1] / "scripts/view_rerun.py")
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.cameras = [dict(camera_id=layer * 16 + i, layer_index=layer, yaw=i * 22.5)
                        for layer in range(3) for i in range(16)]

    def test_default_covers_yaw_and_layers(self):
        selected = viewer.select_cameras(self.cameras, None, None, 6)
        self.assertEqual(len({c["yaw"] for c in selected}), 6)
        self.assertEqual({c["layer_index"] for c in selected}, {0, 1, 2})

    def test_explicit_ids_preserve_order_and_validate_layer(self):
        selected = viewer.select_cameras(self.cameras, [18, 16, 18], 1, 1)
        self.assertEqual([c["camera_id"] for c in selected], [18, 16])
        with self.assertRaises(ValueError):
            viewer.select_cameras(self.cameras, [0], 1, 6)
        with self.assertRaises(ValueError):
            viewer.select_cameras(self.cameras, None, 99, 6)

    def test_pages_cover_every_camera_once_with_neighboring_angles(self):
        rings = viewer.camera_pages(list(reversed(self.cameras)))
        ids = []
        for layer, pages in rings:
            self.assertEqual(len(pages), 4)
            for page in pages:
                self.assertEqual(len(page), 4)
                self.assertEqual({c["layer_index"] for c in page}, {layer})
                yaws = [c["yaw"] for c in page]
                self.assertEqual(yaws, sorted(yaws))
                ids.extend(c["camera_id"] for c in page)
        self.assertEqual(sorted(ids), list(range(48)))
        partial = viewer.camera_pages(self.cameras[:5])
        self.assertEqual([len(page) for page in partial[0][1]], [4, 1])

    def test_geometry_rejects_reflection_and_bad_intrinsics(self):
        camera = dict(camera_id=0, K=np.eye(3), camera_to_world=np.eye(4), image_width=704, image_height=1280)
        viewer.validate_geometry([camera])
        camera["camera_to_world"][0, 0] = -1
        with self.assertRaises(ValueError):
            viewer.validate_geometry([camera])
        camera["camera_to_world"] = np.eye(4)
        camera["K"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            viewer.validate_geometry([camera])


if __name__ == "__main__":
    unittest.main()
