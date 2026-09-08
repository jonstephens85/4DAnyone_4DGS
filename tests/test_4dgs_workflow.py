"""CPU checks for the public 4DGS entry points (run in work/4dgs-env)."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_4dgaussians import select_views
from _4dgs import model_options
import train_4dgs
import render_4dgs


class WorkflowTests(unittest.TestCase):
    def test_all_cameras_and_optional_holdout(self):
        train, test, selected, video = select_views(range(48))
        self.assertEqual((len(train), test, len(selected), video), (48, [], 48, 0))
        train, test, selected, video = select_views(range(48), test_views=[40], video_camera=40)
        self.assertEqual((len(train), test, len(selected), video), (47, [40], 48, 40))
        self.assertNotIn(40, train)

    def test_invalid_selection(self):
        for options in [dict(train_views=[]), dict(train_views=[0], test_views=[0]),
                        dict(test_views=[99]), dict(train_views=[0], video_camera=1)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                select_views(range(48), **options)

    def test_historical_default_compatibility(self):
        from mmengine import Config
        old = Config.fromfile(str(train_4dgs.ROOT / 'integrations/4dgaussians/quality3.py'))
        for key, value in train_4dgs.settings(30000, 3000, 2).items():
            self.assertEqual(value, old[key])
        changed = train_4dgs.settings(20000, 1000, 1)['OptimizationParams']
        self.assertEqual(changed['densify_until_iter'], 19999)
        self.assertEqual(changed['position_lr_max_steps'], 20000)
        self.assertEqual(changed['batch_size'], 1)

    def test_training_saves_options_and_render_skips_empty_eval(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data'
            data.mkdir()
            (data / 'fdanyone.json').write_text(json.dumps(dict(train_views=list(range(48)), test_views=[])))
            model = root / 'model'
            argv = ['train_4dgs.py', '--data', str(data), '--output', str(model), '--iterations', '20000']
            with patch.object(sys, 'argv', argv), patch.object(train_4dgs, 'run') as run, \
                 patch.object(train_4dgs, 'PYTHON', Path(sys.executable)), \
                 patch.object(train_4dgs.subprocess, 'check_output', return_value='test-revision'):
                train_4dgs.main()
            command = list(map(str, run.call_args.args[0]))
            self.assertEqual(command[command.index('--test_iterations') + 1], '-1')
            saved = json.loads((model / 'run.json').read_text())
            self.assertEqual(saved['iterations'], 20000)
            with patch.object(sys, 'argv', ['render_4dgs.py', '--model', str(model)]), patch.object(render_4dgs, 'run') as render:
                render_4dgs.main()
            command = list(map(str, render.call_args.args[0]))
            self.assertIn('--skip_train', command)
            self.assertIn('--skip_test', command)
            self.assertEqual(command[command.index('--iteration') + 1], '20000')
            args = argparse.Namespace(model=model, data=data, config=model/'config.py', iteration=14000)
            self.assertEqual(model_options(args)[3], 14000)


if __name__ == '__main__':
    unittest.main()
