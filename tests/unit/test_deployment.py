import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from common.paths import ROOT, selected_gpu, CONFIGS
from router.registry import ModelRegistry

class DeploymentTests(unittest.TestCase):
    def test_relative_checkpoint_and_runtime_gpu(self):
        registry=ModelRegistry(CONFIGS/'models.json',gpu='2')
        spec=registry.get('smolvla')
        self.assertEqual(spec.checkpoint,(ROOT/'models/active/smolvla').resolve())
        self.assertEqual(spec.cuda_visible_devices,'2')

    def test_gpu_does_not_accept_shell_arguments(self):
        for invalid in ('','-1','2;bad','0,,1'):
            with self.assertRaises(ValueError): selected_gpu(invalid)

    def test_gpu_launcher_sets_environment_before_framework_import(self):
        from common.launch import main
        with patch('sys.argv',['launch','openpi','pi05.training','train','--gpu','3']), patch('os.chdir'), patch('os.execve') as execute:
            main()
        python,args,environment=execute.call_args.args
        self.assertTrue(python.endswith('/openpi/.venv/bin/python'))
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'],'3')
        self.assertEqual(args[-1],'train')
