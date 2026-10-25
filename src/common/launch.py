"""Resolve framework interpreter and GPU before importing CUDA/JAX libraries."""
import os
import sys
from common.paths import ROOT, MODELS, framework_path, selected_gpu

def main():
    framework, module, *args = sys.argv[1:]
    gpu = None
    remaining = []
    index = 0
    while index < len(args):
        item=args[index]
        if item == '--gpu':
            index += 1
            if index >= len(args):
                raise SystemExit('--gpu requires a value')
            gpu=args[index]
        elif item.startswith('--gpu='):
            gpu=item.split('=',1)[1]
        else:
            remaining.append(item)
        index += 1
    if module in ('service.control','training.main'):
        remaining.extend(['--gpu',selected_gpu(gpu)])
    environment=os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT / 'src'), CUDA_VISIBLE_DEVICES=selected_gpu(gpu), HF_HOME=str(MODELS / 'base/huggingface'), OPENPI_DATA_HOME=str(MODELS / 'base/openpi'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    python=(ROOT if framework=='router' else framework_path(framework)) / '.venv/bin/python'
    os.chdir(ROOT)
    os.execve(str(python), [str(python), '-m',module,*remaining],environment)

if __name__ == '__main__':
    main()
