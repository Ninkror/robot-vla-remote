"""Repeatable installation from a Git checkout; never starts a service."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from common.paths import ROOT, CONFIGS, framework_path

UV_VERSION = "0.13.3"

def run(*args, cwd=None):
    subprocess.run([str(value) for value in args], cwd=cwd, check=True)

def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()

def main(argv=None):
    parser = argparse.ArgumentParser(description="Install robot-vla from this Git checkout")
    parser.add_argument("--check", action="store_true", help="Check installation without installing or starting services")
    parser.add_argument("--skip-frameworks", action="store_true", help="Install only the router; framework environments must already exist")
    args = parser.parse_args(argv)
    if not (3,10) <= sys.version_info[:2] < (3,13):
        raise RuntimeError("Python 3.10..3.12 is required for the router")
    if not shutil.which("git"):
        raise RuntimeError("git is required")
    frameworks = json.loads((CONFIGS / "frameworks.json").read_text())
    for config_file in CONFIGS.glob("*.json"):
        if not isinstance(json.loads(config_file.read_text()), dict):
            raise RuntimeError(f"configuration must be an object: {config_file}")
    if args.check:
        report = {"root":str(ROOT), "router_python":(ROOT / '.venv/bin/python').is_file(), "frameworks":{}}
        for name, config in frameworks.items():
            path=framework_path(name)
            report['frameworks'][name] = {"path":str(path), "revision_ok":path.is_dir() and revision(path)==config['revision'], "python":(path / '.venv/bin/python').is_file()}
        print(json.dumps(report,indent=2))
        return 0 if report['router_python'] and all(v['revision_ok'] and v['python'] for v in report['frameworks'].values()) else 2
    for folder in ('data/source','data/datasets','data/temp','data/inference','models/base','models/trained','models/active','models/assets','logs','run'):
        (ROOT / folder).mkdir(parents=True,exist_ok=True)
    python = ROOT / '.venv/bin/python'
    if not python.exists():
        run(sys.executable,'-m','venv',ROOT / '.venv')
    run(python,'-m','pip','install','-r',ROOT / 'requirements-lock.txt')
    run(python,'-m','pip','install','--no-deps','--no-build-isolation','-e',ROOT)
    if not args.skip_frameworks:
        run(python,'-m','pip','install',f'uv=={UV_VERSION}')
        uv=ROOT / '.venv/bin/uv'
        for name, config in frameworks.items():
            path=framework_path(name)
            if not path.exists():
                run('git','clone',config['url'],path)
                run('git','-C',path,'checkout','--detach',config['revision'])
            if revision(path)!=config['revision']:
                raise RuntimeError(f'{name}: existing checkout revision differs; resolve before installing: {path}')
            env_python=path / '.venv/bin/python'
            if not env_python.exists():
                run(uv,'venv','--python',config['python'],path / '.venv')
            lock=ROOT / f'{name}-requirements-lock.txt'
            actual=subprocess.check_output([str(uv),'pip','freeze','--python',str(env_python)],text=True)
            normalized={line for line in actual.splitlines() if line and not line.startswith(('#','-e '))}
            expected={line for line in lock.read_text().splitlines() if line and not line.startswith('#')}
            if normalized != expected:
                run(uv,'pip','install','--python',env_python,'--no-deps','-r',lock)
            editable=[path] + ([path / 'packages/openpi-client'] if name=='openpi' else [])
            run(uv,'pip','install','--python',env_python,'--no-deps',*[item for p in editable for item in ('-e',p)])
            run(env_python,'-c',f'import {name}; print("{name} import OK")')
    print('Installed. Run ./scripts/doctor.sh; model weights and training archives are separate assets.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
