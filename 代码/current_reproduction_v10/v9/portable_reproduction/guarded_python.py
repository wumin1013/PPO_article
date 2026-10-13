"""Run one registered regeneration command with original-directory reads denied."""
import argparse
import os
from pathlib import Path
import runpy
import sys
from archive_tools import write_new
from portable_paths import install_guard,resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--guard-receipt',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:
        raise ValueError('A Python script and its arguments are required.')
    workspace = args.workspace.resolve()
    script = resolve(command[0],workspace,anchor='base')
    os.environ.update(JNN_PORTABLE_ROOT=str(workspace),JNN_V9_BASE=str(workspace),JNN_V9_ROOT=str(workspace/'v9'))
    state = install_guard(workspace)
    code = 0
    try:
        sys.path.insert(0,str(script.parent))
        sys.argv = [str(script),*command[1:]]
        runpy.run_path(str(script),run_name='__main__')
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code,int) else 1
        if code:
            raise
    except BaseException:
        code = 1
        raise
    finally:
        write_new(args.guard_receipt,dict(**state,script=str(script),exit_code=code,
            dependency_scope='Python environment and installed fonts may be read; original research roots are denied.'))


if __name__=='__main__':
    main()
