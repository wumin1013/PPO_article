"""Resolve archived provenance to extracted files; deny original research reads."""
import os
from pathlib import Path, PureWindowsPath
import sys

ORIGINAL_BASE = 'D:/PPO_article_work/JNN_dominance_20261011_v5'
ORIGINAL_V9 = 'D:/PPO_article_work/JNN_efficiency_20261012_v9'
DENIED_ORIGINAL_ROOTS = ('D:/PPO_article_work','D:/PPO_article_joint_runs',
    'D:/Nutstore/1/\u57fa\u4e8e\u5f3a\u5316\u5b66\u4e60\u7684\u8fd0\u52a8\u89c4\u5212\u65b9\u6cd5',
    'D:/PPO_repro_scaled_v8_final1','D:/PPO_repro_boundary_v9','D:/PPO_v10_payload')


def normalized(value):
    return os.fsdecode(value).replace('\\','/').rstrip('/')


def beneath(value,root):
    value,root = normalized(value).casefold(),normalized(root).casefold()
    return value == root or value.startswith(root+'/')


def resolve(value,workspace,*,anchor='v9',must_exist=True):
    workspace = Path(workspace).resolve()
    text = normalized(value)
    if beneath(text,ORIGINAL_V9):
        target = workspace/'v9'/text[len(ORIGINAL_V9):].lstrip('/')
    elif beneath(text,ORIGINAL_BASE):
        target = workspace/text[len(ORIGINAL_BASE):].lstrip('/')
    elif Path(text).is_absolute() or PureWindowsPath(text).is_absolute():
        target = Path(text).resolve()
        if not target.is_relative_to(workspace):
            raise ValueError('Unmapped external research path: '+text)
    else:
        if anchor not in ('base','v9'):
            raise ValueError('Unknown manifest anchor: '+anchor)
        target = workspace/('v9' if anchor=='v9' else '')/text
    target = target.resolve()
    if not target.is_relative_to(workspace):
        raise ValueError('Resolved path escapes extracted workspace: '+text)
    if must_exist and not target.exists():
        raise FileNotFoundError('Required portable member is absent: '+str(target))
    return target


def install_guard(workspace):
    workspace = Path(workspace).resolve()
    if any(beneath(str(workspace),root) for root in DENIED_ORIGINAL_ROOTS):
        raise ValueError('Independent verification output must be outside original research roots.')
    state = {'guard_active':True,'research_reads_inside_workspace':0,'blocked_original_accesses':[]}
    def guard(event,args):
        if event not in ('open','os.listdir','os.scandir') or not args:
            return
        value = args[0]
        if value is None or isinstance(value,int):
            return
        text = normalized(value)
        if not (Path(text).is_absolute() or PureWindowsPath(text).is_absolute()):
            text = str((Path.cwd()/text).resolve())
        if beneath(text,str(workspace)):
            state['research_reads_inside_workspace'] += 1
            return
        if any(beneath(text,root) for root in DENIED_ORIGINAL_ROOTS):
            state['blocked_original_accesses'].append(dict(event=event,path=text))
            raise PermissionError('Portable verification forbids original research access: '+text)
    sys.addaudithook(guard)
    return state
