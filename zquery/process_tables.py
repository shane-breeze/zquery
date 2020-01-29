import os
import glob
import numpy as np
import importlib
import pysge
import tqdm.auto as tqdm

__all__ = ["process_tables"]

def get_list_of_files(paths, njobs=-1):
    full_paths = []
    for p in paths:
        full_paths.extend(sorted(list(glob.glob(p))))
    full_paths = [os.path.abspath(p) for p in full_paths]
    if njobs > 0:
        full_paths = np.array_split(full_paths, njobs)
    else:
        full_paths = [[p] for p in full_paths]
    return full_paths

def task(cfg, modules, paths, verbose=False):
    results = []
    for path in tqdm.tqdm(paths, unit='path'):
        if verbose:
            print(path)

        modules_pbar = tqdm.tqdm(modules, unit='module')
        for module in modules_pbar:
            modules_pbar.set_description(module)
            if verbose:
                print(module)
            tcfg = cfg[module]
            imported_module_name, func_name = tcfg["func"].split(":")
            imported_module = importlib.import_module(imported_module_name)
            func = getattr(imported_module, func_name)
            results.append(func(
                path, *tcfg.get("args", []), **tcfg.get("kwargs", {}),
            ))
    return results

def create_tasks(cfg, modules, paths, verbose=False):
    return [{
        "task": task,
        "args": (cfg, modules, subpaths),
        "kwargs": {"verbose": verbose},
    } for subpaths in paths]

def process_tables(
    cfg, modules, paths, njobs=-1, pysge_func="local_submit", pysge_args=tuple(),
    pysge_kwargs={}, verbose=False,
):
    paths = get_list_of_files(paths, njobs)
    tasks = create_tasks(cfg, modules, paths, verbose=verbose)

    args = [tasks] + list(pysge_args)
    results = getattr(pysge, pysge_func)(*args, **pysge_kwargs)
    return results
