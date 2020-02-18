#!/usr/bin/env python
import argparse
import os
import yaml
import importlib

from zquery import process_tables

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, help="Config file to use")
    parser.add_argument("modules", type=str, help="Modules to run")
    parser.add_argument("paths", type=str, help="Paths to run over")
    parser.add_argument(
        "--njobs", "-n", type=int, default=-1,
        help="Number of jobs to create",
    )
    parser.add_argument(
        "--pysge_func", type=str, default="local_submit",
        help="pysge submission function",
    )
    parser.add_argument(
        "--pysge_args", type=str, default="",
        help="pysge submission arguments. Comma-delimited.",
    )
    parser.add_argument(
        "--pysge_kwargs", type=str, default="",
        help="pysge submission kwargs. Comma-delimited, colon-separated.",
    )
    parser.add_argument(
        "--verbose", "-v", default=False, action="store_true",
        help="Verbose print out whilst running",
    )
    return parser.parse_args()

def read_yaml(path, delimiter=','):
    if os.path.exists(path):
        with open(path, 'r') as f:
            info = yaml.safe_load(f)
    else:
        info = path

    if isinstance(info, str):
        info = info.split(delimiter)

    return info

def main():
    options = parse_args()

    options.cfg = read_yaml(options.cfg)
    options.modules = read_yaml(options.modules)
    options.paths = read_yaml(options.paths)

    for key, process in options.cfg.items():
        module_name, func_name = process["func"].split(":")
        imported_module = importlib.import_module(module_name)
        func = getattr(imported_module, func_name)
        process["func"] = func

    options.pysge_args = options.pysge_args.split(",")
    kwargs = {}
    if len(options.pysge_kwargs) > 0:
        kwargs.update(dict(
            tuple(kw.split(":"))
            for kw in options.pysge_kwargs.split(",")
        ))
    options.pysge_kwargs = kwargs

    results = process_tables(**vars(options))

if __name__ == "__main__":
    main()
