#!/usr/bin/env python
import argparse
import oyaml as yaml

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
    parser.add_argument(
        "--print-modules", default=False, action="store_true",
        help="Print available modules",
    )
    return parser.parse_args()

def main():
    options = parse_args()

    with open(options.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    if options.print_modules:
        print(",".join(cfg.keys()))
        return

    options.cfg = cfg
    del options.print_modules
    results = process_tables(**vars(options))

if __name__ == "__main__":
    main()
