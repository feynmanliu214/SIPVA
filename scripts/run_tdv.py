#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: feynmanliu
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "core"))

from pipeline import execute_TDV_func


def main():
    ap = argparse.ArgumentParser(description="Run the TDV pipeline on one or more real KOIs.")
    ap.add_argument("kois", nargs="*", default=["841.02"],
                    help="KOI number(s) to fit (default: 841.02).")
    ap.add_argument("--detrend", choices=["gp", "savgol"], default=None,
                    help="Detrending method (default: TDV_DETREND env, else 'gp').")
    args = ap.parse_args()

    for koi in args.kois:
        print(execute_TDV_func(koi, detrend_method=args.detrend))


if __name__ == "__main__":
    main()
