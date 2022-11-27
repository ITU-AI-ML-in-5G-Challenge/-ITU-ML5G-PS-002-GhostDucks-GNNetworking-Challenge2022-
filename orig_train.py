"""
training script based o quickstart notebook
trains the model
"""

import argparse
from pathlib import Path
import pandas as pd
from RouteNet_Fermi import main

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data-dir', required=True)
parser.add_argument('-o', '--outdir', required=True)

args = parser.parse_args()

outdir = Path(args.outdir)
if not outdir.exists():
    outdir.mkdir()

ckpt_dir = outdir / 'modelCheckpoints'

# train the model
main(train_path=args.data_dir, ckpt_dir=str(ckpt_dir))


# find best ckpt
def collect_experiment_result(path):
    path = Path(path)
    ret = pd.Series(name=path.name, dtype=float)
    for p in sorted(path.glob('modelCheckpoints/??-*.index')):
        fn = p.stem
        ep, loss = fn.split('-')
        ret.loc[int(ep)] = float(loss)

    return ret

ret = collect_experiment_result(outdir)
print('epoch validation results')
print(ret)

