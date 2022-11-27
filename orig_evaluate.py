"""
evaluation script based o quickstart notebook
runs evaluation using the original challenge repo
"""

import argparse
from RouteNet_Fermi import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('-ckpt', '--ckpt', required=True)

args = parser.parse_args()

evaluate(args.ckpt)
