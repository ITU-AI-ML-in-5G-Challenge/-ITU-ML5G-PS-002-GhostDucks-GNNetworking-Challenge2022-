import argparse
import os.path
from pathlib import Path
from evaluate import my_evaluate
from extract_features import extract_features
from make_embeddings import embed_dir
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-ckpt', '--checkpoint-path', help='path to checkpoint (without .index suffix)', required=True)
parser.add_argument('-data', '--data-path', help='path to data directory (pkl)', required=True)
parser.add_argument('-name', '--name', help='name prefix for the output files', required=True)
parser.add_argument('-o', '--output-dir', help='path to output directory', required=True)
args = parser.parse_args()

ckpt_path = args.checkpoint_path
if ckpt_path.endswith('.index'):
    ckpt_path = ckpt_path[:-len('.index')]

if not os.path.isfile(ckpt_path + '.index'):
    raise RuntimeError(f'could not find checkpoint (or is not a file): {ckpt_path}')

# extract sample losses
# this should create 'eval' directory at the checkpoint parent directory
# with sample loss csv file and evlauation result csv file
print('running evaluate on ', ckpt_path)
my_evaluate([ckpt_path], args.data_path, args.name, n_workers=1, n_gpus=0)
ckpt_path = Path(ckpt_path)
src_eval = ckpt_path.parent.parent / 'eval'
tgt_eval = Path(args.output_dir) / 'eval'
if not tgt_eval.exists():
    tgt_eval.mkdir(parents=True)

print(f'copying eval to {tgt_eval}')
for p in src_eval.glob('*.*'):
    shutil.copy(p, tgt_eval)

# feature extraction
# this creates a directory tree identical to the data dir with pickle files
# which hold the routenet flow/queue/link internal features of the corresponding samples
emb_dir = Path(args.output_dir) / f'sample_embeddings_{ckpt_path.name}'
extract_features(args.checkpoint_path, args.data_path, emb_dir / args.name)

# creating embeddings
# this creates a pkl file containing embedding vector for every sample, by aggregating features from previous step
emb_file = emb_dir / f'{args.name}_min_max_mean.pkl'
samples_dir = emb_dir / args.name
embed_dir(samples_dir, emb_file, path_aggregate='mean')
