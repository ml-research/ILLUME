import os
import glob
import time
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--num_generations', '-ng', dest='n_xgen', type=int, default=2,
                    help='# of generation for explanations')
parser.add_argument('--save_dir', '-s', dest='save_path', type=str, required=True,
                    help='where to save results')
parser.add_argument('--tempsX', '-tx', metavar='N', type=float, nargs='+',
                    help='list of temperature values')
parser.add_argument('--datasplit', '-d', dest='datasplit', choices=['train', 'val', 'test'],
                    type=str, default='val', help='dataset split')
parser.add_argument('--ground_truth', '-gt', dest='gt_condition', action='store_true', default=False, help='Condition explanations on ground truth answer')

args = parser.parse_args()


def chunks(arr, n):
    return [arr[i:i+n] for i in range(0, len(arr), n)]


csv_files = glob.glob(os.path.join(args.save_path, "*.csv"))
dfs = [pd.read_csv(csv) for csv in csv_files]
df = pd.concat(dfs)

gen_explanations = [col for col in df if col.startswith("gen_explanation")]
others = [col for col in df if not col.startswith("gen_explanation")]
assert len(gen_explanations) == args.n_xgen*len(args.tempsX)
gen_explanations_chunks = chunks(gen_explanations, args.n_xgen)
col_names = others + ["gen_explanation"+str(i) for i in range(args.n_xgen)]

for t, gen_explanations_chunk in zip(args.tempsX, gen_explanations_chunks):
    save_path = os.path.join(args.save_path, f'{t}_{str(time.time())}{"_gt" if args.gt_condition else ""}'+ '.csv')
    df_chunk = df[others + gen_explanations_chunk]
    df_chunk.columns = col_names
    print("Save reformated csv file to", save_path)
    df_chunk.to_csv(save_path)