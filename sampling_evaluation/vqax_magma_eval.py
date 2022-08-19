# Generate csv file with answer and explanation for VQA-X validation data
# Manipulate finetune_path and save_path as required
# Assumes that fine-tuning was performed with Prompt "Explanation:"
import os

import pandas as pd
import torch
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from rtpt.rtpt import RTPT

try:
    from multimodal.model import get_multimodal_model
except Exception:
    from multimodal.model import get_multimodal_model

from PIL import Image
from tqdm import tqdm
import time
import math

torch.set_num_threads(16)

config_path = "/workspace/MAGMA/configs/magma_rn50x16-mp1-config.yml"
ckpt_path = "/workspace/MAGMA/models/mp_rank_00_model_states-step30000.pt"
model_dir = "/workspace/MAGMA/models"
seed = 42
device = 'cuda'

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--num_generations', '-ng', dest='n_xgen', type=int, default=2,
                    help='# of generation for explanations')
parser.add_argument('--modal_path', '-m', dest='finetune_path', type=str, default=None,
                    help='language model checkpoint')
parser.add_argument('--save_dir', '-s', dest='save_path', type=str, required=True,
                    help='where to save results')
parser.add_argument('--tempsX', '-tx', metavar='N', type=float, nargs='+',
                    help='list of temperature values')
parser.add_argument('--datasplit', '-d', dest='datasplit', choices=['train', 'val', 'test'],
                    type=str, default='val', help='dataset split')
parser.add_argument('--ground_truth', '-gt', dest='gt_condition', action='store_true', default=False, help='Condition explanations on ground truth answer')
parser.add_argument('--splits', '-sp', dest='splits', nargs='+', type=int, default=(0,1), help='Process split of the entire dataset')

args = parser.parse_args()
assert(len(args.splits) == 2)
assert(args.splits[0] < args.splits[1])

splits = args.splits

t = torch.tensor(1).to(device)
rtpt_tmp = RTPT(name_initials='MB', experiment_name='Loading_model', max_iterations=1)
rtpt_tmp.start()

# setting input args
finetune_path = args.finetune_path
n_xgen = args.n_xgen

save_path = os.path.join(args.save_path, f'{splits[0]}_{splits[1]}_{str(time.time())}{"_gt" if args.gt_condition else ""}'+ '.csv')
tempsX = args.tempsX

# create save file directory
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# how to prompt explanation generation. can be extended as needed
expl_prompts = [', seeing that']


class VQAXDataset(Dataset):
    def __init__(self, annotations_file, img_dir, splits):
        df = pd.read_json(annotations_file)

        index = splits[0]
        max_split = splits[1]
        tmp_len = math.ceil(len(df) / max_split)
        df_tmp = df.loc[(df.index >= index * tmp_len) & (df.index < (index + 1) * tmp_len)]
        self.prefix = 'train' if args.datasplit == 'train' else 'val'
        self.df = df_tmp
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        with torch.no_grad():
            item = self.df.iloc[idx]
            img_id = item['image_id']

            img_path = os.path.join(self.img_dir, f'COCO_{self.prefix}2014_{img_id:012d}.jpg')
            # MAGMA expects QA with explicit "Q" and "A" in prompt
            question = f'Q: {item["question"]} A:'

            answer = item['multiple_choice_answer']

            image = Image.open(img_path)

            input_ = transforms(image)



            return (input_, question), (answer, idx, img_id)


if __name__ == '__main__':

    t = torch.tensor(1).to(device)
    rtpt_tmp = RTPT(name_initials='TBD', experiment_name='Loading_model', max_iterations=1)
    rtpt_tmp.start()

    # load model
    model, transforms, tokenizer = get_multimodal_model(config_path, ckpt_path=ckpt_path, model_dir=model_dir,
                                                        lm_ckpt_path=finetune_path)
    model.eval()
    model.half()
    model.to(device)
    print('model loaded')

    # setup data
    if args.datasplit == 'train':
        dataset = VQAXDataset('/workspace/datasets/COCO/VQA-X/train.json',
                              '/workspace/datasets/COCO/train2014/train2014/', splits)
    elif args.datasplit == 'val':
        dataset = VQAXDataset('/workspace/datasets/COCO/VQA-X/val.json',
                              '/workspace/datasets/COCO/val2014/val2014/', splits)
    elif args.datasplit == 'test':
        dataset = VQAXDataset('/workspace/datasets/COCO/VQA-X/test.json',
                              '/workspace/datasets/COCO/val2014/val2014/', splits)

    # Load single item batch
    val_loader = DataLoader(dataset, shuffle=False, batch_size=1)
    df_res = pd.DataFrame(columns=['idx', 'image_id', 'gen_answer', 'answer'])

    # start evaluation
    rtpt = RTPT(name_initials='MB', experiment_name=f'VQAX_validation_eval_{tempsX[0]}_{splits[0]}_{splits[1]}',
                max_iterations=len(val_loader) / 20)
    rtpt.start()

    for (data, meta) in tqdm(val_loader):
        image, question = data
        answer, idx, img_id = meta
        tokens = tokenizer(question[0], return_tensors='pt')['input_ids'].to(device)
        input_list = [image[0], tokens]

        with torch.no_grad():
            # generate answer
            emb = model.embed(input_list)
            torch.manual_seed(seed)
            answer_out = model.generate(emb, temperature=0.01)[0]

            # generate several explanations
            explanations = list()
            for t in tempsX:
                for expl_prompt in expl_prompts:
                    torch.manual_seed(seed)
                    # Explanation prompt
                    if args.gt_condition:
                        prompt = f'{question[0].strip()} {answer[0].strip()}{expl_prompt}'
                    else:
                        prompt = f'{question[0].strip()} {answer_out.strip()}{expl_prompt}'
                    tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
                    input_list = [image[0], tokens]
                    emb = model.embed(input_list)

                    # Generate n outputs for each temperature and explanation prompt
                    explanations.extend(model.generate(emb.repeat(n_xgen, 1, 1), temperature=t))

        # accumulate results
        res_dict = {'idx': int(idx), 'image_id': int(img_id), 'gen_answer': answer_out, 'answer': answer}
        for i, explanation in enumerate(explanations):
            res_dict[f'gen_explanation{i}'] = explanation
        df_res = df_res.append(res_dict, ignore_index=True)
        if idx % 20 == 0:
            rtpt.step()

    # save results
    print('Finished')
    print('saving results to', save_path)
    df_res.to_csv(save_path)
