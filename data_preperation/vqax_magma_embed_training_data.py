# Generate embeddings for training

import os
import torch
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.nn import DataParallel
import deepspeed
import argparse
import os
from rtpt.rtpt import RTPT


try:
    from multimodal.model import get_multimodal_model
except Exception:
    from multimodal.model import get_multimodal_model
from PIL import Image
import math
torch.set_num_threads(32)

config_path = "/workspace/MAGMA/configs/magma_rn50x16-mp1-config.yml"
ckpt_path = "/workspace/MAGMA/models/mp_rank_00_model_states-step30000.pt"
model_dir = "/workspace/MAGMA/models"


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--splits', '-sp', dest='splits', nargs='+', type=int, default=(0, 1),
                        help='Process split of the entire dataset')

    args = parser.parse_args()
    assert (len(args.splits) == 2)
    assert (args.splits[0] < args.splits[1])
    splits = args.splits

    device = 'cuda'


    print('Preprocessing training data')
    df_tmp = pd.read_json('/workspace/datasets/COCO/VQA-X/train.json')
    index = splits[0]
    max_split = splits[1]
    tmp_len = math.ceil(len(df_tmp) / max_split)
    df = df_tmp.loc[(df_tmp.index >= index * tmp_len) & (df_tmp.index < (index + 1) * tmp_len)]

    src_dir = '/workspace/datasets/COCO/train2014/train2014'
    dest_dir = '/workspace/datasets/COCO/VQA-X/train_prep_seeing'
    os.makedirs(dest_dir, exist_ok=True)
    print(f'Destination: {dest_dir}')
    img_prefix = 'COCO_train2014_'




    os.makedirs(dest_dir, exist_ok=True)
    rtpt = RTPT(name_initials='TBD', experiment_name=f'VQAX_PrepTrain_{splits[0]}_{splits[1]}', max_iterations=len(df))

    model, transforms, tokenizer = get_multimodal_model(config_path, ckpt_path=ckpt_path, model_dir=model_dir)
    model.half()
    model.eval()
    model.to(device)
    rtpt.start()

    for idx, item in tqdm(df.iterrows(), total=len(df)):
        img_id = item['image_id']
        question_id = item['question_id']

        img_path = os.path.join(src_dir, f'{img_prefix}{img_id:012d}.jpg')
        # MAGMA expects explicit "Q" and "A" for QA
        question = f'Q: {item["question"]} A:'

        answer = item["multiple_choice_answer"]
        explanation = item["explanation"]

        question_tokens = tokenizer.encode(
            question,
            return_tensors="pt",
        ).to(device)

        answer_tokens = tokenizer.encode(
            ' ' + answer,
            return_tensors="pt",
            max_length=2048,
            padding="max_length",
            truncation=True,
        ).to(device)


        explanation_prompt = f'{question} {answer}, seeing that'
        explanation_tokens = tokenizer.encode(
            explanation_prompt,
            return_tensors="pt",
        ).to(device)

        explanation_answer_tokens = tokenizer.encode(
            ' ' + explanation,
            return_tensors="pt",
            max_length=2048,
            padding="max_length",
            truncation=True,
        ).to(device)
        image = Image.open(img_path)
        trans = transforms(image)

        input_question = [trans, question_tokens]

        input_explanation = [trans, explanation_tokens]

        with torch.no_grad():
            emb_question = model.embed(input_question)

            emb_explanation = model.embed(input_explanation)

            labels_question = model.build_labels(emb_question, answer_tokens)

            labels_explanation = model.build_labels(emb_explanation, explanation_answer_tokens)

            word_embeddings_question = model.word_embedding(answer_tokens)

            word_embeddings_explanation = model.word_embedding(explanation_answer_tokens)

            input_embeddings_question = torch.cat(
                (
                    emb_question,
                    word_embeddings_question[:, : -emb_question.shape[1], :],
                ),  # remove padding in the word embedding before concatenating
                dim=1,
            )

            input_embeddings_explanation = torch.cat(
                    (
                        emb_explanation,
                        word_embeddings_explanation[:, : -emb_explanation.shape[1], :],
                    ),  # remove padding in the word embedding before concatenating
                    dim=1,
                )

            path_question_input = os.path.join(dest_dir, f'{question_id:012d}_question_input.pt')
            path_explanation_input = os.path.join(dest_dir, f'{question_id:012d}_explanation_input.pt')

            path_question_label = os.path.join(dest_dir, f'{question_id:012d}_question_label.pt')
            path_explanation_label = os.path.join(dest_dir, f'{question_id:012d}_explanation_label.pt')

            # Cut trailing padding for more efficient storage use
            torch.save(input_embeddings_question.cpu()[:, :500, :], path_question_input)

            torch.save(input_embeddings_explanation.cpu()[:, :500, :], path_explanation_input)

            torch.save(labels_question.cpu()[:, :500], path_question_label)

            torch.save(labels_explanation.cpu()[:, :500], path_explanation_label)

        rtpt.step()


