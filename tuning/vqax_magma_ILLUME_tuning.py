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
import copy


try:
    from multimodal.model import get_multimodal_model
except Exception:
    from multimodal.model import get_multimodal_model
from PIL import Image
import json
import math
import argparse
from math import sqrt

torch.set_num_threads(32)

device_count = torch.cuda.device_count()
torch.set_num_threads(32*device_count)

parser = argparse.ArgumentParser(description='ILLUME Tuning')
parser.add_argument('--iteration', '-i', type=int)
parser.add_argument('--model_path', '-m', type=str, default=None, help='language model checkpoint from previous iteration')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--epochs', '-e', type=int, default=5,
                    help='training epochs')

parser = deepspeed.add_config_arguments(parser)
cmd_args = parser.parse_args()


config_path = "/workspace/MAGMA/configs/magma_rn50x16-mp1-config.yml"
ckpt_path = "/workspace/MAGMA/models/mp_rank_00_model_states-step30000.pt"
model_dir = "/workspace/MAGMA/models"

epochs = cmd_args.epochs
iteration = cmd_args.iteration

# target directory for model checkpoints
save_dir = f'/workspace/MAGMA/checkpoints/ILLUME_vqax/it{cmd_args.iteration}'

opt_config= {
        'lr': 1.12e-3 * sqrt(device_count/8),
        'betas': (0.9, 0.99),
        'wd': 0.2
    }
train_config = {
    'opt_config': opt_config,
    'weight_mult': 1.0,
    'device_batch_size': 32,
    'devices': device_count
}
os.makedirs(save_dir, exist_ok=True)
with open(os.path.join(save_dir, 'config.json'), "w+") as outfile:
    json.dump(train_config, outfile, indent=4)

deepspeed_config = {'train_batch_size': device_count * train_config['device_batch_size'],
                    'train_micro_batch_size_per_gpu': train_config['device_batch_size'],
                    'fp16': {'enabled': True}, 'fp32': {'enabled': False}}


class VQAXDataset(Dataset):

    def __init__(self, annotations_file, img_dir, img_dir_selftalk):
        df = pd.read_json(annotations_file)
        st_samples = len(df.loc[df.st])
        others = len(df) - st_samples
        self.df = df
        self.weight = others / st_samples
        self.img_dir = img_dir
        self.img_dir_selftalk = img_dir_selftalk
        self.dummy_input = torch.load(os.path.join(img_dir, 'dummy_input.pt'))[0, :500, :]
        self.dummy_label = torch.load(os.path.join(img_dir, 'dummy_label.pt'))[0, :500]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        img_id = item['question_id']
        sample_id = item['sample_id']
        f_img_id = f'{img_id:012d}'

        st_item = item['st']
        if st_item:
            input_embeddings_question = torch.load(
                os.path.join(self.img_dir_selftalk, f'{sample_id:012d}_question_input.pt'))[0, :, :]
            input_embeddings_explanation = torch.load(os.path.join(self.img_dir_selftalk, f'{sample_id:012d}_explanation_input.pt'))[0,
                                           :, :]

            labels_question = torch.load(os.path.join(self.img_dir_selftalk, f'{sample_id:012d}_question_label.pt'))[0, :]
            labels_explanation = torch.load(os.path.join(self.img_dir_selftalk, f'{sample_id:012d}_explanation_label.pt'))[0, :]
        else:

            input_embeddings_question = torch.load(
                os.path.join(self.img_dir, f'{f_img_id}_question_input.pt'))[0, :, :]

            input_embeddings_explanation = self.dummy_input
            labels_question = torch.load(os.path.join(self.img_dir, f'{f_img_id}_question_label.pt'))[0, :]

            labels_explanation = self.dummy_label

        return (input_embeddings_question, input_embeddings_explanation), (labels_question, labels_explanation, self.weight)




if __name__ == '__main__':

    # Load model
    model, _, _ = get_multimodal_model(config_path, ckpt_path=ckpt_path, model_dir=model_dir, lm_ckpt_path=cmd_args.model_path)

    # Discard anything but the LM to train
    lm = copy.deepcopy(model.lm)
    del model

    # Train only adapter weights
    lm.requires_grad = False
    for param in lm.parameters():
        param.requires_grad = False

    trainable_parameters = []

    for gptblock in lm.transformer.h:
        for param in gptblock.attn.adapter.parameters():
            param.requires_grad = True
            trainable_parameters.append(param)

        for param in gptblock.mlp[1].parameters():
            param.requires_grad = True
            trainable_parameters.append(param)

    lm.half()

    train_dataset = VQAXDataset(f'/workspace/repositories/ILLUME/results/vqax/it{iteration}/train_samples.json',
                              '/workspace/datasets/COCO/VQA-X/train_prep_seeing/',  '/workspace/datasets/COCO/VQA-X/train_ILLUME_tmp')

    print('model loaded')



    optimizer = torch.optim.AdamW(trainable_parameters,
        opt_config['lr'],
        betas=opt_config['betas'],
        weight_decay=opt_config['wd'])



    model_engine, opt, train_loader, lr_scheduler = deepspeed.initialize(
        model=lm,
        optimizer=optimizer,
        model_parameters=trainable_parameters,
        training_data=train_dataset,
        config = deepspeed_config
    )

    steps = math.ceil(len(train_loader) / train_config['device_batch_size']) * epochs
    pbar = tqdm(
        range(0, steps),
        desc="training...",
        total=steps,
    )

    rtpt = RTPT(name_initials='TBD', experiment_name=f'VQAX_Selftalk_it{iteration}',
                max_iterations=steps/5)

    rtpt.start()

    for epoch in range(epochs):
        train_loss = 0.00
        batch_loss = 0.00
        for idx, (inputs, captions) in enumerate(train_loader):

            # train loopc
            opt.zero_grad()
            inputs_question, inputs_explanation = inputs
            answer, explanation, weight = captions

            inputs_question = inputs_question.cuda().half()
            inputs_explanation = inputs_explanation.cuda().half()
            weight = weight[0].cuda()

            answer = answer.cuda()
            explanation = explanation.cuda()

            inputs_question.requires_grad = True
            inputs_explanation.requires_grad = True

            res_answer = model_engine(inputs_embeds=inputs_question, labels=answer)
            loss = res_answer.loss

            if inputs_explanation.shape[-1] > 1:
                res_explanation = model_engine(inputs_embeds=inputs_explanation, labels=explanation)
                loss += res_explanation.loss * train_config['weight_mult'] * weight

            model_engine.backward(loss)

            model_engine.step()

            train_loss += loss.item()
            batch_loss += loss.item()

            if idx > 0 and idx % 5 == 0:
                batch_loss /= 5
                pbar.update(5)
                rtpt.step()
                print(f'Batch loss:{batch_loss}')
                batch_loss = 0.0


        train_loss /= len(train_loader)
        print(f'Train loss: {train_loss:.4f}')

        sd = {"global_step": 'epoch_end', 'epoch': epoch}

        model_engine.save_checkpoint(save_dir, client_state=sd)

