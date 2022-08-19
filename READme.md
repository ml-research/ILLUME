# ILLUME: Rationalizing Vision-Language Models by Interacting with their Jabber

Official implementation of the ILLUME approach for interactive fine-tuning of VLM. Includes data and examples to 
run the ILLUME pipeline for VQA-X as presented in the paper.

## Abstract 

Bootstrapping from pre-trained language models has been proven to be an efficient approach for building foundation vision-language models (VLM) for tasks such as image captioning or visual question answering. However, it is difficult-if not impossible-to utilize it to make the model conform with user's rationales for specific answers. To elicit and reinforce commonsense reasons, we propose an iterative sampling and tuning paradigm, called ILLUME, that executes the following loop: Given an image-question-answer prompt, the VLM samples multiple candidate rationales, and a human critic provides minimal feedback via preference selection, used for fine-tuning. This loop increases the training data and gradually carves out the VLM's rationalization capabilities. Our exhaustive experiments demonstrate that ILLUME is competitive with standard supervised fine-tuning while using significantly fewer training data and only requiring minimal feedback.

## Computing requirements

Evaluating and sampling takes roughly 15 GB on a A100 GPU. 
Fine-tuning the model with deepspeed requires at least 45GB of VRAM with the batch size chosen in the 
paper (256) takes up over 75GB on 8 A100 GPUs each. 

## Prerequisites

The provided code requires downloading the following resources before running it.

### MAGMA checkpoint 
You can download MAGMA weights and config file from the official GitHub repository at https://github.com/Aleph-Alpha/magma.
Note or directly from https://bit.ly/aleph_alpha_magma_download. Please note that this checkpoint may
slightly differ from the one used in the paper. However, the performance is expected to be similar. 

Please consult the ```Docker container``` section on how to properly link your checkpoint and configuration.

### COCO images


Please consult the ```Docker container``` section on how to properly link your image directory.

## Docker Container 

## ILLUME Pipeline 