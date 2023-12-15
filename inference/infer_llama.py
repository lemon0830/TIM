#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from urllib.request import urlopen
import argparse
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, GenerationConfig
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters

def post_processing(text):
    replace_words = ["#PR_SORG#", "#PRS=ORG#", "#PRS_ORG#_ORG#", "#PRS_MUSIC#", "#PURO_BOX#", "#PRS_SIGNUP#", "#prs_org#",
                     "#PERS_ACC#", "#PRS-ORG#", "##PRS_ORG##", "#PRI_ORG", "PRS _ORG#", "PRS.ORG#", "#PRS _ ORG#", "#PROS#",
                     "#PRS_SIG#", "#PR_SYS#", "#PERSO#", "#PrsOrg#", "ＰＲＳＯＧＥ", "PRS ORG", "#PR_SIG#", "#PRS__ORG#",
                     "#PRS_Org#", "PRS #ORG#"]

    for word in replace_words:
        if word in text:
            text = text.replace(word, "#PRS_ORG#")

    if "#PRS" in text:
        text = text.replace("#PRS", "#PRS_ORG#").replace("#PRS_ORG#_ORG#", "#PRS_ORG#")

    return text

def func(text):

    try:

        text = "Write a response that appropriately completes the request.\n\n" + \
               f"### Request:\n{text}\n\n### Response:"

        hyp = pipeline(text,
                       temperature=0.1,
                       top_p=0.9,
                       do_sample=False,
                       num_beams=4,
                       max_new_tokens=256,
                       no_repeat_ngram_size=15,
                       pad_token_id=tokenizer.pad_token_id,
                       eos_token_id=tokenizer.eos_token_id)[0]["generated_text"]
    except:
        text = f"### Request:\n{text}\n\n### Response:"

        hyp = pipeline(text,
                       do_sample=False,
                       num_beams=4,
                       max_new_tokens=256,
                       no_repeat_ngram_size=8,
                       pad_token_id=tokenizer.pad_token_id,
                       eos_token_id=tokenizer.eos_token_id)[0]["generated_text"]

    hyp = "".join(hyp.split("\n"))

    # print(hyp)
    # exit()
    
    return hyp

def init_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=".",  type=str)
    parser.add_argument("-i", "--input_file", help="name of the input file")
    parser.add_argument("-o", "--output_file", help="name of the output file", type=str)
    parser.add_argument("-d", "--draft_file", default=None, help="name of the output file", type=str)

    parser.add_argument("-l","--iflora", default=False, action='store_true')
    parser.add_argument("--src", default="en", type=str)
    parser.add_argument("--tgt", default="zh", type=str)
    parser.add_argument("--rootmodel", default=".",  type=str)
    parser.add_argument("--ifhint", default=False, action='store_true')
    parser.add_argument("--iflang", default=False, action='store_true')
    parser.add_argument("--ifnosrc", default=False, action='store_true')


    return parser

LAN_dict = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "fr": "French",
    "cs": "Czech",
    "ko": "Korean",
    "ja": "Japanese",
    "ru": "Russian",
    "uk": "Ukrainian"
}

if __name__ == '__main__':
    arg_parser = init_opt()

    args = arg_parser.parse_args()

    modelpath = args.model_path
    rootmodel = args.rootmodel
    infile = args.input_file
    outfile = args.output_file
    ifLoRA = args.iflora
    src = LAN_dict[args.src]
    tgt = LAN_dict[args.tgt]
    ifhint = args.ifhint
    iflang = args.iflang
    ifnosrc = args.ifnosrc

    tokenizer = AutoTokenizer.from_pretrained(modelpath, cache_dir=modelpath, use_fast=False)

    if tokenizer.pad_token is None:
        print("no")
        tokenizer.add_special_tokens(dict(pad_token="[PAD]"))
        # exit()

    if ifLoRA:
        import torch

        lora_state_dict_path = modelpath + "/lora_pytorch_model.bin"

        model = AutoModelForCausalLM.from_pretrained(rootmodel, cache_dir=modelpath)

        # for name, module in model.named_modules():
        #     print(name)

        lora_module_name = "q_proj,k_proj,v_proj,o_proj".split(",")
        lora_dim = 8
        lora_alpha = 16
        lora_droppout = 0.00

        model = convert_linear_layer_to_lora(model, lora_module_name=lora_module_name, lora_dim=lora_dim,
                                             lora_alpha=lora_alpha, lora_droppout=lora_droppout).half().cuda()

        state_dict = torch.load(lora_state_dict_path)

        new_state_dict = {}
        for k in state_dict:
            newk = "model." + k
            # print(newk)
            new_state_dict[newk] = state_dict[k]

        model.load_state_dict(new_state_dict, strict=False)

        model = convert_lora_to_linear_layer(model)
        model.eval()

    else:
        model = AutoModelForCausalLM.from_pretrained(modelpath, cache_dir=modelpath).half().cuda()
        # model = AutoModelForCausalLM.from_pretrained(modelpath, torch_dtype=torch.float16, cache_dir=modelpath, device_map="auto", load_in_8bit=True)

    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer,
                                      return_full_text=False, device=model.device,
                                      clean_up_tokenization_spaces=True,
                                      handle_long_generation="hole")

    inf = open(infile, "r")
    outf = open(outfile, "w")

    if args.draft_file:
        draftf = open(args.draft_file, "r")

    NUM=0

    for line in inf:
        line = line.strip()

        if ifnosrc:
            line = f"Please provide the {tgt} translation.\n{line}"
        else:
            line = f"Translate from {src} to {tgt}.\n{line}"

        if ifhint:
            line = line + "\n\n###Note: A translation with no errors could be"

        if iflang:
            line = line + f"\n\n###Output_Language: {tgt}"

        trans = func(line)
        outf.write(trans+"\n")

        NUM += 1
        if NUM % 50 == 0:
            print(f"###Processing {NUM} examples.")
            outf.flush()
            sys.stdout.flush()

    inf.close()
    outf.close()




