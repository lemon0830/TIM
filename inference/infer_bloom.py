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

from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.reward_model import RewardModel

def post_processing(text):
    text = text.replace("<v>", "").replace("</v>", "")

    return text

def func(text, ifsample=False):

    text = "Write a response that appropriately completes the request.\n\n" + \
            f"### Request:\n{text}\n\n### Response:"

    if ifsample:
        hyp = pipeline(text,
                       temperature=0.1,
                       top_p=0.7,
                       do_sample=True,
                       max_new_tokens=256,
                       no_repeat_ngram_size=15,
                       pad_token_id=tokenizer.pad_token_id,
                       eos_token_id=tokenizer.eos_token_id,
                       num_return_sequences=4) #[0]["generated_text"]
    else:
        hyp = pipeline(text,
                       temperature=0.1,
                       top_p=0.9,
                       do_sample=False,
                       num_beams=4,
                       max_new_tokens=256,
                       no_repeat_ngram_size=15,
                       pad_token_id=tokenizer.pad_token_id,
                       eos_token_id=tokenizer.eos_token_id,
                       num_return_sequences=4) #[0]["generated_text"]


    hyps = []
    for i in hyp:
        text = i["generated_text"]
        text = "".join(text.split("\n"))
        hyps.append(post_processing(text))

    return hyps

def init_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=".",  type=str)
    parser.add_argument("-i", "--input_file", help="name of the input file")
    parser.add_argument("-o", "--output_file", help="name of the output file", type=str)
    parser.add_argument("-d", "--draft_file", default=None, help="name of the output file", type=str)

    parser.add_argument("-l","--iflora", default=False, action='store_true')
    parser.add_argument("--src", default="en", type=str)
    parser.add_argument("--tgt", default="zh", type=str)
    parser.add_argument("--ifhint", default=False, action='store_true')
    parser.add_argument("--rootmodel", default=".",  type=str)
    parser.add_argument("--vocab", default=None,  type=str)
    parser.add_argument("--reverse", default=False, action='store_true')
    parser.add_argument("--ifreranking", default=False, action='store_true')
    parser.add_argument("--ifsample", default=False, action='store_true')

    return parser

LAN_dict = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "fr": "French",
    "cs": "Czech",
    "ja": "Japanese",
    "ru": "Russian",
    "uk": "Ukrainian"
}

def get_vocab(file_name, reverse=False):
    zh2en_vocab = {}

    if os.path.exists(file_name):
        f = open(file_name)
        for line in f:
            en, zh = line.strip().split(" ")

            if reverse:
                en, zh = zh, en

            zh2en_vocab[en+"\n"+zh] = 1

        return zh2en_vocab

    return []

prompt_dict = [
        "Translate the following sentences from [SRC] to [TGT].",
        "What do the following sentences mean in [TGT]?",
        "Please provide the [TGT] translation for the following sentences.",
        "Convert the subsequent sentences from [SRC] into [TGT].",
        "Render the listed sentences in [TGT] from their original [SRC] form.",
        "Transform the upcoming sentences from [SRC] language to [TGT] language.",
        "Change the given sentences from [SRC] to [TGT] format.",
        "Turn the following sentences from their [SRC] version to the [TGT] version.",
        "Adapt the mentioned sentences from [SRC] to the [TGT] language.",
        "Transpose the next sentences from the [SRC] format to the [TGT] format.",
        "Switch the specified sentences from their [SRC] form to [TGT] form.",
        "Reinterpret the ensuing sentences from [SRC] to [TGT] language.",
        "Modify the forthcoming sentences, converting them from [SRC] to [TGT].",
        "How can the subsequent sentences be interpreted in [TGT]?",
        "What is the meaning of these sentences when translated to [TGT]?",
        "In the context of [TGT], what do the upcoming sentences signify?",
        "How would you express the meaning of the following sentences in [TGT]?",
        "What is the significance of the mentioned sentences in [TGT]?",
        "In [TGT], what do the given sentences convey?",
        "When translated to [TGT], what message do these sentences carry?",
        "What is the intended meaning of the ensuing sentences in [TGT]?",
        "How should the following sentences be comprehended in [TGT]?",
        "In terms of [TGT], what do the next sentences imply?",
        "Kindly furnish the [TGT] translation of the subsequent sentences.",
        "Could you supply the [TGT] translation for the upcoming sentences?",
        "Please offer the [TGT] rendition for the following statements.",
        "I'd appreciate it if you could present the [TGT] translation for these sentences.",
        "Can you deliver the [TGT] translation for the mentioned sentences?",
        "Please share the [TGT] version of the given sentences.",
        "It would be helpful if you could provide the [TGT] translation of the ensuing sentences.",
        "Kindly submit the [TGT] interpretation for the next sentences.",
        "Please make available the [TGT] translation for the listed sentences.",
        "Can you reveal the [TGT] translation of the forthcoming sentences?"
]

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
    reverse = args.reverse
    ifreranking = args.ifreranking
    ifsample = args.ifsample

    if args.vocab is not None:
        vocab = get_vocab(args.vocab, reverse=reverse)
        print(f"Loading vocab {len(vocab)}")

    tokenizer = AutoTokenizer.from_pretrained(modelpath, cache_dir=modelpath)
    # tokenizer.padding_side = "left"

    if ifLoRA:
        import torch

        lora_state_dict_path = modelpath + "/lora_pytorch_model.bin"

        model = AutoModelForCausalLM.from_pretrained(rootmodel, cache_dir=rootmodel, low_cpu_mem_usage=True).half()
        # model = AutoModel.from_pretrained(modelpath, cache_dir=modelpath)

        lora_module_name = "query_key_value".split(",")
        lora_dim = 8
        lora_alpha = 16
        lora_droppout = 0.05

        model = convert_linear_layer_to_lora(model, lora_module_name=lora_module_name, lora_dim=lora_dim,
                                             lora_alpha=lora_alpha, lora_droppout=lora_droppout).cuda()

        state_dict = torch.load(lora_state_dict_path)

        new_state_dict = {}
        for k in state_dict:
            # newk = k.replace("model.", "transformer.")
            newk = "transformer." + k
            print(newk)
            new_state_dict[newk] = state_dict[k]

        model.load_state_dict(new_state_dict, strict=False)

        model = convert_lora_to_linear_layer(model)
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(modelpath, cache_dir=modelpath,
                                                     low_cpu_mem_usage=True).half().cuda()

    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer,
                                      return_full_text=False, device=model.device,
                                      clean_up_tokenization_spaces=True,
                                      handle_long_generation="hole")

    if ifreranking:
        rwmodel = RewardModel(model, tokenizer, rl_alpha=0.0)
        rw_state_dict_path = modelpath + "/rw_pytorch_model.bin"
        rw_state_dict = torch.load(rw_state_dict_path)

        print(rw_state_dict.keys())

        rwmodel.load_state_dict(rw_state_dict, strict=False)
        rwmodel = rwmodel.half().cuda()


    inf = open(infile, "r")
    outf = open(outfile, "w")

    if args.draft_file:
        draftf = open(args.draft_file, "r")

    NUM=0

    for line in inf:
        line = line.strip()

        # line = f"Translate to {tgt}.\n{line}"
        if args.vocab is not None:
            aligns = {}
            for align in vocab:
                ws, wt = align.split("\n")
                if ws in line:
                    aligns[ws] = wt

            Hint_prompt = ""
            if len(aligns) > 0:
                Hint_prompt = "\n\n###Hint:"
                for ws in aligns:
                    Hint_prompt += " " + ws + " means " + aligns[ws] + "."

                print(line + Hint_prompt)

            line = line + Hint_prompt

        line = f"Translate from {src} to {tgt}.\n{line}"

        if ifhint:
            line = line + "\n\n###Hint: A translation with no errors could be"

        trans = func(line, ifsample=args.ifsample)

        if ifreranking:
            sources_token = tokenizer(line, padding=True, return_tensors="pt")
            hyp = tokenizer(trans, padding=True, return_tensors="pt")

            token_s = sources_token["input_ids"]
            token_s = torch.cat([token_s for _ in range(len(trans))], dim=0)
            token_hyp = hyp["input_ids"]

            cur_input_ids = torch.cat([token_s, token_hyp], dim=1).cuda()
            Len_s = len(token_s[0])
            attn_mask = torch.ones_like(cur_input_ids).cuda()

            with torch.no_grad():
                rw_out = rwmodel.forward_value(input_ids=cur_input_ids, attention_mask=attn_mask, prompt_length=Len_s)

            # score = rw_out["chosen_end_scores"]
            score = rw_out["chosen_end_scores"]
            max_idx = torch.argmax(score).item()

            # print(trans, score, max_idx)
            # exit()
            out_trans = trans[max_idx]
        else:
            out_trans = trans[0]

        outf.write(out_trans+"\n")
        NUM += 1
        if NUM % 50 == 0:
            print(f"###Processing {NUM} examples.")
            outf.flush()
            sys.stdout.flush()

    inf.close()
    outf.close()




