"""
generate_instruction_following_reward_data.py

"""

import argparse
import os
import json
import random
import re
import copy
from multiprocessing import Process

def load_prompts(file_name, task):
    if os.path.exists(file_name):
        f = open(file_name)
        prompts = json.load(f)

        return prompts[task]

    return []

def encode_prompt(prompt_instructions, input, output, bad, SRC="英文", TGT="中文", refs=[]):
    """Encode multiple prompt instructions into a single string."""

    task_dict = {}
    prompt_size = len(prompt_instructions)

    corpus = []

    prompt_id = random.randint(0, prompt_size - 1)


    instruction = prompt_instructions[prompt_id]

    instruction = re.sub(r"\s+", " ", instruction).rstrip(":").rstrip("：")
    input = re.sub(r"\s+", " ", input)
    output = re.sub(r"\s+", " ", output)
    bad = re.sub(r"\s+", " ", bad)

    if "<S1>" in instruction:
        instruction = instruction.replace("<S1>", input)
        input = ""

        if "<S2>" in instruction:
            split_output = output.split(" ")
            tmp = {}
            Len_tgt = len(split_output)
            if Len_tgt >= 3:
                Num = random.randint(1, Len_tgt//2)
                for i in range(0, Num):
                    tmp[split_output[i]] = 1
                instruction = instruction.replace("<S2>", ",".join(tmp))
            else:
                return []
    else:
        input = "" if input.lower() == "" else input

    task_dict["instruction"] = instruction.replace("[SRC]", SRC).replace("[TGT]", TGT)

    task_dict["input"] = input

    task_dict["output"] = output

    if bad == output:
        if len(refs) > 0:
            idx = random.randint(0, len(refs) -1)
            bad = refs[idx]
        else:
            bad = "I don't know."

    task_dict["bad_output"] = bad

    corpus.append(copy.deepcopy(task_dict))

    return corpus

def process_two_file(srcfile0, srcfile1, srcfile2, tgtfile, args):
    prompts = load_prompts(args.prompt_file, args.task)

    print(f"Loading task {args.task} prompts with Size {len(prompts)} from {args.prompt_file} ...")

    fout = open(os.path.join(args.output_data_path, tgtfile), "w", encoding='utf-8')

    print(args.output_data_path, tgtfile)
    instruct_data_corpus = []

    fsrc = open(os.path.join(args.input_data_path, srcfile0), "r")
    ftgt = open(os.path.join(args.input_data_path, srcfile1), "r")
    fbad = open(os.path.join(args.input_data_path, srcfile2), "r")

    count = 0

    refs = []

    for src, tgt, bad in zip(fsrc, ftgt, fbad):
        src = src.strip()
        tgt = tgt.strip()
        bad = bad.strip()

        refs.append(tgt)
        refs.append(bad)

        SRC = args.src
        TGT = args.tgt

        example = encode_prompt(prompts, input=src, output=tgt, bad=bad, SRC=SRC, TGT=TGT, refs=refs)

        count += 1
        instruct_data_corpus.extend(example)

        if count % 1000 == 0:
            print(f"Processing {count} data...")

    json.dump(instruct_data_corpus, fout, indent=4, ensure_ascii=False)

    print(f"Generating Task {args.task} Data with Size {len(instruct_data_corpus)}")

def init_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_data_path", default=".", help="Path of the input data", type=str)
    parser.add_argument("--output_data_path", default=".", help="Path of the input data", type=str)
    parser.add_argument("-i", "--input_files", help="name of the input file", nargs='+')
    parser.add_argument("-o", "--output_file", help="name of the output file", type=str)
    parser.add_argument("-p", "--prompt_file", help="input file of instructs", type=str)
    parser.add_argument("--thread_num", default=0, help="multi processs", type=int)
    parser.add_argument("-t", "--task", default="EN_NMT", help="input file of instructs", type=str)
    parser.add_argument("--src", default=None, help="source language", type=str)
    parser.add_argument("--tgt", default=None, help="target language", type=str)


    return parser


if __name__ == "__main__":
    arg_parser = init_opt()
    args = arg_parser.parse_args()

    input_files = args.input_files

    thread_num = args.thread_num

    if len(input_files) == 1:
        pass
    elif len(input_files) == 3:
        process_list = []
        thread_num = thread_num // 3

        if thread_num > 0:

            for id in range(thread_num):
                if id < 10:
                    id = '0' + str(id)
                else:
                    id = str(id)

                # 多进程
                p = Process(target=process_two_file, args=(input_files[0] + id,
                                                           input_files[1] + id,
                                                           input_files[2] + id,
                                                           args.output_file + id,
                                                           args
                                                           ))
                process_list.append(p)

            for i, p in enumerate(process_list):
                p.start()
                print('Child process {} start.'.format(i + 1))

            for i, p in enumerate(process_list):
                p.join()
                print('Child process {} end.'.format(i + 1))
        else:
            process_two_file(input_files[0], input_files[1], input_files[2], args.output_file, args)






