# Valentin Mace
# valentin.mace@kedgebs.com
# Developed at Qwant Research

"""Main script to add noise to your corpus"""

import argparse

from noise_functions import *
from tqdm import tqdm
from utils import *

def has_chinese_char(w):
    return w and any(u"\u4e00" <= c <= u"\u9fa5" for c in w)


parser = argparse.ArgumentParser()
parser.add_argument('input',
                    help="The text file you want to add noise to")
parser.add_argument('--output', default=None,
                    help="Optional, the name you want to give to your output, default=yourfilename.noisy")
parser.add_argument('--progress', action='store_true',
                    help="Optional, show the progress")
parser.add_argument('--delete_probability', default=0.1, type=float,
                    help="Optional, the probability to remove each token, default=0.1")
parser.add_argument('--replace_probability', default=0.1, type=float,
                    help="Optional, the probability to replace each token with a filler token, default=0.1")
parser.add_argument('--permutation_range', default=3, type=int,
                    help="Optional, Max range for token permutation, default=3")
parser.add_argument('--filler_token', default='BLANK',
                    help="Optional, token to use for replacement function, default=BLANK")

if __name__ == '__main__':
    args = parser.parse_args()

    file_input = args.input
    file_output = file_input + ".noisy"
    if args.output:
        file_output = args.output

    lines_number = count_lines(file_input) if args.progress else None

    with open(file_input, 'r') as corpus, open(file_output, 'w') as output:
        # You can remove a noise function here, modify its parameters or add your own (writing it in noise_functions.py)
        for lines in tqdm(corpus, total=lines_number):

            chinese_flag = False
            if has_chinese_char(lines):
                chinese_flag = True
            lines = lines.split("\\n")

            if chinese_flag:
                lines = [" ".join(list(i)) for i in lines]


            new_lines = []

            for line in lines:
                if chinese_flag:
                    line = line.replace("\n", "")

                line = delete_random_token(line, probability=args.delete_probability)
                line = replace_random_token(line, probability=args.replace_probability, filler_token=args.filler_token)
                line = random_token_permutation(line, _range=args.permutation_range)

                if chinese_flag:
                    line = line.replace(" ","")

                new_lines.append(line)

            lines = "\\n".join(new_lines)
            output.write(lines + '\n')
