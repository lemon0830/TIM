import torch
import argparse

def init_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=".",  type=str)
    return parser

if __name__ == '__main__':
    arg_parser = init_opt()

    args = arg_parser.parse_args()

    modelpath = args.model_path

    state = torch.load(modelpath+"/pytorch_model.bin")
    new_state = {}
    for k in state:
        print(k)
        newk = k.replace("rwmodel.", "")
        # if "model" not in k and "llama" in modelpath:
        #     newk = "model." + k
        # elif "transformer" not in k and "bloomz" in modelpath:
        #     newk = "transformer." + k
        print(newk)
        new_state[newk] = state[k]
    torch.save(new_state, modelpath+"/pytorch_model.bin")
