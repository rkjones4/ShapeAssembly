import torch
from model_prog import Sampler, MLP, FDGRU, ENCGRU, progToData, get_encoding, run_eval_decoder
from hier_execute import hier_execute
import utils
import sys
from tqdm import tqdm

sa_enc = "<PATH TO PRE-TRAINED ShapeAssembly ENCODER>"

device = torch.device("cuda")

def getInds(train_ind_file):
    inds = set()
    with open(train_ind_file) as f:
        for line in f:
            inds.add(line.strip())
    return inds

def do_sa():
    train_ind_file = f'data_splits/chair/train.txt'
    train_inds = list(getInds(train_ind_file))

    encoder = torch.load(sa_enc).to(device)
    
    for ind in tqdm(train_inds):
    
        rprog = utils.loadHPFromFile(f'data/squeeze_chair/{ind}.txt')
        shape = progToData(rprog)        
        enc, _ = get_encoding(shape, encoder, mle=True)

        torch.save(enc, f'sa_encs/{ind}.enc')            
        
if __name__ == '__main__':
    do_sa()
