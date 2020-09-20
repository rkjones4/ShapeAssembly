import os
import torch
from tqdm import tqdm
import utils
from hier_execute_v1 import hier_execute

def loadPts(infile):
    pts = []
    with open(infile) as f:
        for line in f:
            ls = line[:-1].split()
            pts.append([float(l) for l in ls])

    return torch.tensor(pts).float()

def getInds(train_ind_file):
    inds = set()
    with open(train_ind_file) as f:
        for line in f:
            inds.add(line.strip())
    return inds

category = 'chair'

train_inds = getInds(f'data_splits/{category}/train.txt')
val_inds = getInds(f'data_splits/{category}/val.txt')
train_sn_inds = getInds(f'data_splits/sn_{category}/train.txt')
val_sn_inds = getInds(f'data_splits/sn_{category}/val.txt')

all_val = val_sn_inds.intersection(val_inds)
all_train = train_inds.union(train_sn_inds)

all_inds = list(all_val.union(all_train))
odir = 'partnet_pc_sample'
os.system(f'mkdir {odir}')

for ind in tqdm(all_inds):
    pts = loadPts(f'/home/kenny/partnet/data_v0/{ind}/point_sample/pts-10000.txt')    
    torch.save(pts, f'{odir}/{ind}.pts')
