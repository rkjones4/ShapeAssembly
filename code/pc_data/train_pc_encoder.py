import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import faiss
import numpy as np
import time
import random
import sys
from pc_encoder import PCEncoder

BATCH_SIZE = 16
device = torch.device("cuda")
VAE = False
DIM = 256
LR = 0.001
EPOCHS = 5000

MAX_TRAIN = 100000
MAX_EVAL = 100000

PC_DIR = "<WHERE PC ARE LOCATED>"

ho_perc = 0.05

SAVE_EP = 25

rd_seed = 42
random.seed(rd_seed)
np.random.seed(rd_seed)
torch.manual_seed(rd_seed)

def log_print(s, of):
    with open(of, 'a') as f:
        f.write(f"{s}\n")
    print(s)

# Simple version of PC Encoder, in case don't want to use PointNet
    
class simplePCEncoder(nn.Module):

    def __init__(self, feat_len):
        super(simplePCEncoder, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, feat_len, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(feat_len)
        self.mlp2mu = nn.Linear(feat_len, feat_len)

    def forward(self, pc):
        net = pc.transpose(2, 1)
        net = torch.relu(self.bn1(self.conv1(net)))
        net = torch.relu(self.bn2(self.conv2(net)))
        net = torch.relu(self.bn3(self.conv3(net)))
        net = torch.relu(self.bn4(self.conv4(net)))

        net = net.max(dim=2)[0]

        return self.mlp2mu(net)
                

def collate(samples):
    encs = [s[0] for s in samples]
    pcs = [s[1] for s in samples]
    return torch.stack(encs), torch.stack(pcs)


def load_data(enc_dir):

    inds = os.listdir(enc_dir)
    inds = [i.split('.')[0] for i in inds]

    random.shuffle(inds)        

    num_train = len(inds) - int((ho_perc * len(inds)))

    train_inds = inds[:num_train]
    val_inds = inds[num_train:]

    train_samples = []
    val_samples = []
    
    for ind in train_inds:
        enc = torch.load(f"{enc_dir}/{ind}.enc").squeeze().to(device)
        pts = torch.load(f"{PC_DIR}/{ind}.pts").to(device)
        train_samples.append((enc, pts))
        
    for ind in val_inds:
        enc = torch.load(f"{enc_dir}/{ind}.enc").squeeze().to(device)
        pts = torch.load(f"{PC_DIR}/{ind}.pts").to(device)
        val_samples.append((enc, pts))

    print(f"Num Train {len(train_samples)}")
    print(f"Num Val {len(val_samples)}")
            
    train_loader = DataLoader(train_samples[:MAX_TRAIN], BATCH_SIZE, shuffle=True, collate_fn = collate)
    val_loader = DataLoader(val_samples[:MAX_EVAL], BATCH_SIZE, shuffle=False, collate_fn = collate)

    return train_loader, val_loader
    
def train(enc_dir, out_dir):
    os.system(f'mkdir {out_dir}')
    train_loader, val_loader = load_data(enc_dir)

    enc = PCEncoder()
    enc.to(device)

    enc_opt = torch.optim.Adam(
        enc.parameters(),
        lr = LR,
        weight_decay = 1e-5
    )

    # learning rate scheduler
    enc_sch = torch.optim.lr_scheduler.StepLR(
        enc_opt, step_size=500, gamma=0.9)
    
    for ep in range(EPOCHS):
        t = time.time()
        tl = 0.

        for i, (tar_enc, pc) in enumerate(train_loader):
            
            if pc.shape[0] == 1:
                continue
            
            pred_enc = enc(pc)                        
            
            loss = torch.nn.functional.mse_loss(input=pred_enc, target=tar_enc, reduction='none').sum(dim=1).mean()
            
            enc_opt.zero_grad()                    
            loss.backward()
            enc_opt.step()
                            
            tl += loss.detach().item()
            
        enc_sch.step()
            
        tl /= i+1
        vl = 0.
        
        with torch.no_grad():
            for i, (tar_enc, pc) in enumerate(val_loader):
                if pc.shape[0] == 1:
                    continue
                
                pred_enc = enc(pc)

                loss = torch.nn.functional.mse_loss(input=pred_enc, target=tar_enc, reduction='none').sum(dim=1).mean()
                vl += loss.detach().item()
                        
        vl /= i+1

        if ep % SAVE_EP == 0:
            torch.save(enc.state_dict(), f"{out_dir}/enc_dict_{ep}.pt")
        
        log_print(f"Epoch {ep}, time: {time.time() - t} | TRAIN : {tl} | VAL : {vl}", f"{out_dir}/log.txt")

        
if __name__ == '__main__':
    # First arg is encoding directory, should have all encodings saved as {ind.enc}.
    # Second arg is outdir -> where the model should be saved to
    train(sys.argv[1], sys.argv[2])
