import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import losses
from parse_prog import progToTarget, predToProg
from argparse import Namespace
from ShapeAssembly import hier_execute
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import argparse
from tqdm import tqdm
import ast
import metrics
from sem_valid import semValidGen

"""
Modeling logic for a generative model of ShapeAssembly Programs. 

Encoder is defined in ENCGRU. Decoder is defined in FDGRU. 

run_train_decoder has the core logic for training in a teacher-forced manner
run_eval_decoder has the core logic for evaluating in an auto-regressive manner

run_train is the "main" entrypoint.

"""

outpath = "model_output"
device = torch.device("cuda")
#device = torch.device("cpu")

INPUT_DIM = 63 # tensor representation of program. Check ProgLoss in utils for a detailed comment of how lines in ShapeAssembly are represented as Tensors
MAX_PLINES = 100 # Cutoff number of lines in eval hierarchical program
MAX_PDEPTH = 10 # Cutoff number of programs in eval hierarchical program
ADAM_EPS = 1e-6 # increase numerical stability
VERBOSE = True
SAVE_MODELS = True

# Program reconstruction loss logic 
fploss = losses.ProgLoss()
closs = torch.nn.BCEWithLogitsLoss()

# A 'tokenization' of the line command 
def cleanCommand(struct):
    assert struct.shape[1] == 1, 'bad input to clean command'
    struct = struct.squeeze()
    new = torch.zeros(7, dtype=torch.float).to(device)

    c = torch.argmax(struct[:7])
    new[c] = 1.0
                
    new = new.unsqueeze(0).unsqueeze(0)
            
    return new

# A 'tokenization' of the line cube indices 
def cleanCube(struct):
    assert struct.shape[1] == 1, 'bad input to clean cube'
    struct = struct.squeeze()
    new = torch.zeros(33, dtype=torch.float).to(device)
    
    c1 = torch.argmax(struct[:11])        
    new[c1] = 1.0

    c2 = torch.argmax(struct[11:22])
    new[11+c2] = 1.0

    c3 = torch.argmax(struct[22:33])
    new[22+c3] = 1.0
    
    new = new.unsqueeze(0).unsqueeze(0)
            
    return new
        
# Multi-layer perceptron helper function
class MLP(nn.Module):
    def __init__(self, ind, hdim1, hdim2, odim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(ind, hdim1)
        self.l2 = nn.Linear(hdim1, hdim2)
        self.l3 = nn.Linear(hdim2, odim)

    def forward(self, x):
        x = F.leaky_relu(self.l1(x),.2)
        x = F.leaky_relu(self.l2(x),.2)
        return self.l3(x)


# GRU recurrent Decoder
class FDGRU(nn.Module):
    def __init__(self, hidden_dim):
        super(FDGRU, self).__init__()
        
        self.bbdimNet = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first = True)        

        self.inp_net = MLP(
            INPUT_DIM + 3 + 3,
            hidden_dim,
            hidden_dim,
            hidden_dim
        )

        self.cmd_net = MLP(hidden_dim, hidden_dim // 2, hidden_dim //4, 7)
        
        self.cube_net = MLP(hidden_dim + 7, hidden_dim //2, hidden_dim //4, 33)
        
        self.dim_net = MLP(
            hidden_dim + 3,
            hidden_dim // 2,
            hidden_dim // 4,
            3
        )

        self.align_net = MLP(
            hidden_dim + 3,
            hidden_dim // 2,
            hidden_dim // 4,
            1
        )            

        self.att_net = MLP(
            hidden_dim + 22,
            hidden_dim,
            hidden_dim,
            6
        )

        # take in cuboids involved -> predict the face and uv
        self.squeeze_net = MLP(
            hidden_dim + 33,
            hidden_dim // 2,
            hidden_dim // 4,
            8
        )

        self.sym_net = MLP(
            hidden_dim + 3 + 11,
            hidden_dim // 2,
            hidden_dim // 4,
            5
        )

        self.next_net = MLP(hidden_dim*2, hidden_dim, hidden_dim, hidden_dim)
        self.leaf_net = MLP(hidden_dim*2, hidden_dim//2, hidden_dim//8, 1)

    # Inp squence is the input sequence, h is the current hidden state, h_start is the hidden state at the beginning of the local program
    # bb_dims are the dimensions of the bbox, hier_ind is the depth of the hierarchy. gt_struct_seq contains the target cubes + command information for further-teacher forcing during training
    def forward(self, inp_seq, h, h_start, bb_dims, hier_ind, gt_struct_seq=None):

        bb_dims = bb_dims.unsqueeze(0).unsqueeze(0).repeat(1,inp_seq.shape[1],1)
            
        hier_oh = torch.zeros(1, inp_seq.shape[1], 3).to(device)
        hier_oh[0, :, min(hier_ind,2)] = 1.0
                
        inp = self.inp_net(
            torch.cat(
                (inp_seq, bb_dims, hier_oh), dim=2)
        )
            
        gru_out, h = self.gru(inp, h)
        
        cmd_out = self.cmd_net(gru_out)

        if gt_struct_seq is not None:
            clean_cmd = gt_struct_seq[:,:,:7]            
        else:            
            clean_cmd = cleanCommand(cmd_out)
                
        cube_out = self.cube_net(
            torch.cat((gru_out, clean_cmd), dim = 2)
        )

        if gt_struct_seq is not None:
            clean_cube = gt_struct_seq[:,:,7:40]
        else:
            clean_cube = cleanCube(cube_out)
            
        dim_out = self.dim_net(
            torch.cat((gru_out, bb_dims), dim = 2)
        )

        align_out = self.align_net(
            torch.cat((gru_out, bb_dims), dim = 2)
        )

        att_out = self.att_net(
            torch.cat((gru_out, clean_cube[:,:,:22]), dim = 2)
        )

        sym_out = self.sym_net(
            torch.cat((gru_out, clean_cube[:,:,:11], bb_dims), dim =2)
        )

        squeeze_out = self.squeeze_net(
            torch.cat((gru_out, clean_cube), dim=2)
        )
        
        out = torch.cat(
            (cmd_out, cube_out, dim_out, att_out, sym_out, squeeze_out, align_out), dim=2
        )
        
        next_out = self.next_net(
            torch.cat((
                gru_out, h_start.repeat(1, gru_out.shape[1], 1)
            ), dim = 2)
        )

        leaf_out = self.leaf_net(
            torch.cat((
                gru_out, h_start.repeat(1, gru_out.shape[1], 1)
            ), dim = 2)
        )
        
        return out, next_out, leaf_out, h


# Helper class for bottleneck of VAE. If mle is set to true, returns the mean. Otherwise, returns a sample of the predictions mean
# and standard deviations using the standard re-parameterization trick (along with the KL divergence). 
class Sampler(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(Sampler, self).__init__()        
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, feature_size)
        self.mlp2var = nn.Linear(hidden_size, feature_size)

    def forward(self, x, mle):
        encode = torch.relu(self.mlp1(x))
        mu = self.mlp2mu(encode)

        if mle:
            return mu

        else:
            logvar = self.mlp2var(encode)
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)

            kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            return torch.cat([eps.mul(std).add_(mu), kld], 2)


# GRU recurrent Encoder
class ENCGRU(nn.Module):
    def __init__(self, hidden_dim):
        super(ENCGRU, self).__init__()

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first = True)        
        self.inp_net = MLP(INPUT_DIM, hidden_dim, hidden_dim, hidden_dim)
        self.sampler = Sampler(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.h_start = torch.nn.Parameter(
            torch.randn(1, 1, hidden_dim, device=device, requires_grad=True)
        )
        
    def encode_program(self, hp):
        if len(hp) == 0:
            return torch.zeros(1, 1, self.hidden_dim).to(device)
        
        inp_seq = torch.cat((hp['inp'], hp['tar'][-1,:,:].unsqueeze(0)), dim=0).transpose(0,1)

        children = [{}] + hp['children'] + [{} for _ in range(inp_seq.shape[1] - len(hp['children']) - 1)]

        child_mask = torch.tensor([1 if len(c) > 0 else 0 for c in children]).float().to(device)
        child_encs = torch.stack([self.encode_program(c) for c in children], dim=2).view(1, -1, self.hidden_dim)
                
        local_inp = self.inp_net(inp_seq)

        inp = (child_mask.unsqueeze(0).unsqueeze(2) * child_encs) + ((1-child_mask).unsqueeze(0).unsqueeze(2) * local_inp)

        # We don't care about the token predictions, just that the program has been encoded into the hidden state
        _, h = self.gru(inp, self.h_start)
        
        return h

    def get_latent_code(self, hp, mle):
        x = self.encode_program(hp)
        return self.sampler(x, mle)

# Logic for training the decoder (decoder) on a single hierarchical program prog (we don't use batching because hierarchical data +
# batching = hard). h0 is the latent code to be decoded
def run_train_decoder(prog, h0, decoder):
    
    shape_result = {}
    
    bb_pred = decoder.bbdimNet(h0.squeeze())
    bb_loss = (bb_pred - prog["bb_dims"]).abs().sum()

    shape_result['bb'] = bb_loss
        
    # Create a queue of programs
    q = [(prog, h0, 0)]
    num_progs = 0.
    num_lines = 0.
    
    while len(q) > 0:

        node, h, hier_ind = q.pop(0)
        nc_ind = 0
        
        num_progs += 1.
        
        children = node["children"]
        inp_seq = node["inp"].transpose(0,1)
        tar_seq = node["tar"].transpose(0,1)
        weights = node["weights"].unsqueeze(0)
        bb_dims = node["bb_dims"]
                
        num_lines += inp_seq.shape[1]

        h_start = h.clone()

        # Teacher forcing the decoder
        pout, pnext, pleaf, _ = decoder(
            inp_seq, h, h_start, bb_dims, hier_ind, tar_seq[:,:,:40]
        )

        # This is the core reconstruction loss calculation
        prog_result = fploss(
            pout,
            tar_seq,            
            weights
        )
        
        for key in prog_result:
            if key not in shape_result:
                shape_result[key] = prog_result[key]
            else:
                shape_result[key] += prog_result[key]
        
        cub_inds = (
            torch.argmax(tar_seq[:, :, :7], dim = 2) == 1
        ).squeeze().nonzero().squeeze()
        
        cub_pleaf = pleaf[:, cub_inds, :]

        lt = torch.tensor([0.0 if len(c) > 0 else 1.0 for c in children]).to(device)
        _leaf_loss = closs(cub_pleaf.squeeze(), lt)

        _cleaf = ((cub_pleaf.squeeze() > 0.).float() == lt).sum().float()
        
        if 'leaf' not in shape_result:
            shape_result['leaf'] = _leaf_loss
            shape_result['cleaf'] = _cleaf
        else:
            shape_result['leaf'] += _leaf_loss
            shape_result['cleaf'] += _cleaf
            
        for i in range(cub_inds.shape[0]):
            child = children[nc_ind]
            nc_ind += 1
            if len(child) > 0:
                q.append(
                    (child, pnext[:,cub_inds[i],:].unsqueeze(0), hier_ind+1)
                )

    shape_result['np'] = num_progs
    shape_result['nl'] = num_lines

    return shape_result

# Runs one epoch of training on the dataset
def model_train_results(dataset, encoder, decoder, dec_opt, enc_opt, variational, loss_config, name, do_print, exp_name):
    ep_result = {}
    bc = 0.
    for batch in dataset:
        bc += 1. 
        batch_result = model_train(
            batch, encoder, decoder, dec_opt, enc_opt, \
            variational, loss_config
        )

        for key in batch_result:
            res = batch_result[key]
                
            if torch.is_tensor(res):
                res = res.detach()
                        
            if key not in ep_result:                    
                ep_result[key] = res
            else:
                ep_result[key] += res

    if len(ep_result) == 0:
        return {}
                
    arl = 0.
    
    for loss in loss_config:        
        ep_result[loss] /= bc
        if loss == 'kl':
            continue            
        if torch.is_tensor(ep_result[loss]):
            arl += ep_result[loss].detach().item()
        else:
            arl += ep_result[loss]
            
    ep_result['recon'] = arl
    if ep_result['nl'] > 0:
        ep_result['cmdc'] /= ep_result['nl']
    if ep_result['na'] > 0:
        ep_result['cubc'] /= ep_result['na']
    if ep_result['nc'] > 0:
        ep_result['cleaf'] /= ep_result['nc']
    if ep_result['nap'] > 0:
        ep_result['palignc'] /= ep_result['nap']
    if ep_result['nan'] > 0:
        ep_result['nalignc'] /= ep_result['nan']
    if ep_result['ns'] > 0:
        ep_result['sym_cubc'] /= ep_result['ns']
        ep_result['axisc'] /= ep_result['ns']
    if ep_result['nsq'] > 0:
        ep_result['sq_cubc'] /= ep_result['nsq']
        ep_result['facec'] /= ep_result['nsq']
        
    ep_result.pop('na')
    ep_result.pop('nl')
    ep_result.pop('nc')
    ep_result.pop('nap')
    ep_result.pop('nan')
    ep_result.pop('np')
    ep_result.pop('ns')
    ep_result.pop('nsq')

    if do_print:
        utils.log_print(
            f"""
  TF Results for {name}

  Recon Loss = {ep_result['recon']}
  Cmd Loss = {ep_result['cmd']}
  Cub Prm Loss = {ep_result['cub_prm']}
  XYZ Prm Loss = {ep_result['xyz_prm']}
  UV Prm Loss = {ep_result['uv_prm']}
  Sym Prm Loss = {ep_result['sym_prm']}
  Cub Loss = {ep_result['cub']}
  Squeeze Cub Loss = {ep_result['sq_cub']}
  Sym Cub Loss = {ep_result['sym_cub']}
  Sym Axis Loss = {ep_result['axis']}
  Face Loss = {ep_result['face']}
  Leaf Loss = {ep_result['leaf']}
  Align Loss = {ep_result['align']}
  KL Loss = {ep_result['kl'] if 'kl' in ep_result else None}
  BBox Loss = {ep_result['bb']}
  Cmd Corr % {ep_result['cmdc']}
  Cub Corr % {ep_result['cubc']}
  Sq Cubb Corr % {ep_result['sq_cubc']}
  Face Corr % {ep_result['facec']}
  Leaf Corr % {ep_result['cleaf']}
  Align Pos Corr = {ep_result['palignc']}
  Align Neg Corr = {ep_result['nalignc']}
  Sym Cub Corr % {ep_result['sym_cubc']}
  Sym Axis Corr % {ep_result['axisc']}""", f"{outpath}/{exp_name}/log.txt")
    
    return ep_result

# Full encoder + decoder training logic for a single program (i.e. a batch)
def model_train(batch, encoder, decoder, dec_opt, enc_opt, variational, loss_config):
    decoder.train()
    encoder.train()

    batch_result = {}

    # Should only be one shape in batch
    for shape in batch:
        encoding, kl_loss = get_encoding(shape[0], encoder, not variational)

        if encoding is not None:

            if 'kl' in loss_config:
                if 'kl' not in batch_result:
                    batch_result['kl'] = kl_loss
                else:
                    batch_result['kl'] += kl_loss


            shape_result = run_train_decoder(
                shape[0], encoding, decoder
            )
            
            for res in shape_result:                
                if res not in batch_result:
                    batch_result[res] = shape_result[res]
                else:
                    batch_result[res] += shape_result[res]
                                        
    loss = 0.
    if len(batch_result) > 0:
        for key in loss_config:
            batch_result[key] *= loss_config[key]
            if torch.is_tensor(batch_result[key]):            
                loss += batch_result[key]

    if torch.is_tensor(loss) and enc_opt is not None and dec_opt is not None:
        dec_opt.zero_grad()
        enc_opt.zero_grad()                    
        loss.backward()
        dec_opt.step()
        enc_opt.step()

    return batch_result


# Given the decoder's predictions, create a well-structured ShapeAssembly Program (in text)
def getHierProg(hier_prog, all_preds):
    if len(hier_prog) == 0:
        return
    prog = predToProg(all_preds[hier_prog["name"]])
    hier_prog["prog"] = prog
    for c in hier_prog["children"]:
        getHierProg(c, all_preds)

        
# Decode latent code in a hierarchical shapeAssembly program using decoder in an auto-regressive manner.
def run_eval_decoder(h0, decoder, rejection_sample, gt_prog = None):

    index = 0

    bb_pred = decoder.bbdimNet(h0.squeeze())    
    
    hier_prog = {
        "children": [],
        "bb_dims": bb_pred
    }
    
    q = [(h0, hier_prog, 0, gt_prog)]        
    pc = 0
    num_lines = 0.
    all_preds = []
    
    shape_result = {
        'corr_line_num': 0.,
        'bad_leaf': 0.
    }

    if gt_prog is not None and len(gt_prog) > 0:
        bb_loss = (bb_pred - gt_prog["bb_dims"]).abs().sum()
        shape_result['bb'] = bb_loss    

    
    while len(q) > 0:
        pc += 1
                
        h, prog, hier_ind, gt_prog = q.pop(0)

        prog["name"] = index
        index += 1
        
        if gt_prog is None or len(gt_prog) == 0:
            shape_result['bad_leaf'] += 1.

        # Semantic validity logic that handles local program creation
        preds, prog_out, next_q = semValidGen(
            prog, decoder, h, hier_ind, MAX_PLINES, INPUT_DIM, device, gt_prog, rejection_sample
        )
        
        num_lines += len(preds)
        
        q += next_q

        # Logic for calculating loss / metric performance in eval mode
        if gt_prog is not None and len(gt_prog) > 0:
            gt_tar_seq = gt_prog["tar"].transpose(0,1)
            gt_weights = gt_prog["weights"].unsqueeze(0)
            gt_bb_dims = gt_prog["bb_dims"]
            
            try:
                if len(prog_out) > 1:
                    prog_out = torch.cat([p for p in prog_out], dim = 1)
                else:
                    prog_out = torch.zeros(1,1,INPUT_DIM).to(device)
                    
                if prog_out.shape[1] == gt_tar_seq.shape[1]:
                    shape_result['corr_line_num'] += 1.
                
                prog_result = fploss(
                    prog_out[:,:gt_tar_seq.shape[1],:],
                    gt_tar_seq,
                    gt_weights
                )
                
                for key in prog_result:
                    if key not in shape_result:
                        shape_result[key] = prog_result[key]
                    else:
                        shape_result[key] += prog_result[key]

            except Exception as e:
                if VERBOSE:
                    print(e)
                pass
                                        
        all_preds.append(preds)
        # Have a max number of programs to decode for any given root program
        if pc > MAX_PDEPTH:
            break        

    shape_result['np'] = pc
    shape_result['nl'] = num_lines
    
    try:
        getHierProg(hier_prog, all_preds)                
        return hier_prog, shape_result
            
    except Exception as e:
        if VERBOSE:
            print(f"FAILED IN EVAL DECODER WITH {e}")
        return None, shape_result

# Runs an epoch of evaluation logic on the train + val datasets
def model_eval(eval_train_dataset, eval_val_dataset, encoder, decoder, exp_name, epoch, num_gen):
    decoder.eval()
    encoder.eval()

    eval_results = []

    for name, dataset in [('train', eval_train_dataset), ('val', eval_val_dataset)]:

        if len(dataset) == 0:
            continue
        
        named_results = {
            'count': 0.,
            'miss_hier_prog': 0.
        }

        recon_sets = []
        
        for batch in dataset:        
            for shape in batch:
                
                named_results[f'count'] += 1.
                
                # Always get maximum likelihood estimation (i.e. mean) of shape encoding at eval time
                encoding, _ = get_encoding(shape[0], encoder, mle=True)
                
                prog, shape_result = run_eval_decoder(
                    encoding, decoder, False, shape[0]
                )

                for key in shape_result:
                    nkey = f'{key}'
                    if nkey not in named_results:
                        named_results[nkey] = shape_result[key]
                    else:
                        named_results[nkey] += shape_result[key]

                if prog is None:
                    named_results[f'miss_hier_prog'] += 1.
                    continue
                                        
                recon_sets.append((prog, shape[0], shape[1]))

        # For reconstruction, get metric performance
        recon_results, recon_misses = metrics.recon_metrics(
            recon_sets, outpath, exp_name, name, epoch, VERBOSE
        )

        for key in recon_results:
            named_results[key] = recon_results[key]
        
        named_results[f'miss_hier_prog'] += recon_misses
        
        named_results[f'prog_creation_perc'] = (
            named_results[f'count'] - named_results[f'miss_hier_prog']
        ) / named_results[f'count']

        eval_results.append((name, named_results))

    gen_progs = []

    gen_prog_fails = 0.

    # Also generate a set of unconditional ShapeAssembly Programs
    for i in range(0, num_gen):
        try:
            h0 = torch.randn(1, 1, args.hidden_dim).to(device)
            prog, _ = run_eval_decoder(h0, decoder, True)
            gen_progs.append(prog)

        except Exception as e:
            gen_prog_fails += 1.
            
            if VERBOSE:
                print(f"Failed generating new program with {e}")

    # Get metrics for unconditional generations
    gen_results, gen_misses = metrics.gen_metrics(
        gen_progs, outpath, exp_name, epoch, VERBOSE
    )

    if num_gen > 0:
        gen_results['prog_creation_perc'] = (num_gen - gen_misses - gen_prog_fails) / num_gen

    else:
        gen_results['prog_creation_perc'] = 0.
                                
    return eval_results, gen_results

# Given a sample (a hierarchical program) get the encoding of it and the KL loss
def get_encoding(sample, encoder, mle=False):
            
    hd = encoder.hidden_dim
    sample = encoder.get_latent_code(sample, mle)
    
    enc = sample[:, :, :hd]
    kld = sample[:, :, hd:]

    if mle:
        return enc, 0.0        
    else:
        return enc, -kld.sum()                        

# convert a text based hierarchical program into a tensorized version
def progToData(prog):
    if len(prog) == 0:
        return {}
    
    inp, tar, weights, bb_dims = progToTarget(prog["prog"])
    prog["inp"] = inp.unsqueeze(1).to(device)
    prog["tar"] = tar.unsqueeze(1).to(device)
    prog["weights"] = weights.to(device)    
    prog["children"] = [progToData(c) for c in prog["children"]]
    prog["bb_dims"] = bb_dims.to(device)
        
    return prog

# Dummy collate function
def _col(samples):
    return samples

# Used to re-start training from a previous run
def loadConfigFile(exp_name):
    args = None
    with open(f"{outpath}/{exp_name}/config.txt") as f:
        for line in f:
            args = eval(line)

    assert args is not None, 'failed to load config' 
    return args
    
# Set-up new experiment directory
def writeConfigFile(args):
    os.system(f'mkdir {outpath} > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name} > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/plots > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/plots/train > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/plots/eval > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/plots/gen > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/programs > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/programs/train > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/programs/val > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/programs/gen > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/programs/gt > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/objs > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/objs/train > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/objs/val > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/objs/gen > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/objs/gt > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/models > /dev/null 2>&1')
    with open(f"{outpath}/{args.exp_name}/config.txt", "w") as f:
        f.write(f"{args}\n")

# Load a dataset of hierarchical ShapeAssembly Programs
def load_progs(dataset_path, max_shapes):
    inds = os.listdir(dataset_path)
    inds = [i.split('.')[0] for i in inds[:max_shapes]]
    good_inds = []
    progs = []
    for ind in tqdm(inds):
        hp = utils.loadHPFromFile(f'{dataset_path}/{ind}.txt')
        if hp is not None and len(hp) > 0:
            progs.append(hp)
            good_inds.append(ind)
    return good_inds, progs

# Helper function for keeping consistent train/val splits
def getInds(train_ind_file):
    inds = set()
    with open(train_ind_file) as f:
        for line in f:
            inds.add(line.strip())
    return inds

# Main entry-point of modeling logic
def run_train(dataset_path, exp_name, max_shapes, epochs,
              hidden_dim, eval_per, variational, loss_config, enc_lr,
              dec_lr, enc_step, dec_step, enc_decay, dec_decay,
              batch_size, holdout_perc, rd_seed,
              print_per, num_gen, num_eval, keep_missing,
              category, load_epoch=None):

    random.seed(rd_seed)
    np.random.seed(rd_seed)
    torch.manual_seed(rd_seed)
    
    raw_indices, progs = load_progs(dataset_path, max_shapes)

    inds_and_progs = list(zip(raw_indices, progs))
    random.shuffle(inds_and_progs)
                            
    inds_and_progs = inds_and_progs[:max_shapes]
    
    decoder = FDGRU(hidden_dim)
    decoder.to(device)

    encoder = ENCGRU(hidden_dim)
    encoder.to(device)
             
    print('Converting progs to tensors')

    samples = []
    for ind, prog in tqdm(inds_and_progs):
        nprog = progToData(prog)
        samples.append((nprog, ind))
            
    dec_opt = torch.optim.Adam(
        decoder.parameters(),
        lr = dec_lr,
        eps = ADAM_EPS
    )

    enc_opt = torch.optim.Adam(
        encoder.parameters(),
        lr = enc_lr,
        eps = ADAM_EPS
    )

    dec_sch = torch.optim.lr_scheduler.StepLR(
        dec_opt, 
        step_size = dec_step,
        gamma = dec_decay
    )

    enc_sch = torch.optim.lr_scheduler.StepLR(
        enc_opt, 
        step_size = enc_step,
        gamma = enc_decay
    )    

    train_ind_file = f'data_splits/{category}/train.txt'
    val_ind_file = f'data_splits/{category}/val.txt'
        
    train_samples = []
    val_samples = []

    train_inds = getInds(train_ind_file)
    val_inds = getInds(val_ind_file)

    misses = 0.
        
    for (prog, ind) in samples:
        if ind in train_inds:
            train_samples.append((prog, ind))            
        elif ind in val_inds:
            val_samples.append((prog, ind))
        else:
            if keep_missing:
                kept += 1
                if random.random() < holdout_perc:
                    val_samples.append((prog, ind))                    
                else:
                    train_samples.append((prog, ind))                    
            else:                
                misses += 1

    print(f"Samples missed: {misses}")
    train_num = len(train_samples)
    val_num = len(val_samples)
        
    train_dataset = DataLoader(train_samples, batch_size, shuffle=True, collate_fn = _col)
    eval_train_dataset = DataLoader(train_samples[:num_eval], batch_size=1, shuffle=False, collate_fn = _col)
    val_dataset = DataLoader(val_samples, batch_size, shuffle = False, collate_fn = _col)
    eval_val_dataset = DataLoader(val_samples[:num_eval], batch_size=1, shuffle = False, collate_fn = _col)
        
    utils.log_print(f"Training size: {train_num}", f"{outpath}/{exp_name}/log.txt")
    utils.log_print(f"Validation size: {val_num}", f"{outpath}/{exp_name}/log.txt")
    
    with torch.no_grad():
        gt_gen_results, _ = metrics.gen_metrics([s[0] for s in val_samples[:num_eval]], '', '', '', VERBOSE, False)

    utils.log_print(
f""" 
  GT Val Number of parts = {gt_gen_results['num_parts']}
  GT Val Variance = {gt_gen_results['variance']}
  GT Val Rootedness = {gt_gen_results['rootedness']}
  GT Val Stability = {gt_gen_results['stability']}
""", f"{outpath}/{exp_name}/log.txt")
    
    aepochs = []

    train_res_plots = {}
    val_res_plots = {}
    gen_res_plots = {}
    eval_res_plots = {'train': {}, 'val': {}}

    print('training ...')

    if load_epoch is None:
        start = 0
    else:
        start = load_epoch+1
    
    for e in range(start, epochs):
        do_print = (e+1) % print_per == 0
        t = time.time()
        if do_print:
            utils.log_print(f"\nEpoch {e}:", f"{outpath}/{exp_name}/log.txt")

        train_ep_result = model_train_results(train_dataset, encoder, decoder, dec_opt, enc_opt, variational, loss_config, 'train', do_print, exp_name)

        dec_sch.step()
        enc_sch.step()                            

        if do_print:
            utils.log_print(f"  Train Epoch Time = {time.time() - t}", f"{outpath}/{exp_name}/log.txt")
        
        if (e+1) % eval_per == 0:
                                            
            with torch.no_grad():
                t = time.time()
                utils.log_print(f"Doing Evaluation", f"{outpath}/{exp_name}/log.txt")

                val_ep_result = model_train_results(val_dataset, encoder, decoder, None, None, False, loss_config, 'val', True, exp_name)
                                
                eval_results, gen_results  = model_eval(
                    eval_train_dataset, eval_val_dataset, encoder, decoder, exp_name, e, num_gen
                )
                
                for name, named_results in eval_results:
                    if named_results['nc'] > 0:
                        named_results['cub_prm'] /= named_results['nc']
                        
                    if named_results['na'] > 0:
                        named_results['xyz_prm'] /= named_results['na']
                        named_results['cubc'] /= named_results['na']

                    if named_results['count'] > 0:
                        named_results['bb'] /= named_results['count']

                    if named_results['nl'] > 0:
                        named_results['cmdc'] /= named_results['nl']
                    
                    if named_results['ns'] > 0:
                        named_results['sym_cubc'] /= named_results['ns']
                        named_results['axisc'] /= named_results['ns']

                    if named_results['np'] > 0:
                        named_results['corr_line_num'] /= named_results['np']
                        named_results['bad_leaf'] /= named_results['np']

                    if named_results['nsq'] > 0:
                        named_results['uv_prm'] /= named_results['nsq']
                        named_results['sq_cubc'] /= named_results['nsq']
                        named_results['facec'] /= named_results['nsq']
                        
                    if named_results['nap'] > 0:
                        named_results['palignc'] /= named_results['nap']
                        
                    if named_results['nan'] > 0:
                        named_results['nalignc'] /= named_results['nan']
                        
                    named_results.pop('nc')
                    named_results.pop('nan')
                    named_results.pop('nap')
                    named_results.pop('na')
                    named_results.pop('ns')
                    named_results.pop('nsq')
                    named_results.pop('nl')
                    named_results.pop('count')
                    named_results.pop('np')
                    named_results.pop('cub')
                    named_results.pop('sym_cub')
                    named_results.pop('axis')
                    named_results.pop('cmd')
                    named_results.pop('miss_hier_prog')
                    
                    utils.log_print(
f"""

  Evaluation on {name} set:
                  
  Eval {name} F-score = {named_results['fscores']}
  Eval {name} IoU = {named_results['iou_shape']}
  Eval {name} PD = {named_results['param_dist_parts']}
  Eval {name} Prog Creation Perc: {named_results['prog_creation_perc']}
  Eval {name} Cub Prm Loss = {named_results['cub_prm']} 
  Eval {name} XYZ Prm Loss = {named_results['xyz_prm']}
  Eval {name} UV Prm Loss = {named_results['uv_prm']}
  Eval {name} Sym Prm Loss = {named_results['sym_prm']}
  Eval {name} BBox Loss = {named_results['bb']}
  Eval {name} Cmd Corr % {named_results['cmdc']}
  Eval {name} Cub Corr % {named_results['cubc']}
  Eval {name} Squeeze Cub Corr % {named_results['sq_cubc']}
  Eval {name} Face Corr % {named_results['facec']}
  Eval {name} Pos Align Corr % {named_results['palignc']}
  Eval {name} Neg Align Corr % {named_results['nalignc']}
  Eval {name} Sym Cub Corr % {named_results['sym_cubc']}
  Eval {name} Sym Axis Corr % {named_results['axisc']}
  Eval {name} Corr Line # % {named_results['corr_line_num']}
  Eval {name} Bad Leaf % {named_results['bad_leaf']}

""", f"{outpath}/{exp_name}/log.txt")

                utils.log_print(
f"""
  Gen Prog creation % = {gen_results['prog_creation_perc']}
  Gen Number of parts = {gen_results['num_parts']}
  Gen Variance = {gen_results['variance']}
  Gen Rootedness = {gen_results['rootedness']}
  Gen Stability = {gen_results['stability']}
""", f"{outpath}/{exp_name}/log.txt")
                    
                utils.log_print(f"Eval Time = {time.time() - t}", f"{outpath}/{exp_name}/log.txt")

                # Plotting logic
                
                for key in train_ep_result:
                    res = train_ep_result[key]
                    if torch.is_tensor(res):
                        res = res.detach().item()
                    if not key in train_res_plots:
                        train_res_plots[key] = [res]
                    else:
                        train_res_plots[key].append(res)

                for key in val_ep_result:
                    res = val_ep_result[key]
                    if torch.is_tensor(res):
                        res = res.detach().item()
                    if not key in val_res_plots:
                        val_res_plots[key] = [res]
                    else:
                        val_res_plots[key].append(res)
                        
                for key in gen_results:
                    res = gen_results[key]
                    if torch.is_tensor(res):
                        res = res.detach().item()
                    if not key in gen_res_plots:
                        gen_res_plots[key] = [res]
                    else:
                        gen_res_plots[key].append(res)
                        
                for name, named_results in eval_results:
                    for key in named_results:
                        res = named_results[key]
                        if torch.is_tensor(res):
                            res = res.detach().item()
                        if not key in eval_res_plots[name]:
                            eval_res_plots[name][key] = [res]
                        else:
                            eval_res_plots[name][key].append(res)
                        
                aepochs.append(e)                                                                
                
                for key in train_res_plots:
                    plt.clf()                    
                    plt.plot(aepochs, train_res_plots[key], label='train')
                    if key in val_res_plots:
                        plt.plot(aepochs, val_res_plots[key], label='val')
                    plt.legend()
                    if key == "recon":
                        plt.yscale('log')
                    plt.grid()
                    plt.savefig(f"{outpath}/{exp_name}/plots/train/{key}.png")

                for key in gen_res_plots:
                    plt.clf()                    
                    plt.plot(aepochs, gen_res_plots[key])
                    if key == "variance":
                        plt.yscale('log')
                    plt.grid()
                    plt.savefig(f"{outpath}/{exp_name}/plots/gen/{key}.png")
                
                for key in eval_res_plots['train']:
                    plt.clf()                    
                    t_p, = plt.plot(aepochs, eval_res_plots['train'][key], label='train')

                    if 'val' in eval_res_plots:
                        if key in eval_res_plots['val']:
                            v_p, = plt.plot(aepochs, eval_res_plots['val'][key], label='val')
                            plt.legend(handles=[t_p, v_p])
                    plt.grid()
                    plt.savefig(f"{outpath}/{exp_name}/plots/eval/{key}.png")
                                        
            try:
                if SAVE_MODELS:
                    utils.log_print("Saving Models", f"{outpath}/{exp_name}/log.txt")
                    # TODO: torch.save(x.state_dict(), so only the model parameters get saved (along with their names))
                    torch.save(decoder, f"{outpath}/{exp_name}/models/decoder_{e}.pt")
                    torch.save(encoder, f"{outpath}/{exp_name}/models/encoder_{e}.pt")
            except Exception as e:
                utils.log_print(f"Couldnt save models for {e}", f"{outpath}/{exp_name}/log.txt")

# Losses that will be used to train the model
def getLossConfig(args):
    loss_config = {
        'cmd': args.cmd_lw,
        'cub_prm': args.cub_prm_lw,

        'xyz_prm': args.att_prm_lw,
        'uv_prm': args.att_prm_lw,
        'sym_prm': args.att_prm_lw,
        
        'cub': args.cub_lw,
        'sym_cub': args.cub_lw,
        'sq_cub': args.cub_lw,

        'leaf': args.leaf_lw,
        'bb': args.bb_lw,
                
        'axis': args.axis_lw,
        'face': args.face_lw,
        'align': args.align_lw
    }

    if args.variational:
        loss_config['kl'] = args.kl_lw

    return loss_config


def run_generate(args):
    os.system(f'mkdir {outpath}/gen_{args.exp_name} > /dev/null 2>&1')

    decoder = torch.load(f"{outpath}/{args.model_name}/models/decoder_{args.load_epoch}.pt").to(device)
    encoder = torch.load(f"{outpath}/{args.model_name}/models/encoder_{args.load_epoch}.pt").to(device)
    
    random.seed(args.rd_seed)
    np.random.seed(args.rd_seed)
    torch.manual_seed(args.rd_seed)
    
    with torch.no_grad():
        if args.mode == "eval_gen":
            i = 0
            miss = 0
            while (i < args.num_gen):
                print(f"Gen {i}")
                try:
                    h0 = torch.randn(1, 1, args.hidden_dim).to(device)
                    prog, _ = run_eval_decoder(h0, decoder, True)
                    verts, faces = hier_execute(prog)
                    
                    utils.writeObj(verts, faces, f"{outpath}/gen_{args.exp_name}/gen_{i}.obj")                    
                    utils.writeHierProg(prog, f"{outpath}/gen_{args.exp_name}/gen_prog_{i}.txt")
                    i += 1
                    
                except Exception as e:
                    print(f"Failed to generate prog with {e}")
                    miss += 1

            print(f"Gen reject %: {miss / (args.num_gen + miss)}")
                    
        if args.mode == "eval_recon":            
            ind_file = f'data_splits/{args.category}/val.txt'
            inds = getInds(ind_file)
            for ind in tqdm(inds):
                gtprog = utils.loadHPFromFile(f'{args.dataset_path}/{ind}.txt')
                gtverts, gtfaces = hier_execute(gtprog)                
                shape = progToData(gtprog)
                enc, _ = get_encoding(shape, encoder, mle=True)
                prog, _ = run_eval_decoder(enc, decoder, False)
                verts, faces = hier_execute(prog)                
                utils.writeObj(verts, faces, f"{outpath}/gen_{args.exp_name}/{ind}_recon.obj")                                
                utils.writeObj(gtverts, gtfaces, f"{outpath}/gen_{args.exp_name}/{ind}_gt.obj")
                                                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run PGP model")
    parser.add_argument('-ds', '--dataset_path', help='Path to program data', type = str)
    parser.add_argument('-en', '--exp_name', help='name of experiment', type = str)
    parser.add_argument('-c', '--category', help='category', type = str)
    parser.add_argument('-mn', '--model_name', default=None, help='name of the model used for evaluation, do not specify for model_name == exp_name', type = str)
    
    parser.add_argument('-ms', '--max_shapes', default = 50000, type = int, help = 'maximum number of shapes')
    parser.add_argument('-e', '--epochs', default = 100000, type = int, help = 'number of epochs')
    parser.add_argument('-hd', '--hidden_dim', default = 256, type = int, help = 'hidden dimensions size')
    parser.add_argument('-evp', '--eval_per', default = 10, type = int, help = 'how often to run evaluation')
    parser.add_argument('-v', '--variational', default = "True", type = str, help = 'If running VAE or AE')
    parser.add_argument('-cmd_lw', '--cmd_lw', default = 1., type = float, help = 'Command loss weight')    
    parser.add_argument('-cub_lw', '--cub_lw', default = 1., type = float, help = 'Cube loss weight')
    parser.add_argument('-leaf_lw', '--leaf_lw', default = 1., type = float, help = 'Leaf loss weight')
    parser.add_argument('-axis_lw', '--axis_lw', default = 1., type = float, help = 'Axis loss weight')
    parser.add_argument('-face_lw', '--face_lw', default = 1., type = float, help = 'Face loss weight')
    parser.add_argument('-align_lw', '--align_lw', default = 1. , type = float, help = 'Align loss weight')
    parser.add_argument('-cub_prm_lw', '--cub_prm_lw', default = 50. , type = float , help = 'Cube parameter loss weight')
    parser.add_argument('-att_prm_lw', '--att_prm_lw', default = 50., type = float, help = 'Attach paramter loss weight')
    parser.add_argument('-bb_lw', '--bb_lw', default = 50., type = float, help = 'Bounding Box dimensions loss weight')
    parser.add_argument('-kl_lw', '--kl_lw', default = 0.1, type = float, help = 'KL Loss weight')
    parser.add_argument('-enc_lr', '--enc_lr', default = 0.0001, type = float, help = 'Encoder learning rate')
    parser.add_argument('-dec_lr', '--dec_lr', default = 0.0001, type = float, help = 'Decoder learning rate')
    parser.add_argument('-enc_step', '--enc_step', default = 5000, type = int, help = 'Encoder learning rate steps')
    parser.add_argument('-dec_step', '--dec_step', default = 5000, type = int, help = 'Decoder learning rate steps')
    parser.add_argument('-enc_decay', '--enc_decay', default = 1.0, type = float, help = 'Encoder learning rate decay')
    parser.add_argument('-dec_decay', '--dec_decay', default = 1.0, type = float, help = 'Decoder learning rate decay')
    parser.add_argument('-b', '--batch_size', default = 1, type = int, help = 'Batch Size, non-operational')
    parser.add_argument('-ho', '--holdout_perc', default = .1, type = float, help = 'Train/Val split % if non-given')
    parser.add_argument('-rd', '--rd_seed', default = 42, type = int, help = 'Random seed')
    parser.add_argument('-m', '--mode', default = "train", type = str, help = 'Mode -> options {train/load/eval_gen/eval_recon}')
    parser.add_argument('-le', '--load_epoch', default = None, type = int, help = 'Epoch to load from')
    parser.add_argument('-ng', '--num_gen', default = 50, type = int, help = 'Number of shapes to generate')
    parser.add_argument('-ne', '--num_eval', default = 50, type = int, help = 'Number of shapes to evaluate')
    parser.add_argument('-si', '--shape_inds', default = None, type = str, help = 'Shape inds (used during eval modes)')
    parser.add_argument('-mi', '--keep_missing', default = None, type = str, help = 'Should missing indices be added to train/val split (from given)')
    parser.add_argument('-efs', '--eval_fscores', action='store_true', default=False, help='use this switch to compute f-scores during reconstruction eval')
    parser.add_argument('-prp', '--print_per', default = 2, type = int, help = 'How often to print training results')
    
    args = parser.parse_args()

    args.variational = ast.literal_eval(args.variational)
    
    if args.model_name is None:
        args.model_name = args.exp_name 
        
    if "eval" in args.mode:
        run_generate(args)
        
    else:
        if args.mode == "load":
            load_epoch = args.load_epoch
            args = loadConfigFile(args.exp_name)
            
        else:
            writeConfigFile(args)
            load_epoch = None
                    
        loss_config = getLossConfig(args)
        
        run_train(
            dataset_path=args.dataset_path,
            exp_name=args.exp_name,
            max_shapes=args.max_shapes,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            eval_per=args.eval_per,
            variational=args.variational,
            loss_config=loss_config,
            enc_lr=args.enc_lr,
            dec_lr=args.dec_lr,
            enc_step=args.enc_step,
            dec_step=args.dec_step,
            enc_decay=args.enc_decay,
            dec_decay=args.dec_decay,
            batch_size=args.batch_size,
            holdout_perc=args.holdout_perc,
            rd_seed=args.rd_seed,
            print_per=args.print_per,
            num_gen=args.num_gen,
            num_eval=args.num_eval,
            keep_missing=args.keep_missing,
            category=args.category,
            load_epoch=load_epoch
        )


    
