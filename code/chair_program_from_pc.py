import sys
import os
import torch
from model_prog import Sampler, MLP, FDGRU, ENCGRU, run_eval_decoder
from ShapeAssembly import hier_execute
import utils
from tqdm import tqdm
from pc_encoder import PCEncoder
import losses

shapeAssembly_decoder = "pc_data/sa_pc_dec.pt"
point_cloud_encoder = "pc_data/sa_pc_enc.pt"
point_cloud_folder = "pc_data/samples/"

device = torch.device("cuda")
fscore = losses.FScore(device)

def getInds(train_ind_file):
    inds = set()
    with open(train_ind_file) as f:
        for line in f:
            inds.add(line.strip())
    return inds

def eval_recon(outdir, data_inds):
    decoder = torch.load(shapeAssembly_decoder).to(device)
    decoder.eval()
    encoder = PCEncoder()
    encoder.load_state_dict(torch.load(point_cloud_encoder))
    encoder.eval()
    encoder.to(device)

    os.system(f'mkdir {outdir}')

    count = 0.
    tdist = 0.
    
    for ind in tqdm(data_inds):

        pc_samp = torch.load(f'{point_cloud_folder}/{ind}.pts').to(device)
        enc = encoder(pc_samp.unsqueeze(0))                        
        prog, _ = run_eval_decoder(enc.unsqueeze(0), decoder, False)
        verts, faces = hier_execute(prog)

        utils.writeObj(verts, faces, f"{outdir}/{ind}.obj")
        utils.writeHierProg(prog, f"{outdir}/{ind}.txt")
        
        verts = verts.to(device)
        faces = faces.to(device)        
        
        pred_samp = utils.sample_surface(
            faces,
            verts.unsqueeze(0),
            10000,
            True
        )

        # Center PC 
        
        offset = (pc_samp.max(dim=0).values + pc_samp.min(dim=0).values) / 2
        pc_samp -= offset
        
        #utils.writeSPC(pc_samp,f'tar_pc_{ind}.obj')
        #utils.writeSPC(pred_samp[0,:,:3],f'scripts/output/pred_pc_{ind}.obj')

        pc_samp = pc_samp.repeat(1,2).unsqueeze(0)
        tdist += fscore.score(
            pred_samp.squeeze().T.unsqueeze(0),
            pc_samp.squeeze().T.unsqueeze(0)
        )
        count += 1

    print(f"Average F-score: {tdist/count}")
        
    

        
if __name__ == '__main__':
    with torch.no_grad():
        eval_recon(sys.argv[1],sys.argv[2:])
