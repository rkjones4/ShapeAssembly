import torch
import re
import utils
import os
import trimesh as tm
from trimesh.collision import CollisionManager
from trimesh.creation import box
from valid import check_stability, check_rooted
from pointnet_classification import eval_get_var
import sys
from tqdm import tqdm
from ShapeAssembly import hier_execute

MAX_DEPTH = 3
MAX_LEAVES = 12
MIN_LEAVES = 3
BB_THRESH = 1.05
DO_STABLE = False

def simplifyHP(node, depth=1):
    if depth > MAX_DEPTH:
        node.pop('name')
        node.pop('children')
        node.pop('prog')
        return 1.
    
    lc = 0.

    for child in node['children'][1:]:
        if len(child) > 0:
            lc += simplifyHP(child, depth+1)            
        else:
            lc += 1.
            
    return lc


def load_progs(dataset_path):
    inds = os.listdir(dataset_path)
    inds = [i.split('.')[0] for i in inds]
    #to debug
    #inds = ["172"]
    good_inds = []
    progs = []
    for ind in tqdm(inds):
        hp = utils.loadHPFromFile(f'{dataset_path}/{ind}.txt')
        if hp is not None and len(hp) > 0:
            progs.append(hp)
            good_inds.append(ind)
    return good_inds, progs


def main(dataset_path, outdir):
    indices, progs = load_progs(dataset_path)

    os.system(f'mkdir {outdir}')
    os.system(f'mkdir {outdir}/valid')
    os.system(f'mkdir {outdir}/non_valid')

    count = 0
    
    for ind, prog in tqdm(list(zip(indices, progs))):
        count +=1

        if count < 0:
            continue
        
        lc = simplifyHP(prog)                    
        verts, faces = hier_execute(prog)

        bbdims = torch.tensor(
            [float(a) for a in re.split(r'[()]', prog['prog'][0])[1].split(',')[:3]]
        )
        
        bb_viol = (verts.abs().max(dim=0).values / (bbdims / 2)).max()

        try:
            rooted = check_rooted(verts, faces)
        except Exception as e:
            print(f"Failed rooted check with {e}")
            rooted = False
            
        if DO_STABLE:
            stable = check_stability(verts, faces)
        else:
            stable = True
            
        if lc <= MAX_LEAVES and lc >= MIN_LEAVES and bb_viol < BB_THRESH and rooted and stable:
            utils.writeHierProg(prog, f'{outdir}/valid/{ind}.txt')
            utils.writeObj(verts, faces, f'{outdir}/valid/{ind}.obj')
        else:
            utils.writeHierProg(prog, f'{outdir}/non_valid/{ind}.txt')
            utils.writeObj(verts, faces, f'{outdir}/non_valid/{ind}.obj')                    

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
    
