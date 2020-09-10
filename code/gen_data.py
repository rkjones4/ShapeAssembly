import torch
import re
import numpy as np
import math
import random
import utils
import os
import sys
from ShapeAssembly import hier_execute, Cuboid, Program
import json_parse as jp
import generate as gen
from voxelize import voxelize
import losses 

device = torch.device("cuda")
fscore = losses.FScore(device)

def propsToProgram(gt):
    T = Program()
    for i, a in enumerate(gt):
        c = Cuboid(str(i))
        
        c.dims = torch.stack([
            a['xd'],
            a['yd'],
            a['zd'],
        ])
        c.pos = a['center']
        c.rfnorm = a['xdir']
        c.tfnorm = a['ydir']
        c.ffnorm = a['zdir']
        
        T.cuboids[str(i)] = c

    return T


def CuboidToParams(c):
    p = {}
    p['xd'] = c.dims[0]
    p['yd'] = c.dims[1]
    p['zd'] = c.dims[2]
    p['center'] = c.pos
    p['xdir'] = c.rfnorm
    p['ydir'] = c.tfnorm
    p['zdir'] = c.ffnorm
    return p

def TensorToParams(t):
    p = {}
    p['xd'] = t[0]
    p['yd'] = t[1]
    p['zd'] = t[2]
    p['center'] = t[3:6]
    p['xdir'] = t[6:9]
    p['ydir'] = t[9:12]
    p['zdir'] = t[12:15]
    return p

def get_gt_geom(hp, return_cubes):
    cubes = []
    queue = [hp]
    while (len(queue) > 0):
        node = queue.pop(0)
        for i in range(1, len(node['children'])):
            if node['children'][i] is None or len(node['children'][i]) == 0:
                cubes.append(node['cubes'][i])
            else:
                queue.append(node['children'][i])

    P = propsToProgram(cubes)
    if return_cubes:
        return [c.getParams() for c in list(P.cuboids.values())[1:]]
    return P.getShapeGeo()
    
if __name__ == '__main__':

    CATEGORY = sys.argv[1]
    data_path = f"/home/{os.environ.get('USER')}/pnhier/{CATEGORY}_hier/"
    outdir = f"parse_{CATEGORY}"
    
    inds = os.listdir(data_path)
    inds = [i.split('.')[0] for i in inds]
            
    count = 0.
    errors = 0.        
    gpfsv = 0.    
    gpcfsv = 0.    
    
    os.system(f'mkdir {outdir}')
    
    for ind in inds:        
        count += 1.
        
        try:
            hier = jp.parseJsonToHier(ind, CATEGORY)
            nshier = jp.parseJsonToHier(ind, CATEGORY, True)
            
            gen.generate_program(hier)
                                                
            pverts, pfaces = hier_execute(hier)
            tverts, tfaces = get_gt_geom(nshier, False)    

            tsamps = utils.sample_surface(tfaces, tverts.unsqueeze(0), 10000)
                
            try:

                psamps = utils.sample_surface(pfaces, pverts.unsqueeze(0), 10000)
            
                pfs = fscore.score(
                    psamps.squeeze().T.unsqueeze(0),
                    tsamps.squeeze().T.unsqueeze(0)
                )
                
            except Exception:
                pfs = 0.
                        
            if pfs >= 50:
                gpfsv +=1.

            if pfs >= 75:                
                gpcfsv += 1.                                        
                utils.writeHierProg(hier, f"{outdir}/{ind}.txt")            
                utils.writeObj(tverts, tfaces, f"{outdir}/{ind}_target.obj")
                utils.writeObj(pverts, pfaces, f"{outdir}/{ind}_parse.obj")
                        
            
            print(f"CAT {CATEGORY}, P: {ind}, C: {count}| Greedy | Parse -- FS {round(gpfsv/count, 3)}, HFS {round(gpcfsv/count, 3)} | Errors {errors/count}")
                                    

        except Exception as e:
            if str(e) != 'disconnected graph':
                errors += 1.
            print(f"Prog {ind} -> ERROR {e}")

