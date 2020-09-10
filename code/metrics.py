from ShapeAssembly import hier_execute
import utils
from voxelize import voxelize, shape_voxelize
import torch
import os
import sys
import ShapeAssembly as ex
import trimesh as tm
from trimesh.collision import CollisionManager
from trimesh.creation import box
from valid import check_stability, check_rooted
from pointnet_classification import eval_get_var
from tqdm import tqdm
import losses

device = torch.device("cuda")
fscore = losses.FScore(device)
NUM_SAMPS = 10000

def CuboidToParams(c):
    p = {}
    p['xd'] = c.dims[0].to(device)
    p['yd'] = c.dims[1].to(device)
    p['zd'] = c.dims[2].to(device)
    p['center'] = c.pos.to(device)
    p['xdir'] = c.rfnorm.to(device)
    p['ydir'] = c.tfnorm.to(device)
    p['zdir'] = c.ffnorm.to(device)
    return p

def getFScore(verts, faces, gt_verts, gt_faces):
    p_samps = utils.sample_surface(faces, verts.unsqueeze(0), NUM_SAMPS)
    t_samps = utils.sample_surface(gt_faces, gt_verts.unsqueeze(0), NUM_SAMPS)

    return fscore.score(
        p_samps.squeeze().T.unsqueeze(0),
        t_samps.squeeze().T.unsqueeze(0)
    )

def getShapeIoU(cubes, gt_cubes, bbox):
    flat_cubes = []
    flat_gt_cubes = []

    for col in cubes:
        flat_cubes += col

    for col in gt_cubes:
        flat_gt_cubes  += col
    
    pvoxels = shape_voxelize(flat_cubes, bbox)
    tvoxels = shape_voxelize(flat_gt_cubes, bbox)
    pinter = ((tvoxels + pvoxels) == 2.).nonzero().shape[0]
    punion = ((tvoxels + pvoxels) > 0.).nonzero().shape[0]
    
    return (pinter * 100.0) / punion


def getTotalVol(gt_cubes):
    volume = 0.
    for col in gt_cubes:
        for c in col:
            volume += c['xd'] * c['yd'] * c['zd']
    return volume

def getDefCube():
    def_cube = {}
    def_cube['xd'] = torch.tensor(0.).to(device)
    def_cube['yd'] = torch.tensor(0.).to(device)
    def_cube['zd'] = torch.tensor(0.).to(device)
    def_cube['center'] = torch.tensor([0.0,0.0,0.0]).to(device)
    def_cube['xdir'] = torch.tensor([0.0,0.0,0.0]).to(device)
    def_cube['ydir'] = torch.tensor([0.0,0.0,0.0]).to(device)
    def_cube['zdir'] = torch.tensor([0.0,0.0,0.0]).to(device)
    return def_cube

def param_dist(cube1, cube2, bbox):
    ref = max([bbox['xd'], bbox['yd'], bbox['zd']])
    c_diff = (cube1['center'] - cube2['center']).norm() / (ref + 1e-8)
    xd_diff = (cube1['xd'] - cube2['xd']).abs() / (cube1['xd'] + 1e-8)
    yd_diff = (cube1['yd'] - cube2['yd']).abs() / (cube1['yd'] + 1e-8)
    zd_diff = (cube1['zd'] - cube2['zd']).abs() / (cube1['zd'] + 1e-8)
    xdir_diff = .5 - (cube1['xdir'].dot(cube2['xdir']) / 2)
    ydir_diff = .5 - (cube1['ydir'].dot(cube2['ydir']) / 2)
    zdir_diff = .5 - (cube1['zdir'].dot(cube2['zdir']) / 2)
    return c_diff + xd_diff + yd_diff + zd_diff + xdir_diff + ydir_diff + zdir_diff
    
def getParamDist(cubes, gt_cubes, bbox):
    tv = getTotalVol(gt_cubes)

    metric = 0.

    def_cube = getDefCube()
    
    for i, gt_col in enumerate(gt_cubes):
        col = None
        if i < len(cubes):
            col = cubes[i]
        
        for j in range(len(gt_col)):            
            if col is not None and j < len(col):
                pd = param_dist(gt_col[j], col[j], bbox)                
            else:
                pd = param_dist(gt_col[j], def_cube, bbox)

            vol = gt_col[j]['xd'] * gt_col[j]['yd'] * gt_col[j]['zd']

            metric += pd * (vol/tv)
            
    return metric
    
def getBBox(gt_prog):
    P = ex.Program()
    bbox_line = P.parseCuboid(gt_prog['prog'][0])
    bbox = {}
    bbox['center'] = torch.tensor([0.0,0.0,0.0]).to(device)
    bbox['xdir'] = torch.tensor([1.0,0.0,0.0]).to(device)
    bbox['ydir'] = torch.tensor([0.0,1.0,0.0]).to(device)
    bbox['zdir'] = torch.tensor([0.0,0.0,1.0]).to(device)
    bbox['xd'] = bbox_line[1].to(device)
    bbox['yd'] = bbox_line[2].to(device)
    bbox['zd'] = bbox_line[3].to(device)
    return bbox
    

def recon_metrics(recon_sets, outpath, exp_name, name, epoch, VERBOSE):
    misses = 0.
    results = {
        'fscores': [],
        'iou_shape': [],
        'param_dist_parts': [],
    }
        
    for prog, gt_prog, prog_ind in recon_sets:    

        bbox = getBBox(gt_prog)
        
        gt_verts, gt_faces, gt_hscene = hier_execute(gt_prog, return_all = True)

        gt_cubes = [[CuboidToParams(c) for c in scene] for scene in gt_hscene]
        
        try:
            verts, faces, hscene = hier_execute(prog, return_all = True)
            cubes = [[CuboidToParams(c) for c in scene] for scene in hscene]
            
            assert not torch.isnan(verts).any(), 'saw nan vert'

        except Exception as e:
            misses += 1.
            if VERBOSE:
                print(f"failed recon metrics for {prog_ind} with {e}")
            continue

        verts = verts.to(device)
        gt_verts = gt_verts.to(device)
        faces = faces.to(device)
        gt_faces = gt_faces.to(device)
        
        gt_objs = os.listdir(f"{outpath}/{exp_name}/objs/gt/")
        
        if f"{prog_ind}.obj" not in gt_objs:
            utils.writeObj(gt_verts, gt_faces, f"{outpath}/{exp_name}/objs/gt/{prog_ind}.obj")
            utils.writeHierProg(gt_prog, f"{outpath}/{exp_name}/programs/gt/{prog_ind}.txt")

        try:
            utils.writeObj(
                verts, faces, f"{outpath}/{exp_name}/objs/{name}/{epoch}_{prog_ind}.obj"
            )
            utils.writeHierProg(
                prog, f"{outpath}/{exp_name}/programs/{name}/{epoch}_{prog_ind}.txt"
            )
            
        except Exception as e:
            print(f"Failed writing prog/obj for {prog_ind} with {e}")

        try:
            fs = getFScore(verts, faces, gt_verts, gt_faces)
            if fs is not None:
                results['fscores'].append(fs)
        except Exception as e:
            if VERBOSE:
                print(f"failed Fscore for {prog_ind} with {e}")                

        try:
            siou = getShapeIoU(cubes, gt_cubes, bbox)
            if siou is not None:
                results['iou_shape'].append(siou)
        except Exception as e:
            if VERBOSE:
                print(f"failed Shape Iou for {prog_ind} with {e}")
        

        try:
            pd = getParamDist(cubes, gt_cubes, bbox)
            if pd is not None:
                results['param_dist_parts'].append(pd)
        except Exception as e:
            if VERBOSE:
                print(f"failed param dist for {prog_ind} with {e}")


    for key in results:
        if len(results[key]) > 0:
            res = torch.tensor(results[key]).mean().item()
        else:
            res = 0.

        results[key] = res
        
    return results, misses
        
def gen_metrics(gen_progs, outpath, exp_name, epoch, VERBOSE, write_progs = True):
    misses = 0.
    results = {
        'num_parts': [],        
        'rootedness': [],
        'stability': [],                
    }

    samples = []
    
    for i, prog in enumerate(gen_progs):
        try:
            verts, faces = hier_execute(prog)            
            assert not torch.isnan(verts).any(), 'saw nan vert'
            if write_progs:
                utils.writeObj(verts, faces, f"{outpath}/{exp_name}/objs/gen/{epoch}_{i}.obj")
                utils.writeHierProg(prog, f"{outpath}/{exp_name}/programs/gen/{epoch}_{i}.txt")

            results['num_parts'].append(verts.shape[0] / 8.0)
            samples.append((verts, faces))
            
        except Exception as e:
            misses += 1.
            if VERBOSE:
                print(f"failed gen metrics for {i} with {e}")
            continue

        try:
            if check_rooted(verts, faces):
                results['rootedness'].append(1.)
            else:
                results['rootedness'].append(0.)

            if check_stability(verts, faces):
                results['stability'].append(1.)
            else:
                results['stability'].append(0.)

        except Exception as e:
            if VERBOSE:
                print(f"failed rooted/stable with {e}")
                
    for key in results:
        if len(results[key]) > 0:
            res = torch.tensor(results[key]).mean().item()
        else:
            res = 0.

        results[key] = res

    try:
        results['variance'] = eval_get_var(samples)
    except Exception as e:
        results['variance'] = 0.
        if VERBOSE:
            print(f"failed getting variance with {e}")
        
    return results, misses


def calc_tab3():
    ddir = sys.argv[1]
    inds = os.listdir(ddir)
    outs = []

    if len(sys.argv) > 2:
        inds = inds[:int(sys.argv[2])]
    
    for ind in tqdm(inds):
        if '.txt' in ind:
            hp = utils.loadHPFromFile(f'{ddir}/{ind}')
            verts, faces = hier_execute(hp)            
        else:
            verts, faces = utils.loadObj(f'{ddir}/{ind}')
            verts = torch.tensor(verts)
            faces = torch.tensor(faces)
        outs.append((verts, faces))
        
    misses = 0.
    results = {
        'num_parts': [],        
        'rootedness': [],
        'stability': [],                
    }

    samples = []
    
    for (verts, faces) in tqdm(outs):        
        
        results['num_parts'].append(verts.shape[0] / 8.0)
        samples.append((verts, faces))
            

        if check_rooted(verts, faces):
            results['rootedness'].append(1.)
        else:
            results['rootedness'].append(0.)

        if check_stability(verts, faces):
            results['stability'].append(1.)
        else:
            results['stability'].append(0.)
                
    for key in results:
        if len(results[key]) > 0:
            res = torch.tensor(results[key]).mean().item()
        else:
            res = 0.

        results[key] = res

    
    results['variance'] = eval_get_var(samples)

    for key in results:
        print(f"Result {key} : {results[key]}")


if __name__ == '__main__':
    with torch.no_grad():
        calc_tab3()
    #inds = os.listdir(sys.argv[1])
    #for ind in inds:
    #hp = utils.loadHPFromFile(f'{sys.argv[1]}')
    #verts, faces = hier_execute(hp)
    #verts, faces = utils.loadObj(f'{sys.argv[1]}')
    #verts = torch.tensor(verts)
    #faces = torch.tensor(faces)
    #utils.writeObj(verts, faces, 'test.obj')
    #print(check_rooted(verts, faces))
    #print(check_stability(verts, faces, True))

