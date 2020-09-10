import torch
import utils
from intersect import sampleCorners

DIM = 20

a = (torch.arange(DIM).float()/(DIM-1))
b = a.unsqueeze(0).unsqueeze(0).repeat(DIM, DIM, 1)
c = a.unsqueeze(0).unsqueeze(2).repeat(DIM, 1, DIM)
d = a.unsqueeze(1).unsqueeze(2).repeat(1, DIM, DIM)
g = torch.stack((b,c,d), dim=3).view(-1, 3)
device = torch.device("cuda")
xyz = g.unsqueeze(0).to(device)

def voxelize(cubes, bbox):
    scene_geom = torch.cat((
        bbox['xd'].unsqueeze(0),
        bbox['yd'].unsqueeze(0),
        bbox['zd'].unsqueeze(0),
        bbox['center'],            
        bbox['xdir'],
        bbox['ydir']
    ))

    s_r = torch.cat(
            (
                (scene_geom[6:9] / (scene_geom[6:9].norm(dim=0).unsqueeze(0) + 1e-8)).unsqueeze(1),
                (scene_geom[9:12] / (scene_geom[9:12].norm(dim=0).unsqueeze(0) + 1e-8)).unsqueeze(1),
                torch.cross(
                    scene_geom[6:9] / (scene_geom[6:9].norm(dim=0).unsqueeze(0) + 1e-8),
                    scene_geom[9:12] / (scene_geom[9:12].norm(dim=0).unsqueeze(0) + 1e-8)                
                ).unsqueeze(1)
            ), dim = 1)

    voxels = ((s_r @ (((xyz - .5) * scene_geom[:3]).unsqueeze(-1))).squeeze() + scene_geom[3:6]).squeeze()

    offset = (voxels.max(dim=0).values - voxels.min(dim=0).values)/ (DIM * 2.)

    corners = sampleCorners(cubes).view(-1,8,3)

    occs = []
        
    for i in range(corners.shape[0]):
        cube_occ = torch.zeros(voxels.shape[0]).to(device)
        o = corners[i][0]
        a = corners[i][4]
        b = corners[i][2]
        c = corners[i][1]

        oa = a - o
        ob = b - o
        oc = c - o

        for x in [-1, -.5, 0, .5, 1]:
            for y in [-1, -.5, 0, .5, 1]:
                for z in [-1, -.5, 0, .5, 1]:
        
                    occ = torch.zeros(voxels.shape[0]).to(device)
        
                    vec = torch.stack((oa, ob, oc)).T

                    ovoxels = voxels - (torch.tensor([x, y, z]).to(device) * offset).unsqueeze(0)
                    
                    res = ((ovoxels - o) @ vec).T
        
                    occ[(res[0,:] >= 0).nonzero().squeeze()] += 1.
                    occ[(res[0,:] <= oa.dot(oa)).nonzero().squeeze()] += 1.
                    occ[(res[1,:] >= 0).nonzero().squeeze()] += 1.
                    occ[(res[1,:] <= ob.dot(ob)).nonzero().squeeze()] += 1.
                    occ[(res[2,:] >= 0).nonzero().squeeze()] += 1.
                    occ[(res[2,:] <= oc.dot(oc)).nonzero().squeeze()] += 1.

                    cube_occ[(occ == 6.).nonzero().squeeze()] = 1.
                    
        occs.append(cube_occ)
        
    return occs


def shape_voxelize(cubes, bbox):
    scene_geom = torch.cat((
        bbox['xd'].unsqueeze(0),
        bbox['yd'].unsqueeze(0),
        bbox['zd'].unsqueeze(0),
        bbox['center'],            
        bbox['xdir'],
        bbox['ydir']
    ))

    s_r = torch.cat(
            (
                (scene_geom[6:9] / (scene_geom[6:9].norm(dim=0).unsqueeze(0) + 1e-8)).unsqueeze(1),
                (scene_geom[9:12] / (scene_geom[9:12].norm(dim=0).unsqueeze(0) + 1e-8)).unsqueeze(1),
                torch.cross(
                    scene_geom[6:9] / (scene_geom[6:9].norm(dim=0).unsqueeze(0) + 1e-8),
                    scene_geom[9:12] / (scene_geom[9:12].norm(dim=0).unsqueeze(0) + 1e-8)                
                ).unsqueeze(1)
            ), dim = 1)

    voxels = ((s_r @ (((xyz - .5) * scene_geom[:3]).unsqueeze(-1))).squeeze() + scene_geom[3:6]).squeeze()

    offset = (voxels.max(dim=0).values - voxels.min(dim=0).values)/ (DIM * 2.)

    corners = sampleCorners(cubes).view(-1,8,3)

    socc = torch.zeros(voxels.shape[0]).to(device)
    
    for i in range(corners.shape[0]):
        
        o = corners[i][0]
        a = corners[i][4]
        b = corners[i][2]
        c = corners[i][1]

        oa = a - o
        ob = b - o
        oc = c - o

        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    occ = torch.zeros(voxels.shape[0]).to(device)
                    vec = torch.stack((oa, ob, oc)).T

                    ovoxels = voxels - (torch.tensor([x, y, z]).to(device) * offset).unsqueeze(0)
                    
                    res = ((ovoxels - o) @ vec).T
        
                    occ[(res[0,:] >= 0).nonzero().squeeze()] += 1.
                    occ[(res[0,:] <= oa.dot(oa)).nonzero().squeeze()] += 1.
                    occ[(res[1,:] >= 0).nonzero().squeeze()] += 1.
                    occ[(res[1,:] <= ob.dot(ob)).nonzero().squeeze()] += 1.
                    occ[(res[2,:] >= 0).nonzero().squeeze()] += 1.
                    occ[(res[2,:] <= oc.dot(oc)).nonzero().squeeze()] += 1.

                    socc[(occ == 6.).nonzero().squeeze()] = 1.
                    
    return socc
