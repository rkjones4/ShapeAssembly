import utils
import torch
import itertools
import trimesh
import scipy
import numpy as np
import faiss

DOING_PARSE = True

DIM = 20
ATT_DIM = 50

device = torch.device("cuda")

resource = faiss.StandardGpuResources()

def robust_norm(var, dim=2):
    return ((var ** 2).sum(dim=dim) + 1e-8).sqrt()

class collisionEngine():
    def __init__(self, device):
        self.dimension = 3
        self.k = 1
        self.device = device

    def search_nn(self, index, query, k):
        _, I = index.search(query, k)

        I_var = torch.from_numpy(np.ascontiguousarray(I).astype(np.int64))

        I_var = I_var.to(self.device)

        return I_var

    def findPCInter(self, predict_pc, gt_pc, index_predict, index_gt, thresh):

        predict_pc = predict_pc.to(self.device)
        gt_pc = gt_pc.to(self.device)
        
        predict_pc_size = predict_pc.size()
        gt_pc_size = gt_pc.size()
        
        predict_pc_np = np.ascontiguousarray(
            torch.transpose(predict_pc.data.clone(), 1, 2).cpu().numpy()
        )  # BxMx3
        
        gt_pc_np = np.ascontiguousarray(
            torch.transpose(gt_pc.data.clone(), 1, 2).cpu().numpy()
        )  # BxNx3

        # selected_gt: Bxkx3xM
        selected_gt_by_predict = torch.FloatTensor(
            predict_pc_size[0], self.k, predict_pc_size[1], predict_pc_size[2]
        )
        # selected_predict: Bxkx3xN
        selected_predict_by_gt = torch.FloatTensor(
            gt_pc_size[0], self.k, gt_pc_size[1], gt_pc_size[2]
        )
        

        selected_gt_by_predict = selected_gt_by_predict.to(self.device)
        selected_predict_by_gt = selected_predict_by_gt.to(self.device)
            
        # process each batch independently.
        for i in range(predict_pc_np.shape[0]):
            # database is gt_pc, predict_pc -> gt_pc -----------------------------------------------------------
            I_var = self.search_nn(index_gt, predict_pc_np[i], self.k)

            # process nearest k neighbors
            for k in range(self.k):
                selected_gt_by_predict[i, k, ...] = gt_pc[i].index_select(
                    1, I_var[:, k]
                )
                

            # database is predict_pc, gt_pc -> predict_pc -------------------------------------------------------
            I_var = self.search_nn(index_predict, gt_pc_np[i], self.k)

            # process nearest k neighbors
            for k in range(self.k):
                selected_predict_by_gt[i, k, ...] = predict_pc[i].index_select(
                    1, I_var[:, k]
                )                


        # compute loss ===================================================
        # selected_gt(Bxkx3xM) vs predict_pc(Bx3xM)
        r_to_gt = predict_pc.unsqueeze(1).expand_as(selected_gt_by_predict) - selected_gt_by_predict 
        gt_to_r = selected_predict_by_gt - gt_pc.unsqueeze(1).expand_as(selected_predict_by_gt) 

        r_to_gt = robust_norm(
            selected_gt_by_predict
            - predict_pc.unsqueeze(1).expand_as(selected_gt_by_predict),
            dim=2
        )

        gt_to_r = robust_norm(
            selected_predict_by_gt
            - gt_pc.unsqueeze(1).expand_as(selected_predict_by_gt),
            dim=2
        )

        r_to_gt = r_to_gt.flatten()
        gt_to_r = gt_to_r.flatten()

        return (r_to_gt < thresh).nonzero().squeeze(), (gt_to_r < thresh).nonzero().squeeze()

if DOING_PARSE:

    a = (torch.arange(DIM).float()/(DIM-1))
    b = a.unsqueeze(0).unsqueeze(0).repeat(DIM, DIM, 1)
    c = a.unsqueeze(0).unsqueeze(2).repeat(DIM, 1, DIM)
    d = a.unsqueeze(1).unsqueeze(2).repeat(1, DIM, DIM)
    g = torch.stack((b,c,d), dim=3).view(-1, 3).to(device)

    fr = (g[:,0] == 1.).nonzero().squeeze().to(device)
    fl = (g[:,0] == 0.).nonzero().squeeze().to(device)
    ft = (g[:,1] == 1.).nonzero().squeeze().to(device)
    fbo = (g[:,1] == 0.).nonzero().squeeze().to(device)
    ff = (g[:,2] == 1.).nonzero().squeeze().to(device)
    fba = (g[:,2] == 0.).nonzero().squeeze().to(device)
    
    bb_mask = torch.ones((DIM**3), 3).float().to(device) * 100    
    bb_mask[torch.cat((ft, fbo)), :] *= 0.

    top_mask = torch.ones((DIM**3), 3).float().to(device) * 100
    bot_mask = torch.ones((DIM**3), 3).float().to(device) * 100    
    top_mask[ft] *= 0.
    bot_mask[fbo] *= 0.
    
    s_xyz = g.unsqueeze(0).to(device)

    c_xyz = torch.tensor([
        [0,0,0],
        [0,0,1],
        [0,1,0],
        [0,1,1],
        [1,0,0],
        [1,0,1],
        [1,1,0],
        [1,1,1],
    ]).unsqueeze(0).float()
    
    colEngine = collisionEngine(device)

    atta = (torch.arange(ATT_DIM).float()/(ATT_DIM-1))
    attb = atta.unsqueeze(0).unsqueeze(0).repeat(ATT_DIM, ATT_DIM, 1)
    attc = atta.unsqueeze(0).unsqueeze(2).repeat(ATT_DIM, 1, ATT_DIM)
    attd = atta.unsqueeze(1).unsqueeze(2).repeat(1, ATT_DIM, ATT_DIM)
    attg = torch.stack((attb,attc,attd), dim=3).view(-1, 3)
    
    attxyz = attg.unsqueeze(0).to(device)

    
    
def smp_pt(geom, pt):
    xdir = geom[6:9] / (geom[6:9].norm() + 1e-8)
    ydir = geom[9:12] / (geom[9:12].norm() + 1e-8)
    zdir = torch.cross(xdir, ydir)
    r = torch.stack((
        xdir,
        ydir,
        zdir
    )).T
    return (r @ ((pt -.5) * geom[:3])) + geom[3:6]

def findHiddenCubes(cubes):
    cube_geom = []
    for c in cubes:
        cube_geom.append(torch.cat((
            c['xd'].unsqueeze(0),
            c['yd'].unsqueeze(0),
            c['zd'].unsqueeze(0),
            c['center'],            
            c['xdir'],
            c['ydir']
        )))

    scene_geom = torch.stack([c for c in cube_geom]).to(device)
    scene_corners = sampleCorners(cubes).view(-1,8,3)
    
    bad_inds = [] 

    for ind in range(len(cubes)):        
        points = scene_corners[ind,:,:]

        all_inside = False
        
        for nind in range(len(cubes)):
            if nind == ind:
                continue
            if all_inside:
                break

            O = smp_pt(scene_geom[nind], torch.zeros(3).to(device))
            A = torch.stack([
                scene_geom[nind][0] * scene_geom[nind][6:9],
                scene_geom[nind][1] * scene_geom[nind][9:12],
                scene_geom[nind][2] * torch.cross(
                    scene_geom[nind][6:9], scene_geom[nind][9:12]
                )
            ]).T
            B = (points.T - O.unsqueeze(1)).cpu()
            p = torch.tensor(np.linalg.solve(A.cpu(), B.cpu())).T

            if p.min() >= -0.01 and p.max() <= 1.01:
                all_inside = True

        if all_inside:
            bad_inds.append(ind)

    return bad_inds


def findOverlapCubes(cubes, CTHRESH):
    cube_geom = []

    for c in cubes:
        cube_geom.append(torch.cat((
            c['xd'].unsqueeze(0),
            c['yd'].unsqueeze(0),
            c['zd'].unsqueeze(0),
            c['center'],            
            c['xdir'],
            c['ydir']
        )))

    scene_geom = torch.stack([c for c in cube_geom]).to(device)
    scene_corners = sampleCube(cubes).view(-1,DIM**3,3)
    
    bad_inds = set()

    for ind in range(len(cubes)):        
        points = scene_corners[ind,:,:]

        covered = False
        
        for nind in range(len(cubes)):
            if nind == ind or nind in bad_inds:
                continue
            if covered:
                break

            O = smp_pt(scene_geom[nind], torch.zeros(3).to(device))
            A = torch.stack([
                scene_geom[nind][0] * scene_geom[nind][6:9],
                scene_geom[nind][1] * scene_geom[nind][9:12],
                scene_geom[nind][2] * torch.cross(
                    scene_geom[nind][6:9], scene_geom[nind][9:12]
                )
            ]).T
            B = (points.T - O.unsqueeze(1)).cpu()
            p = torch.tensor(np.linalg.solve(A.cpu(), B.cpu())).T
            
            num_outside_s = (p - torch.clamp(p, 0.0, 1.0)).abs().sum(dim=1).nonzero().squeeze().shape

            
            
            if len(num_outside_s) == 0 or (((DIM**3 - num_outside_s[0])*1.) / DIM**3) > CTHRESH:
                covered = True

        if covered:
            bad_inds.add(ind)

    return bad_inds


def samplePC(cubes, flip_bbox=False, split_bbox=False):
    cube_geom = []
    for c in cubes:
        cube_geom.append(torch.cat((
            c['xd'].unsqueeze(0),
            c['yd'].unsqueeze(0),
            c['zd'].unsqueeze(0),
            c['center'],            
            c['xdir'],
            c['ydir']
        )))
        
    scene_geom = torch.stack([c for c in cube_geom]).to(device)
    ind_to_pc = {}
    
    for i in range(0, scene_geom.shape[0]):
        xyz = s_xyz
            
        s_inds = (torch.ones(1,xyz.shape[1]) * i).long().to(device)
        
        s_r = torch.cat(
            (
                (scene_geom[s_inds][:, :, 6:9] / (scene_geom[s_inds][:, :, 6:9].norm(dim=2).unsqueeze(2) + 1e-8)).unsqueeze(3),
                (scene_geom[s_inds][:, :, 9:12] / (scene_geom[s_inds][:, :, 9:12].norm(dim=2).unsqueeze(2) + 1e-8)).unsqueeze(3),
                torch.cross(
                    scene_geom[s_inds][:, :, 6:9] / (scene_geom[s_inds][:, :, 6:9].norm(dim=2).unsqueeze(2) + 1e-8),
                    scene_geom[s_inds][:, :, 9:12] / (scene_geom[s_inds][:, :, 9:12].norm(dim=2).unsqueeze(2) + 1e-8)                
                ).unsqueeze(3)
            ), dim = 3)
    
        s_out = ((s_r @ (((xyz - .5) * scene_geom[s_inds][:, :, :3]).unsqueeze(-1))).squeeze() + scene_geom[s_inds][:, :, 3:6]).squeeze()
        ind_to_pc[i] = s_out

    if flip_bbox:
        ind_to_pc[0] += bb_mask
        temp = ind_to_pc[0].clone()
        ind_to_pc[0][ft] = temp[fbo]
        ind_to_pc[0][fbo] = temp[ft]

    if split_bbox:
        bbox_pc = ind_to_pc.pop(0)              
        ind_to_pc[-2] = bbox_pc.clone() + bot_mask
        ind_to_pc[-1] = bbox_pc.clone() + top_mask

    res = {}
    for key in ind_to_pc:
        index_cpu = faiss.IndexFlatL2(3)
        
        index = faiss.index_cpu_to_gpu(
            resource,
            torch.cuda.current_device(),
            index_cpu
        )
        index.add(
            np.ascontiguousarray(ind_to_pc[key].cpu().numpy())
        )
        res[key] = (ind_to_pc[key], index)
        
    return res, scene_geom

def sampleCorners(cubes):
    
    cube_geom = []
    for c in cubes:
        cube_geom.append(torch.cat((
            c['xd'].unsqueeze(0),
            c['yd'].unsqueeze(0),
            c['zd'].unsqueeze(0),
            c['center'],            
            c['xdir'],
            c['ydir']
        )))
        
    scene_geom = torch.stack([c for c in cube_geom]).to(device)

    s_inds = torch.arange(scene_geom.shape[0]).unsqueeze(1).repeat(1,8).view(1, -1).to(device)
    
    xyz = c_xyz.repeat(1,scene_geom.shape[0],1).to(device)
            
    s_r = torch.cat(
        (
            (scene_geom[s_inds][:, :, 6:9] / (scene_geom[s_inds][:, :, 6:9].norm(dim=2).unsqueeze(2) + 1e-8)).unsqueeze(3),
            (scene_geom[s_inds][:, :, 9:12] / (scene_geom[s_inds][:, :, 9:12].norm(dim=2).unsqueeze(2) + 1e-8)).unsqueeze(3),
            torch.cross(
                scene_geom[s_inds][:, :, 6:9] / (scene_geom[s_inds][:, :, 6:9].norm(dim=2).unsqueeze(2) + 1e-8),
                scene_geom[s_inds][:, :, 9:12] / (scene_geom[s_inds][:, :, 9:12].norm(dim=2).unsqueeze(2) + 1e-8)                
            ).unsqueeze(3)
        ), dim = 3)
    
    s_out = ((s_r @ (((xyz - .5) * scene_geom[s_inds][:, :, :3]).unsqueeze(-1))).squeeze() + scene_geom[s_inds][:, :, 3:6]).squeeze()

    return s_out

def sampleCube(cubes):
    
    cube_geom = []
    for c in cubes:
        cube_geom.append(torch.cat((
            c['xd'].unsqueeze(0),
            c['yd'].unsqueeze(0),
            c['zd'].unsqueeze(0),
            c['center'],            
            c['xdir'],
            c['ydir']
        )))
        
    scene_geom = torch.stack([c for c in cube_geom]).to(device)

    s_inds = torch.arange(scene_geom.shape[0]).unsqueeze(1).repeat(1,DIM**3).view(1, -1).to(device)
    
    xyz = s_xyz.repeat(1,scene_geom.shape[0],1).to(device)
            
    s_r = torch.cat(
        (
            (scene_geom[s_inds][:, :, 6:9] / (scene_geom[s_inds][:, :, 6:9].norm(dim=2).unsqueeze(2) + 1e-8)).unsqueeze(3),
            (scene_geom[s_inds][:, :, 9:12] / (scene_geom[s_inds][:, :, 9:12].norm(dim=2).unsqueeze(2) + 1e-8)).unsqueeze(3),
            torch.cross(
                scene_geom[s_inds][:, :, 6:9] / (scene_geom[s_inds][:, :, 6:9].norm(dim=2).unsqueeze(2) + 1e-8),
                scene_geom[s_inds][:, :, 9:12] / (scene_geom[s_inds][:, :, 9:12].norm(dim=2).unsqueeze(2) + 1e-8)                
            ).unsqueeze(3)
        ), dim = 3)
    
    s_out = ((s_r @ (((xyz - .5) * scene_geom[s_inds][:, :, :3]).unsqueeze(-1))).squeeze() + scene_geom[s_inds][:, :, 3:6]).squeeze()

    return s_out


def findInters(ind_to_pc, scene_geom, ind_pairs=None):
    inters = {}
    if ind_pairs is None:
        l = list(ind_to_pc.keys())
        l.sort()
        ind_pairs = [(a,b) for a,b in list(
            itertools.product(l,l)
        ) if a < b ]

    for ind1, ind2 in ind_pairs:

        # This makes us catch more bbox intersections
        thresh = scene_geom[[0 if ind1 < 0 else ind1, ind2]][:,:3].max() / DIM
        
        c1_inds, c2_inds = colEngine.findPCInter(
            ind_to_pc[ind1][0].T.unsqueeze(0),
            ind_to_pc[ind2][0].T.unsqueeze(0),
            ind_to_pc[ind1][1],
            ind_to_pc[ind2][1],
            thresh,
        )
        if len(c1_inds.shape) == 0 or len(c2_inds.shape) == 0 or c1_inds.shape[0] == 0 or c2_inds.shape[0] == 0:
            continue

        inters[f"{ind1}_{ind2}"] = (c1_inds, c2_inds)

    return inters


def points_obb(points, precision):
    try:
        to_origin, size = trimesh.bounds.oriented_bounds(points, angle_digits=precision)
        center = to_origin[:3, :3].transpose().dot(-to_origin[:3, 3])
        xdir = to_origin[0, :3]
        ydir = to_origin[1, :3]
    except scipy.spatial.qhull.QhullError:
        print('WARNING: falling back to PCA OBB computation since the more accurate minimum OBB computation failed.')
        center = points.mean(axis=0, keepdims=True)
        points = points - center
        center = center[0, :]
        pca = PCA()
        pca.fit(points)
        pcomps = pca.components_
        points_local = np.matmul(pcomps, points.transpose()).transpose()
        size = points_local.max(axis=0) - points_local.min(axis=0)
        xdir = pcomps[0, :]
        ydir = pcomps[1, :]
        
    box = torch.from_numpy(np.hstack([center, size, xdir, ydir]).reshape(1, -1)).to(torch.float32)
    box = box.cpu().numpy().squeeze()
    center = box[:3]
    size = box[3:6]
    xdir = box[6:9]
    xdir /= np.linalg.norm(xdir)
    ydir = box[9:]
    ydir /= np.linalg.norm(ydir)
    zdir = np.cross(xdir, ydir)
    zdir /= np.linalg.norm(zdir)

    return utils.orientProps(center, size[0], size[1], size[2], xdir, ydir, zdir)

# Checks if any cube1 face midpoints are inside pc2
def checkFaces(cube1, cube2, scene_geom):

    ind1 = 0 if cube1 < 0 else cube1
    ind2 = 0 if cube2 < 0 else cube2    
            
    faces = {
        'right': (torch.tensor([1.0,0.5,0.5],device=device), 0, 0),      
        'left': (torch.tensor([0.0,0.5,0.5],device=device), 0, 1),
        'top': (torch.tensor([0.5,1.0,0.5],device=device), 1, 0),
        'bot': (torch.tensor([0.5,0.0,0.5],device=device), 1, 1),
        'front': (torch.tensor([0.5,0.5,1.0],device=device), 2, 0),
        'back': (torch.tensor([0.5,0.5,0.0],device=device), 2, 1),
    }

    if cube2 == -1:
        faces.pop('bot')
        faces.pop('left')
        faces.pop('right')
        faces.pop('front')
        faces.pop('back')

        
    if cube2 == -2:
        faces.pop('top')
        faces.pop('left')
        faces.pop('right')
        faces.pop('front')
        faces.pop('back')
        
    best_score = 1e8
    best_face= None
    best_pt = None
    
    O = smp_pt(scene_geom[ind2], torch.zeros(3).to(device))
    A = torch.stack([
        scene_geom[ind2][0] * scene_geom[ind2][6:9],
        scene_geom[ind2][1] * scene_geom[ind2][9:12],
        scene_geom[ind2][2] * torch.cross(
            scene_geom[ind2][6:9], scene_geom[ind2][9:12]
        )
    ]).T
    
    for face in faces:
        smpt, fi, ft = faces[face]
        pt = smp_pt(scene_geom[ind1], smpt)
        B = pt - O            
        p = torch.tensor(np.linalg.solve(A.cpu(), B.cpu()))

        if p.min() < -0.01 or p.max() > 1.01:
            continue

        if ind2 == 0 and (p[1] >= .05 and p[1] <= .95):
            continue

        score = (p[fi] - ft).abs()

        if score < best_score:
            best_face = face
            best_score = score
            best_pt = pt
            
    return best_pt

    

def calcAttPoint(pair, pcs, scene_geom, ind_to_pc):
    cube1 = int(pair.split('_')[0])
    cube2 = int(pair.split('_')[1])

    ind1 = cube1 if cube1 > 0 else 0
    ind2 = cube2 if cube2 > 0 else 0
    
    if ind1 == ind2:
        return None

    pc1, pc2 = pcs

    dim1 = scene_geom[ind1][0] * scene_geom[ind1][1] * scene_geom[ind1][2]
    dim2 = scene_geom[ind2][0] * scene_geom[ind2][1] * scene_geom[ind2][2]

    att_point = None

    if cube1 > 0:
        att_point = checkFaces(cube1, cube2, scene_geom)

    if att_point is None and cube2 > 0:
        att_point = checkFaces(cube2, cube1, scene_geom)
        
    if att_point is None:
        if ind1 > ind2:        
            att_point = getBestATTPoint(pc1, pc2, cube1, cube2, ind_to_pc, ind1, ind2, scene_geom)
        else:
            att_point = getBestATTPoint(pc2, pc1, cube2, cube1, ind_to_pc, ind2, ind1, scene_geom)
    if att_point is None:
        return
        
    # att_point is point in 3D space, transform into local coordinate frames and reutrn

    collision = [ind1, ind2, None, None] 

    for i, ind in ((2, ind1), (3, ind2)):
            
        O = smp_pt(scene_geom[ind], torch.zeros(3).to(device))            
        A = torch.stack([
            scene_geom[ind][0] * scene_geom[ind][6:9],
            scene_geom[ind][1] * scene_geom[ind][9:12],
            scene_geom[ind][2] * torch.cross(
                scene_geom[ind][6:9], scene_geom[ind][9:12]
            )
        ]).T            
        B = att_point - O            
        p = np.linalg.solve(A.cpu(), B.cpu())
        p = np.clip(p, 0.0, 1.0)
        collision[i] = p.tolist()

    return collision

def calcAnyPoint(pair, pcs, scene_geom, ind_to_pc):
    cube1 = int(pair.split('_')[0])
    cube2 = int(pair.split('_')[1])

    ind1 = cube1 if cube1 > 0 else 0
    ind2 = cube2 if cube2 > 0 else 0
    
    if ind1 == ind2:
        return None

    pc1, pc2 = pcs

    dim1 = scene_geom[ind1][0] * scene_geom[ind1][1] * scene_geom[ind1][2]
    dim2 = scene_geom[ind2][0] * scene_geom[ind2][1] * scene_geom[ind2][2]

    att_point = None

    
    att_point = getanATTPoint(pc1, pc2, cube1, cube2, ind_to_pc, ind1, ind2, scene_geom)

    if att_point is None:

        att_point = getanATTPoint(pc2, pc1, cube2, cube1, ind_to_pc, ind2, ind1, scene_geom)
        
    if att_point is None:
        return
        
    # att_point is point in 3D space, transform into local coordinate frames and reutrn

    collision = [ind1, ind2, None, None] 

    for i, ind in ((2, ind1), (3, ind2)):
            
        O = smp_pt(scene_geom[ind], torch.zeros(3).to(device))            
        A = torch.stack([
            scene_geom[ind][0] * scene_geom[ind][6:9],
            scene_geom[ind][1] * scene_geom[ind][9:12],
            scene_geom[ind][2] * torch.cross(
                scene_geom[ind][6:9], scene_geom[ind][9:12]
            )
        ]).T            
        B = att_point - O            
        p = np.linalg.solve(A.cpu(), B.cpu())
        p = np.clip(p, 0.0, 1.0)
        collision[i] = p.tolist()

    return collision
    

def calcAttachments(inters, scene_geom, ind_to_pc):
    attachments = []    
    for pair in inters:        
        attachments.append(calcAttPoint(pair, inters[pair], scene_geom, ind_to_pc))
    return [a for a in attachments if a is not None]


# Takes in intersections and an ind
# Calculates intersection of all covered points
# Checks amount that each face can be shortened
# Updates parts to be shortened
# Returns None if nothing shortened
# Returns tuple of intersections to recalculate

def shorten_cube(inters, parts, ind, scene_geom):
    cov_inds = torch.zeros((DIM**3)).float().to(device)
    cov_inters = []
    nparts = []
    
    for pair in inters:
        ind1 = int(pair.split('_')[0])
        ind2 = int(pair.split('_')[1])
        if ind1 == ind:
            cov_inters.append(pair)
            cov_inds[inters[pair][0]] = 1.
            nparts.append(ind2)
            
        elif ind2 == ind:
            cov_inters.append(pair)
            cov_inds[inters[pair][1]] = 1.
            nparts.append(ind1)
            
    if cov_inds.sum().item() == 0:
        return None

    face_shorten = {}
                    
    dirs = {
        'right': (fr, torch.tensor([-.01, 0., 0.]).unsqueeze(0).repeat(400,1), 0),
        'left': (fl, torch.tensor([.01, 0., 0.]).unsqueeze(0).repeat(400,1), 0),
        'top': (ft, torch.tensor([0., -.01, 0.]).unsqueeze(0).repeat(400,1), 1),
        'bot': (fbo, torch.tensor([0., .01, 0.]).unsqueeze(0).repeat(400,1), 1),
        'front': (ff, torch.tensor([0., 0., -.01]).unsqueeze(0).repeat(400,1), 2),
        'back': (fba, torch.tensor([0., 0., .01]).unsqueeze(0).repeat(400,1), 2)
    }

    cube_geom = scene_geom[ind]

    p_r = torch.cat(
        (
            (cube_geom[6:9] / (cube_geom[6:9].norm(dim=0).unsqueeze(0) + 1e-8)).unsqueeze(1),
            (cube_geom[9:12] / (cube_geom[9:12].norm(dim=0).unsqueeze(0) + 1e-8)).unsqueeze(1),
            torch.cross(
                cube_geom[6:9] / (cube_geom[6:9].norm(dim=0).unsqueeze(0) + 1e-8),
                cube_geom[9:12] / (cube_geom[9:12].norm(dim=0).unsqueeze(0) + 1e-8)                
            ).unsqueeze(1)
        ), dim = 1)

    for d in dirs:
        dinds, inc, ki = dirs[d]
        inc = inc.to(device)

        for nd in range(101):
                        
            points = ((p_r @ ((((g[dinds]  + (inc * nd)).unsqueeze(0) - .5) * cube_geom[:3]).unsqueeze(-1))).squeeze() + cube_geom[3:6]).squeeze()   
            inside = False

            for nind in nparts:
                if inside:
                    break
                
                O = smp_pt(scene_geom[nind], torch.zeros(3).to(device))
                A = torch.stack([
                    scene_geom[nind][0] * scene_geom[nind][6:9],
                    scene_geom[nind][1] * scene_geom[nind][9:12],
                    scene_geom[nind][2] * torch.cross(
                        scene_geom[nind][6:9], scene_geom[nind][9:12]
                    )
                ]).T
                B = (points.T - O.unsqueeze(1)).cpu()
                p = torch.tensor(np.linalg.solve(A.cpu(), B.cpu())).T

                if p.min() >= -0.01 and p.max() <= 1.01:
                    inside = True
                    
            if not inside:
                break
            
        face_shorten[d] = max((nd-1) / 100, 0)

    dims = [
        ('right', 'left', 'xd', 'xdir'),
        ('top', 'bot', 'yd', 'ydir'),
        ('front', 'back', 'zd', 'zdir')
    ]
    
    for d1, d2, key, dn in dims:
        if face_shorten[d1] > 0 or face_shorten[d2] > 0:
            x = face_shorten[d1] * parts[ind][key]
            y = face_shorten[d2] * parts[ind][key]
            parts[ind][key] -= x + y

            parts[ind]['center'] += parts[ind][dn] * ((y-x) / 2)

    return cov_inters


def getBestATTPoint(pc1, pc2, cube1, cube2, ind_to_pc, ind1, ind2, scene_geom):

    a = ind_to_pc[cube1][0][pc1]
    b = ind_to_pc[cube2][0][pc2]
    
    joint = torch.cat((a, b), dim=0)

    jmin = joint.min(dim=0).values
    jmax = joint.max(dim=0).values
    
    xd = (jmax[0] - jmin[0]).abs()
    yd = (jmax[1] - jmin[1]).abs()
    zd = (jmax[2] - jmin[2]).abs()

    center = (jmax + jmin) / 2

    bbox_geom = torch.cat((
        xd.unsqueeze(0),
        yd.unsqueeze(0),
        zd.unsqueeze(0),
        center,
        torch.tensor([1.0,0.0,0.0]).to(device),
        torch.tensor([0.0,1.0,0.0]).to(device)
    ))

    p_r = torch.cat(
        (
            (bbox_geom[6:9] / (bbox_geom[6:9].norm(dim=0).unsqueeze(0) + 1e-8)).unsqueeze(1),
            (bbox_geom[9:12] / (bbox_geom[9:12].norm(dim=0).unsqueeze(0) + 1e-8)).unsqueeze(1),
            torch.cross(
                bbox_geom[6:9] / (bbox_geom[6:9].norm(dim=0).unsqueeze(0) + 1e-8),
                bbox_geom[9:12] / (bbox_geom[9:12].norm(dim=0).unsqueeze(0) + 1e-8)                
            ).unsqueeze(1)
        ), dim = 1)

    points = ((p_r @ (((attxyz - .5) * bbox_geom[:3]).unsqueeze(-1))).squeeze() + bbox_geom[3:6]).squeeze()

    # for each point find atp1 and atp2
    
    O1 = smp_pt(scene_geom[ind1], torch.zeros(3).to(device))

    A1 = torch.stack([
        scene_geom[ind1][0] * scene_geom[ind1][6:9],
        scene_geom[ind1][1] * scene_geom[ind1][9:12],
        scene_geom[ind1][2] * torch.cross(
            scene_geom[ind1][6:9], scene_geom[ind1][9:12]
        )
    ]).T

    O2 = smp_pt(scene_geom[ind2], torch.zeros(3).to(device))

    A2 = torch.stack([
        scene_geom[ind2][0] * scene_geom[ind2][6:9],
        scene_geom[ind2][1] * scene_geom[ind2][9:12],
        scene_geom[ind2][2] * torch.cross(
            scene_geom[ind2][6:9], scene_geom[ind2][9:12]
        )
    ]).T
    
    B1 = (points.T - O1.unsqueeze(1)).cpu()
    B2 = (points.T - O2.unsqueeze(1)).cpu()
        
    atp1 = torch.tensor(np.linalg.lstsq(A1.cpu(), B1, rcond=None)[0]).T
    atp2 = torch.tensor(np.linalg.lstsq(A2.cpu(), B2, rcond=None)[0]).T

    atps = torch.cat((atp1, atp2), dim = 1)
    
    ne_inds = (((atps >= -0.01).sum(dim=1) == 6).int() + ((atps <= 1.01).sum(dim=1) == 6)).int()
        
    if ind2 == 0:
        for i in range(1, 6):
            offset = 0.01 * i
            bb_ne_inds = (((atps[:, 4] <= offset).int() + (atps[:, 4] >= (1-offset)).int()) == 1).int()
            bb_ne_inds += ne_inds.int()

            bb_ne_inds = (bb_ne_inds == 3).nonzero().squeeze()
            if bb_ne_inds.sum() > 0:
                return points[bb_ne_inds].mean(dim=0)
            
        return None
        
    ne_inds = (ne_inds == 2).nonzero().squeeze()
 
    if ne_inds.sum() > 0:
        return points[ne_inds].mean(dim=0)                

    return None


def getanATTPoint(pc1, pc2, cube1, cube2, ind_to_pc, ind1, ind2, scene_geom):

    a = ind_to_pc[cube1][0][pc1]
    b = ind_to_pc[cube2][0][pc2]
    
    joint = torch.cat((a, b), dim=0)

    jmin = joint.min(dim=0).values
    jmax = joint.max(dim=0).values
    
    xd = (jmax[0] - jmin[0]).abs()
    yd = (jmax[1] - jmin[1]).abs()
    zd = (jmax[2] - jmin[2]).abs()

    center = (jmax + jmin) / 2

    bbox_geom = torch.cat((
        xd.unsqueeze(0),
        yd.unsqueeze(0),
        zd.unsqueeze(0),
        center,
        torch.tensor([1.0,0.0,0.0]).to(device),
        torch.tensor([0.0,1.0,0.0]).to(device)
    ))

    p_r = torch.cat(
        (
            (bbox_geom[6:9] / (bbox_geom[6:9].norm(dim=0).unsqueeze(0) + 1e-8)).unsqueeze(1),
            (bbox_geom[9:12] / (bbox_geom[9:12].norm(dim=0).unsqueeze(0) + 1e-8)).unsqueeze(1),
            torch.cross(
                bbox_geom[6:9] / (bbox_geom[6:9].norm(dim=0).unsqueeze(0) + 1e-8),
                bbox_geom[9:12] / (bbox_geom[9:12].norm(dim=0).unsqueeze(0) + 1e-8)                
            ).unsqueeze(1)
        ), dim = 1)

    points = ((p_r @ (((attxyz - .5) * bbox_geom[:3]).unsqueeze(-1))).squeeze() + bbox_geom[3:6]).squeeze()

    # for each point find atp1 and atp2
    
    O1 = smp_pt(scene_geom[ind1], torch.zeros(3).to(device))

    A1 = torch.stack([
        scene_geom[ind1][0] * scene_geom[ind1][6:9],
        scene_geom[ind1][1] * scene_geom[ind1][9:12],
        scene_geom[ind1][2] * torch.cross(
            scene_geom[ind1][6:9], scene_geom[ind1][9:12]
        )
    ]).T

    O2 = smp_pt(scene_geom[ind2], torch.zeros(3).to(device))

    A2 = torch.stack([
        scene_geom[ind2][0] * scene_geom[ind2][6:9],
        scene_geom[ind2][1] * scene_geom[ind2][9:12],
        scene_geom[ind2][2] * torch.cross(
            scene_geom[ind2][6:9], scene_geom[ind2][9:12]
        )
    ]).T
    
    B1 = (points.T - O1.unsqueeze(1)).cpu()
    B2 = (points.T - O2.unsqueeze(1)).cpu()
        
    atp1 = torch.tensor(np.linalg.lstsq(A1.cpu(), B1, rcond=None)[0]).T
    atp2 = torch.tensor(np.linalg.lstsq(A2.cpu(), B2, rcond=None)[0]).T

    atps = torch.cat((atp1, atp2), dim = 1)
    
    ne_inds = (((atps >= -0.05).sum(dim=1) == 6).int() + ((atps <= 1.05).sum(dim=1) == 6)).int()
        
    if ind2 == 0:
        for i in range(1, 6):
            offset = 0.01 * i
            bb_ne_inds = (((atps[:, 4] <= offset).int() + (atps[:, 4] >= (1-offset)).int()) == 1).int()
            bb_ne_inds += ne_inds.int()

            bb_ne_inds = (bb_ne_inds == 3).nonzero().squeeze()
            if bb_ne_inds.sum() > 0:
                return points[bb_ne_inds].mean(dim=0)
            
        return None
        
    ne_inds = (ne_inds == 2).nonzero().squeeze()
 
    if ne_inds.sum() > 0:
        return points[ne_inds].mean(dim=0)                

    return None
