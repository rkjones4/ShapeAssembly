import torch
import sys
import ast
import numpy as np
import random
import os
import pickle
import intersect as inter
import random
from copy import deepcopy
import utils as utils
import networkx as nx
from symmetry import addSimpSymmetries, addSymSubPrograms

VERBOSE = False

DO_SHORTEN = True
DO_SIMP_SYMMETRIES = True
DO_VALID_CHECK = True
DO_NORM_AA_CUBES = True
DO_SEM_FLATTEN = True
DO_SEM_REHIER = True

SQUARE_THRESH = 0.1
SD_THRESH = 0.01
AA_THRESH = 0.995

SCOL_MAP = {
    'chair': set(['caster', 'mechanical_control']),
    'table': set(['caster', 'cabinet_door', 'drawer', 'keyboard_tray']),
    'storagefurniture': set(['drawer', 'cabinet_door', 'mirror', 'caster'])
}

SFLAT_MAP = {
    'chair': set(['chair_back', 'chair_arm', 'chair_base', 'chair_seat', 'footrest', 'chair_head']),
    'table': set(['tabletop', 'table_base']),
    'storagefurniture': set(['cabinet_frame', 'cabinet_base'])
}

SRH_MAP = {
    'storagefurniture': ('cabinet_frame', set(['countertop', 'shelf', 'drawer', 'cabinet_door', 'mirror']))
}

def isSquare(a, b, c):
    v = (a - b).abs()/max(a.item(), b.item(), c.item()) < SQUARE_THRESH
    return v
    
def isAxisAligned(props, bbox):    

    xdir = props['xdir']
    ydir = props['ydir']
    zdir = props['zdir']
    
    xdir /= xdir.norm()
    ydir /= ydir.norm()
    zdir /= zdir.norm()
    
    if xdir.dot(bbox['xdir']) < AA_THRESH:
        return False
                
    if ydir.dot(bbox['ydir']) < AA_THRESH:
        return False
                
    if zdir.dot(bbox['zdir']) < AA_THRESH:
        return False

    return True

def shouldChange(props, bbox):    

    if isAxisAligned(props, bbox):
        return True
    
    xsquare = isSquare(props['zd'], props['yd'], props['xd'])
    ysquare = isSquare(props['zd'], props['xd'], props['yd'])
    zsquare = isSquare(props['xd'], props['yd'], props['zd'])
    
    if xsquare and ysquare and zsquare:
        return True

    if xsquare:
        return isSpecAxisAligned(props['xdir'], bbox['xdir'])

    if ysquare:
        return isSpecAxisAligned(props['ydir'], bbox['ydir'])

    if zsquare:
        return isSpecAxisAligned(props['zdir'], bbox['zdir'])
      
    return False

def isSpecAxisAligned(cdir, axis):
    cdir /= cdir.norm()

    if cdir.dot(axis) >= AA_THRESH:
        return True
    else:
        return False

def getDataPath(category):
    return f"/home/{os.getenv('USER')}/pnhier/{category}_hier/"

def getSemOrder(category):
    if category == "chair":
        sem_order_path = "../stats/part_semantics/PGP-Chair.txt"
    elif category == "storagefurniture":
        sem_order_path = "../stats/part_semantics/PGP-Storage.txt"
    elif category == "table":
        sem_order_path = "../stats/part_semantics/PGP-Table.txt"
    else:
        assert False, f'Invalid Category {category}'
        
    sem_order = {"bbox": "-1","other":"100"}

    with open(sem_order_path) as f:
        for line in f:
            sem_order[line.split()[1].split('/')[-1]] = line.split()[0]
    return sem_order


def cubeOrder(cubes, names, sem_order):
    d = []
    
    min_c = np.array([1e8,1e8,1e8])
    max_c = np.array([-1e8,-1e8,-1e8])
    
    for rw in cubes:
        min_c = np.min((min_c, rw['center'].numpy()), axis = 0)
        max_c = np.max((max_c, rw['center'].numpy()), axis = 0)

    mac = np.max(max_c)
    mic = np.min(min_c)

    for c_ind, (rw, name) in enumerate(zip(cubes, names)):        
        sc = (rw['center'].numpy() - mic) / (mac - mic)
        
        x_r = round(sc[0]*2)/2
        y_r = round(sc[1]*2)/2
        z_r = round(sc[2]*2)/2
        
        d.append((
            int(sem_order[name]),
            x_r + y_r + z_r,
            x_r,
            y_r,
            z_r,
            sc[0],
            sc[1],
            sc[2],
            c_ind
        ))

    d.sort()
    return [c_ind for _,_,_,_,_,_,_,_,c_ind in d]

Sbbox = {
    'xdir': torch.tensor([1.0, 0.0, 0.0]),
    'ydir': torch.tensor([0.0, 1.0, 0.0]),
    'zdir': torch.tensor([0.0, 0.0, 1.0]),    
}

def jsonToProps(json):
    json = np.array(json)
    center = np.array(json[:3])
    
    xd = json[3]
    yd = json[4]
    zd = json[5]
    xdir = json[6:9]
    xdir /= np.linalg.norm(xdir)
    ydir = json[9:]
    ydir /= np.linalg.norm(ydir)
    zdir = np.cross(xdir, ydir)
    zdir /= np.linalg.norm(zdir)

    if xd < SD_THRESH or yd < SD_THRESH or zd < SD_THRESH:
        return None
    
    props = utils.orientProps(center, xd, yd, zd, xdir, ydir, zdir)

    if DO_NORM_AA_CUBES:
        if shouldChange(props, Sbbox):
            props['xdir'] = Sbbox['xdir'].clone().detach()
            props['ydir'] = Sbbox['ydir'].clone().detach()
            props['zdir'] = Sbbox['zdir'].clone().detach()
    
    return props


def addAttachments(node, sem_order):
    co = cubeOrder(node['cubes'], node['children_names'], sem_order)
    
    for key in ['cubes', 'children', 'children_names']:    
        node[key] = [node[key][i] for i in co]

    ind_to_pc, scene_geom = inter.samplePC(node['cubes'], split_bbox=True)
    inters = inter.findInters(ind_to_pc, scene_geom)
    node['attachments'] = inter.calcAttachments(inters, scene_geom, ind_to_pc)
    for child in node['children']:
        if len(child) > 0:
            addAttachments(child, sem_order)

            
# From raw json, get graph structure and all leaf cuboids
def getShapeHier(ind, category):
    data_path = getDataPath(category)
    with open(data_path + ind + ".json") as f:
        json = None
        for line in f:
            json = ast.literal_eval(line)

    hier = {}
            
    queue = [(json, hier)]
    
    while len(queue) > 0:
        json, node = queue.pop(0)
        
        if "children" not in json:
            continue

        name = json["label"]

        # Don't add sub-programs when we collapse
        collapse_names = SCOL_MAP[category]
        if name in collapse_names:
            continue
        
        while("children" in json and len(json["children"])) == 1:
            json = json["children"][0]

        if "children" not in json:
            continue
        
        node.update({
            "children": [],
            "children_names": [],
            "cubes": [],
            "name": name
        })
        
        for c in json["children"]:
            cprops = jsonToProps(c["box"])
            if cprops is not None:
                new_c = {}
                queue.append((c, new_c))
                node["children"].append(new_c)
                node["cubes"].append(cprops)
                node["children_names"].append(c["label"])

    return hier


# TODO should probably have loop here that enforces
# bounding box faces connect to child geom, if obb comp
# is done correctly this shouldn't be needed though
def getOBB(parts):
    part_corners = inter.sampleCorners(parts)
    bbox = inter.points_obb(part_corners.cpu(), 1)
    return bbox


def cleanHier(hier):
    for i in range(len(hier['children'])):
        if len(hier['children'][i]) > 0:
            cleanHier(hier['children'][i])
            
    if len(hier['children']) == 0:
        for key in list(hier.keys()):
            hier.pop(key)


def trimHier(hier):
    for i in range(len(hier['children'])):
        if len(hier['children'][i]) > 0:
            if len(hier['children'][i]['children']) == 1:
                hier['cubes'][i] = hier['children'][i]['cubes'][0]
                hier['children_names'][i] = hier['children'][i]['children_names'][0]
                hier['children'][i] = hier['children'][i]['children'][0]
                
        if len(hier['children'][i]) > 0:
            trimHier(hier['children'][i])                

            
def fillHier(hier):    
    for i in range(len(hier['children'])):
        if len(hier['children'][i]) > 0:
            hier['cubes'][i] = fillHier(hier['children'][i])            

    hier['bbox'] = getOBB(hier['cubes'])
    return deepcopy(hier['bbox'])


# centers and orients root bounding box
# propogates transformation to all cuboids
# also instanties bounding box into the cube + children spots
def normalizeHier(hier):
    hier.pop('bbox')
    
    rbbox = {}
    
    samps = inter.sampleCorners(hier['cubes']).cpu()

    dims = samps.max(dim=0).values - samps.min(dim=0).values

    rbbox['xd'] = dims[0]
    rbbox['yd'] = dims[1]
    rbbox['zd'] = dims[2]

    rbbox['center'] = (samps.max(dim=0).values + samps.min(dim=0).values) / 2

    rbbox['xdir'] = torch.tensor([1.,0.,0.])
    rbbox['ydir'] = torch.tensor([0.,1.,0.])
    rbbox['zdir'] = torch.tensor([0.,0.,1.])

    hier['bbox'] = rbbox
    
    offset = rbbox['center']

    q = [hier]

    while len(q) > 0:
        
        n = q.pop(0)
        bbox = n.pop('bbox')
        n['children'] = [{}] + n['children']
        n['children_names'] = ["bbox"] + n['children_names']
        n['cubes'] = [bbox] + n['cubes']
        
        for i in range(len(n['cubes'])):            
            n['cubes'][i]['center'] = n['cubes'][i]['center'] - offset            
            
        for c in n['children']:
            if len(c) > 0:
                q.append(c)                


def markLeafCubes(hier):
    parts = []
    q = [hier]
    while(len(q) > 0):
        n = q.pop(0)
        n['leaf_inds'] = []
        assert(len(n['cubes']) == len(n['children']))
        for cu, ch in zip(n['cubes'], n['children']):
            if len(ch) > 0:
                q.append(ch)
                n['leaf_inds'].append(-1)
            else:
                n['leaf_inds'].append(len(parts))
                parts.append(cu)

    return parts


def replace_parts(hier, parts, key):
    q = [hier]
    while(len(q) > 0):
        n = q.pop(0)

        binds = []
        
        for i in range(len(n[key])):
            if n[key][i] != -1:
                lpart = parts[n[key][i]]
                
                if lpart is None:
                    binds.append(i)
                else:
                    n['cubes'][i] = lpart

        binds.sort(reverse=True)
        for bi in binds:
            n['children'].pop(bi)
            n['cubes'].pop(bi)
            n['children_names'].pop(bi)
            
        for c in n['children']:
            if c is not None and len(c) > 0:
                q.append(c)

        n.pop(key)

# Takes in a hierarchy of just leaf cuboids,
# finds new parameters for leaf cuboids so that
# part-to-part connections are as valid as possible

def shortenLeaves(hier):
    if VERBOSE:
        print("Doing Shortening")

    parts = markLeafCubes(hier)
    bad_inds = inter.findHiddenCubes(parts)

    ind_to_pc, scene_geom = inter.samplePC(parts)
    inters = inter.findInters(ind_to_pc, scene_geom)

    dim_parts = [
        (p['xd'] * p['yd'] * p['zd'], i) for i,p in enumerate(parts)
    ]
    
    dim_parts.sort()

    for _, ind in dim_parts:
        if ind in bad_inds:
            continue
        
        if VERBOSE:
            print(f"Shortening Leaf ind: {ind}")
            
        sres = inter.shorten_cube(inters, parts, ind, scene_geom)
        if sres is not None:
            t_ind_to_pc, t_scene_geom = inter.samplePC([parts[ind]])
            ind_to_pc[ind] = t_ind_to_pc[0]
            scene_geom[ind] = t_scene_geom[0]
            sres = [(int(s.split('_')[0]), int(s.split('_')[1])) for s in sres]
            new_inters = inter.findInters(ind_to_pc, scene_geom, sres)
            inters.update(new_inters)
            if parts[ind]['xd'] < SD_THRESH or \
               parts[ind]['yd'] < SD_THRESH or \
               parts[ind]['zd'] < SD_THRESH:
                bad_inds.append(ind)

    for bi in bad_inds:
        parts[bi] = None
            
    replace_parts(hier, parts, 'leaf_inds')
                
def make_conn_graph(num_nodes, attachments):

    edges = []
    for (ind1, ind2, _, _) in attachments:
        edges.append((ind1, ind2))

    G = nx.Graph()
    G.add_nodes_from(list(range(num_nodes)))
    G.add_edges_from(edges)
    
    return G


def assertConnected(num_nodes, attachments):
    G = make_conn_graph(num_nodes, attachments)
    assert nx.number_connected_components(G) == 1, 'disconnected graph'

    
def checkConnected(node):
    assertConnected(len(node['cubes']), node['attachments'])
    for c in node['children']:
        if len(c) > 0:
            checkConnected(c)
    
def memoize(f):
    def helper(ind, category):
        cdir = "parse_cache"
        cached_res = os.listdir(cdir)
        if ind in cached_res:
            return pickle.load(open(cdir+"/"+ind, "rb"))
        else:
            hier = f(ind, category)
            pickle.dump(hier, open(cdir+"/"+ind, "wb"))
            return hier
    return helper


def checkCubeNum(node):
    assert len(node['cubes']) <= 11, f"Saw program with {len(node['cubes'])} cubes"
    for c in node['children']:
        if len(c) > 0:
            checkCubeNum(c)


def flattenNode(node):
    for c in node['children']:
        if len(c) > 0:
            flattenNode(c)

    # Now everything has one sub-program
    fcubes = []
    fchildren = []
    fchildren_names = []

    for i in range(len(node['children'])):
        if len(node['children'][i]) == 0:
            fcubes.append(node['cubes'][i])
            fchildren.append(node['children'][i])
            fchildren_names.append(node['children_names'][i])
        else:
            fcubes += node['children'][i]['cubes']
            fchildren += node['children'][i]['children']
            fchildren_names += node['children'][i]['children_names']

    node['cubes'] = fcubes
    node['children'] = fchildren
    node['children_names'] = fchildren_names
    
            
def semFlattenHier(hier, category):
    flat_names = SFLAT_MAP[category]
    q = [hier]
    while (len(q) > 0):
        node = q.pop(0)
        if node['name'] in flat_names:
            flattenNode(node)
        else:
            for c in node['children']:
                if len(c) > 0:
                    q.append(c)

def semReHier(hier, category):

    if category not in SRH_MAP:
        return

    rh_tar, rh_names = SRH_MAP[category]

    if rh_tar not in hier['children_names']:
        return

    rhinds = []
    for i,name in enumerate(hier['children_names']):
        if name in rh_names:
            rhinds.append(i)

    rhinds.sort(reverse=True)

    ti = hier['children_names'].index(rh_tar)

    for i in rhinds:        
        for key in ['children_names', 'cubes', 'children']:        
            hier['children'][ti][key].append(hier[key][i])

    for i in rhinds:        
        for key in ['children_names', 'cubes', 'children']:        
            hier[key].pop(i)
    
    if len(hier['children']) == 1:
        hier['children_names'] = hier['children'][0]['children_names']
        hier['cubes'] = hier['children'][0]['cubes']
        hier['children'] = hier['children'][0]['children']
                    
#@memoize
def parseJsonToHier(ind, category, get_gt=False):
    sem_order = getSemOrder(category)
        
    hier = getShapeHier(ind, category)

    assert len(hier) > 0, 'saw empty hier'
    
    if DO_SEM_FLATTEN:
        semFlattenHier(hier, category)

    if DO_SEM_REHIER:
        semReHier(hier, category)

    if DO_SHORTEN:
        shortenLeaves(hier)

    cleanHier(hier)        
    trimHier(hier)
    fillHier(hier)
    
    normalizeHier(hier)

    addAttachments(hier, sem_order)
        
    if get_gt:
        return hier

    if DO_VALID_CHECK:
        checkConnected(hier)
    
    if DO_SIMP_SYMMETRIES:
        addSymSubPrograms(hier)
        addSimpSymmetries(hier)

    if DO_VALID_CHECK:
        checkCubeNum(hier)
        
    return hier
