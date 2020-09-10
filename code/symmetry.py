import torch
import numpy as np
import json_parse as jp
import intersect as inter
from copy import deepcopy
import math
import utils

SDIM_THRESH = .15
SANG_THRESH = .1
SPOS_THRESH = .1
VATT_THRESH = .05
CATT_THRESH = .3
GROUP_THRESH = .1
CPT_THRESH = .1

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


def smp_rel_pos(geom, gpt):
    O = smp_pt(geom, torch.zeros(3))

    A = torch.stack([
        geom[0] * geom[6:9],
        geom[1] * geom[9:12],
        geom[2] * torch.cross(
            geom[6:9], geom[9:12]
        )
    ]).T

    B = gpt - O
    p = torch.tensor(np.linalg.solve(A.cpu(), B.cpu()))
    
    return p


def approx_same_dims(a, b, m):    
    if ((a['xd'] - b['xd']).abs() / m) > SDIM_THRESH or \
       ((a['yd'] - b['yd']).abs() / m) > SDIM_THRESH or \
       ((a['zd'] - b['zd']).abs() / m) > SDIM_THRESH:
        return False

    return True

def approx_same_point(a, b, m, thresh=SPOS_THRESH):
    if (a-b).norm() / m > thresh:
        return False
    
    return True

# Checks if pt is a valid place to be for cuboid c
def isValidAtt(node, pt, cind):
    ct = cubeToTensor(node['cubes'][cind])
    O = smp_pt(ct, torch.zeros(3))            
    A = torch.stack([
        ct[0] * ct[6:9],
        ct[1] * ct[9:12],
        ct[2] * torch.cross(
            ct[6:9], ct[9:12]
        )
    ]).T            
    B = pt - O            
    p = torch.tensor(np.linalg.solve(A.cpu(), B.cpu()))

    if cind == 0 and (p[1] - .5).abs() < .5 - VATT_THRESH:
        return False

    if (p - .5).abs().max() >= .5 + VATT_THRESH:
        return False

    return True


# checks if pt location in c is close to att
def isCloseAtt(node, pt, cind, att):
    ct = cubeToTensor(node['cubes'][cind])
    O = smp_pt(ct, torch.zeros(3))            
    A = torch.stack([
        ct[0] * ct[6:9],
        ct[1] * ct[9:12],
        ct[2] * torch.cross(
            ct[6:9], ct[9:12]
        )
    ]).T            
    B = pt - O            
    p = torch.tensor(np.linalg.solve(A.cpu(), B.cpu()))
    
    if (p - torch.tensor(att)).norm() > CATT_THRESH:
        return False
    
    return True

# checks if pt location in c is close to att
def isClosePt(node, pt, cind, att):
    ct = cubeToTensor(node['cubes'][cind])
    spt = smp_pt(ct, torch.tensor(att))                
        
    if (pt - spt).norm() > CPT_THRESH:
        return False
    
    return True
    
    
def cubeToTensor(c):
    return torch.cat((
        c['xd'].unsqueeze(0),
        c['yd'].unsqueeze(0),
        c['zd'].unsqueeze(0),
        c['center'],            
        c['xdir'],
        c['ydir']
    ))

    
# Is there a translational symmetry from i -> o
# if so return (axis, scale in terms of bbox dimension)
def checkTranSym(node, i, o):
    c_i = node['cubes'][i]
    c_o = node['cubes'][o]
    
    if not approx_same_dims(c_i, c_o, max(node['cubes'][0]['xd'], node['cubes'][0]['yd'], node['cubes'][0]['zd'])):
        return None

    cdir = c_o['center'] - c_i['center']
    scale = cdir.norm()
    cdir /= cdir.norm()

    if cdir.dot(node['cubes'][0]['xdir']) > 1 - SANG_THRESH:
        tn = 'tr_X'
        td = node['cubes'][0]['xd']
        tdir = node['cubes'][0]['xdir']
        
    elif cdir.dot(node['cubes'][0]['ydir']) > 1 - SANG_THRESH:
        tn = 'tr_Y'
        td = node['cubes'][0]['yd']
        tdir = node['cubes'][0]['ydir']
        
    elif cdir.dot(node['cubes'][0]['zdir']) > 1 - SANG_THRESH:
        tn = 'tr_Z'
        td = node['cubes'][0]['zd']
        tdir = node['cubes'][0]['zdir']
        
    else:
        return None

    for n, at1 in node['pc_atts'][i]:
        opt = smp_pt(
            cubeToTensor(c_i),
            torch.tensor(at1)
        )
        tpt = opt + (tdir * scale)

        #if not isValidAtt(node, tpt, n) or not isCloseAtt(node, tpt, o, at1):
        if (isValidAtt(node, opt, n) and not isValidAtt(node, tpt, n)) or not isClosePt(node, tpt, o, at1):
            return None

    return tn, (scale/td).item()



def getRefMatrixHomo(axis, center):

    m = center
    d = axis / axis.norm()

    refmat = torch.stack((
        torch.stack((1 - 2 * d[0] * d[0], -2 * d[0] * d[1], -2 * d[0] * d[2], 2 * d[0] * d[0] * m[0] + 2 * d[0] * d[1] * m[1] + 2 * d[0] * d[2] * m[2])),
        torch.stack((-2 * d[1] * d[0], 1 - 2 * d[1] * d[1], -2 * d[1] * d[2], 2 * d[1] * d[0] * m[0] + 2 * d[1] * d[1] * m[1] + 2 * d[1] * d[2] * m[2])),
        torch.stack((-2 * d[2] * d[0], -2 * d[2] * d[1], 1 - 2 * d[2] * d[2], 2 * d[2] * d[0] * m[0] + 2 * d[2] * d[1] * m[1] + 2 * d[2] * d[2] * m[2]))
    ))

    return refmat
    
def reflect_cube(c, center, ndir):
    ref_c = {}
    pad = torch.nn.ConstantPad1d((0, 1), 1.0)
    reflection = getRefMatrixHomo(ndir, center)
    nreflection = torch.cat((reflection[:,:3], torch.zeros(3,1)), dim=1)

    posHomo = pad(c['center'])
    ref_c['center'] = reflection @ posHomo

    xHomo = pad(c['xdir'])
    yHomo = pad(c['ydir'])
    zHomo = pad(c['zdir'])
    
    ref_c['xdir'] =  nreflection @ xHomo
    ref_c['ydir'] =  nreflection @ yHomo
    ref_c['zdir'] =  nreflection @ zHomo

    return ref_c


def reflect_point(p, center, ndir):
    pad = torch.nn.ConstantPad1d((0, 1), 1.0)
    reflection = getRefMatrixHomo(ndir, center)
    posHomo = pad(p)
    return reflection @ posHomo


# Is there reflectional symmetry from i->o
# If so return (plane)
def checkRefSym(node, i, o):

    c_i = node['cubes'][i]
    c_o = node['cubes'][o]

    if not approx_same_dims(c_i, c_o, max(node['cubes'][0]['xd'], node['cubes'][0]['yd'], node['cubes'][0]['zd'])):
        return None
    
    cdir = c_o['center'] - c_i['center']
    cdir /= cdir.norm()

    if cdir.dot(node['cubes'][0]['xdir']) > 1 - SANG_THRESH:
        rn = 'ref_X'
        rdir = node['cubes'][0]['xdir']
        
    elif cdir.dot(node['cubes'][0]['ydir']) > 1 - SANG_THRESH:
        rn = 'ref_Y'
        rdir = node['cubes'][0]['ydir']
        
    elif cdir.dot(node['cubes'][0]['zdir']) > 1 - SANG_THRESH:
        rn = 'ref_Z'
        rdir = node['cubes'][0]['zdir']
        
    else:
        return None

    
    center = smp_pt(cubeToTensor(node['cubes'][0]), torch.tensor([.5, .5, .5]))
    
    c_ref = reflect_cube(c_i, center, rdir)
    
    if not approx_same_point(
            c_ref['center'],
            c_o['center'], 
            max(node['cubes'][0]['xd'], node['cubes'][0]['yd'], node['cubes'][0]['zd'])
    ):
        return None
    
    for n, at1 in node['pc_atts'][i]:
        opt = smp_pt(
            cubeToTensor(c_i),
            torch.tensor(at1)
        )
        rpt = reflect_point(opt, center, rdir)

        if rn == 'ref_X':
            rat1 = [1-at1[0], at1[1], at1[2]]
        elif rn == 'ref_Y':
            rat1 = [at1[0], 1-at1[1], at1[2]]
        elif rn == 'ref_Z':
            rat1 = [at1[0], at1[1], 1-at1[2]]
            
        #if not isValidAtt(node, rpt, n) or not isCloseAtt(node, rpt, o, rat1):
        if (isValidAtt(node, opt, n) and not isValidAtt(node, rpt, n)) or not isClosePt(node, rpt, o, rat1):
            return None

    return rn
    
# takes in a dict of symmetries, if any of the translational symmetries have more than one element try to group them
# returned group symmetries, and the longest symmetry seen
def groupSyms(syms):

    gsyms = {}
    longest = 2
    for st in syms:
        if 'tr' not in st:
            gsyms[st] = syms[st]
            
        elif len(syms[st]) == 1:
            gsyms[st] = (syms[st][0][0], [syms[st][0][1]])

        else:
            best_l = 0
            syms[st].sort()
            for l in range(0, len(syms[st])):
                md = syms[st][l][0]
                exps = (torch.tensor([(l+1)*s[0] / md for s in syms[st][:l+1]]) - torch.arange(1, l+2)).abs().mean()
                
                if exps > GROUP_THRESH:
                    break
                best_l = l
            
            group = [s[1] for s in syms[st][:best_l+1]]
            longest = max(longest, len(group) + 1)
            gsyms[st] = (syms[st][best_l][0], group)
            
    return gsyms, longest

# returns list of lists of indices, where each sub-list is in the same semantic group
def getSemanticGroups(node):
    groups = {}
    #for i, l in enumerate(node['sem_labels'].argmax(dim=1).tolist()):
    for i, l in enumerate(node['children_names']):
        #spec_add(groups, l, i)
        spec_add(groups, 'all', i)
    return groups
        
def spec_add(d, k, v):
    if k in d:
        d[k].append(v)
    else:
        d[k] = [v]


def checkSameAtts(node, ind, oind):
    pc_atts = node['pc_atts']
    i_atts = [i[0] for i in pc_atts[ind]]
    o_atts = [i[0] for i in pc_atts[oind]]

    i_atts.sort()
    o_atts.sort()
    
    return (i_atts == o_atts)    
        
# For the indices in group, returns all symmetries where index is canonical member
def getSymsForGroup(node, group):
    
    if len(group) == 1:
        return {}

    if 'syms' in node:
        prev_syms = set([s[0] for s in node['syms']])
    else:
        prev_syms = set([])
        
    syms = {}
    for ind in group:
        nsyms = {}
        for oind in group:
            if ind == oind or ind in prev_syms or oind in prev_syms:
                continue

            if not checkSameAtts(node, ind, oind):
                continue            
            
            trans_sym = checkTranSym(node, ind, oind)

            ref_sym = checkRefSym(node, ind, oind)            

            if trans_sym is not None:
                spec_add(nsyms, trans_sym[0], (trans_sym[1], oind))
            
            if ref_sym is not None:
                spec_add(nsyms, ref_sym, oind)

        ngsyms, longest = groupSyms(nsyms)

        if len(ngsyms) > 0:
            syms[ind] = (longest, ngsyms)

    return syms

def getLongestTranSym(ind, sg, mv):
    for st in ['tr_X', 'tr_Y', 'tr_Z']:
        if st in sg[1] and len(sg[1][st][1]) == (mv-1) :
            ntr = sg[1][st][1]
            ns = [ind, st, len(sg[1][st][1]), sg[1][st][0], ntr]
            return ns, ntr

    
# Any node that shows up in ntr, remove all symmetries that they are involved in in pot_syms
def removePotSyms(pot_syms, ntr):
    for n in ntr:
        if n in pot_syms:
            pot_syms.pop(n)

    ntr = set(ntr)

    ps_inds = []
    
    for i in pot_syms:
        sts_to_pop = []
        for st in pot_syms[i][1]:
            if 'ref' in st:
                if pot_syms[i][1][st][0] in ntr:
                    sts_to_pop.append(st)

            elif 'tr' in st:
                js_to_pop = []
                for j in range(len(pot_syms[i][1][st][1])):
                    if pot_syms[i][1][st][1][j] in ntr:
                        js_to_pop = [j] + js_to_pop

                for j in js_to_pop:
                    pot_syms[i][1][st][1].pop(j)

                if len(pot_syms[i][1][st][1]) == 0:
                    sts_to_pop.append(st)

                elif len(js_to_pop) > 0:
                    pot_syms[i] = (pot_syms[i][0] - len(js_to_pop), pot_syms[i][1])
                    

        for st in sts_to_pop:
            pot_syms[i][1].pop(st)

        ps_inds = [i] + ps_inds
            
    for i in ps_inds:
        if len(pot_syms[i][1]) == 0:
            pot_syms.pop(i)
                             

# Takes in all of the symmetries in a group, chooses 'best' one, removes all other ones from the other nodes
def getMaxGroupSyms(pot_syms, syms = [], tr = []):

    if len(pot_syms) == 0:
        return syms, tr
    
    ml = 2
    mi = None

    for ind in pot_syms:
        if pot_syms[ind][0] > ml:
            ml = pot_syms[ind][0]
            mi = ind

    if mi is not None:
        ns, ntr = getLongestTranSym(mi, pot_syms[mi], ml)        
        removePotSyms(pot_syms, [mi] + ntr)
        return getMaxGroupSyms(pot_syms, syms + [ns], tr + ntr)

    for st in ['ref_X', 'ref_Y', 'ref_Z']:
        for ind in pot_syms:
            if st in pot_syms[ind][1]:
                ntr = pot_syms[ind][1][st]
                ns = [ind, st, ntr]
                removePotSyms(pot_syms, [ind] + ntr)
                return getMaxGroupSyms(pot_syms, syms + [ns], tr + ntr)

    for st in ['tr_X', 'tr_Y', 'tr_Z']:
        for ind in pot_syms:
            if st in pot_syms[ind][1]:
                ntr = [pot_syms[ind][1][st][1][0]]
                ns = [ind, st, 1, pot_syms[ind][1][st][0], ntr]                
                removePotSyms(pot_syms, [ind] + ntr)
                return getMaxGroupSyms(pot_syms, syms + [ns], tr + ntr)
            

            
# remove all nodes to remove from the list, from children, remove all attachments that mention these nodes
def cleanNode(node, ntr, syms):
        
    if len(syms) == 0:
        if 'syms' not in node:
            node['syms'] = []
        return
        
    sntr = []

    num_cubes_per_sym = {}
    
    for sym in syms:
        sym_cubes = []
        num_cubes_per_sym[sym[0]] = len(sym[-1])
        for itr in sym[-1]:
            sntr.append(itr)
            sym_cubes.append(node['cubes'][itr])
        sym[-1] = sym_cubes
        
    syms.sort()
    
    sym_cubes = []
    for sym in syms:
        scubes = sym.pop(-1)        
        sym_cubes += scubes
        
    sntr.sort(reverse=True)

    cube_map = {}
    count = 0
    for i in range(len(node['cubes'])):
        if i not in ntr:
            cube_map[i] = count
            count += 1
    
    for n in sntr:
        for key in ['children', 'children_names', 'cubes']:        
            node[key].pop(n)
    
    atts = []

    temp_atts = node.pop('attachments')

    for att in temp_atts:
        if att[0] not in ntr and att[1] not in ntr:            
            atts.append((cube_map[att[0]], cube_map[att[1]], att[2], att[3]))
    
    if 'syms' in node:
        node['syms'] += syms
    else:
        node['syms'] = syms 

    node['syms'] = [[cube_map[s[0]]] + s[1:] for s in node['syms']]

    node['attachments'] = atts

            
def getMaxSyms(pot_group_syms):
    nodes_to_remove = set()
    syms = []
    for pgs in pot_group_syms:
        if len(pgs) == 0:
            continue

        group_syms, gntr = getMaxGroupSyms(pgs)
        syms += group_syms
        nodes_to_remove = nodes_to_remove.union(set(gntr))
        
    return syms, nodes_to_remove


def getPerCubeAtts(node):
    pc_atts = {}
    
    for c1, c2, at1, at2 in node['attachments']:
        spec_add(pc_atts, c1, (c2, at1))
        spec_add(pc_atts, c2, (c1, at2))

    return pc_atts


def addSimpSymmetries(node):
    for child in node['children']:
        if len(child) > 0:
            addSimpSymmetries(child)
            
    node['pc_atts'] = getPerCubeAtts(node)
    
    groups = getSemanticGroups(node)
    
    pot_group_syms = [getSymsForGroup(node, groups[g]) for g in groups]
    syms, nodes_to_remove = getMaxSyms(pot_group_syms)    
    cleanNode(node, nodes_to_remove, syms)


    
##########################



def checkXRefSym(node, i, o):
    c_i = node['cubes'][i]
    c_o = node['cubes'][o]
    
    if not approx_same_dims(c_i, c_o, max(node['cubes'][0]['xd'], node['cubes'][0]['yd'], node['cubes'][0]['zd'])):
        return False

    cdir = c_o['center'] - c_i['center']
    cdir /= cdir.norm()

    if cdir.dot(node['cubes'][0]['xdir']) < 1 - SANG_THRESH:
        return False
    
    center = smp_pt(cubeToTensor(node['cubes'][0]), torch.tensor([.5, .5, .5]))

    rdir = node['cubes'][0]['xdir']
    
    c_ref = reflect_cube(c_i, center, rdir)
    
    if not approx_same_point(
        c_ref['center'],
        c_o['center'], 
        max(node['cubes'][0]['xd'], node['cubes'][0]['yd'], node['cubes'][0]['zd'])
    ):
        return False
    
    for n, at1 in node['pc_atts'][i]:
        opt = smp_pt(
            cubeToTensor(c_i),
            torch.tensor(at1)
        )
        rpt = reflect_point(opt, center, rdir)

        rat1 = [1-at1[0], at1[1], at1[2]]
        
        #if not isCloseAtt(node, rpt, o, rat1):
        if not isClosePt(node, rpt, o, rat1):
            return False

    return True


# Get all nodes that have a reflectional symmetry
# For now just do this over x-axis
# Check that all attachments are close in the opposite member
def getRefSyms(node):
    nn = len(node['cubes'])

    ref_syms = {}
    
    for ind in range(1, nn):
        if len(node['children'][ind]) > 0:
            continue
        for oind in range(1, nn):
            if ind == oind:
                continue

            if len(node['children'][oind]) > 0:
                continue
            
            if checkXRefSym(node, ind, oind):
                spec_add(ref_syms, ind, oind)

    return ref_syms


# Get all connected components that share the same symmetry, with more than one member
# (for rotational just to take the ones with the "largest" # of members)
def getRefConnComps(pot_ref_syms, node):

    groups = []
    added = set()

    for i in pot_ref_syms:
        if i in added:
            continue
        
        group = [i]
        pc_atts = set(n[0] for n in  node['pc_atts'][i])
        added.add(i)

        again = True
        while(again):
            again = False
            for j in pot_ref_syms:
                if j not in added:
                    if j in pc_atts:
                        group.append(j)
                        added.add(j)
                        pc_atts = pc_atts.union(set(n[0] for n in node['pc_atts'][j]))
                        again=True                                                
                    
        if len(group) > 1:
            groups.append([
                group,
                [pot_ref_syms[g][0] for g in group]
            ])
            
    return groups


def allConnected(a, b, node):
    prev_atts = node['pc_atts']
    
    for (_, ind), (_, oind) in zip(a, b):
        found = False
        for n, _ in prev_atts[ind]:
            if n == oind:
                found = True

        if found is False:
            return False

    return True


    


# Make sure for each attachment to a cube 'outside' of the group, that the attachment is Valid
def checkValidRefAtts(ref_ccs, node):
    valid_groups = []

    center = smp_pt(cubeToTensor(node['cubes'][0]), torch.tensor([.5, .5, .5]))
    rdir = node['cubes'][0]['xdir']
    
    for group in ref_ccs:
        isValid = True
        mems = set(group[0])
        
        for i in group[0]:
            for n, at1 in node['pc_atts'][i]:
                if n in mems:
                    continue                
                opt = smp_pt(
                    cubeToTensor(node['cubes'][i]), torch.tensor(at1)
                )
                rpt = reflect_point(opt, center, rdir)

                if isValidAtt(node, opt, n) and not isValidAtt(node, rpt, n):
                    isValid = False
                                
        if isValid:
            valid_groups.append(group)

    return valid_groups

# For any group of valid ref syms, turn it into a sub-program
# This involves:
#   - re-ordering cubes/children
#   - finding new OBB
#   - refinding all attachments
# Adding this information to node['syms']
def createRefSubPrograms(valid_ref_syms, node):
    node.pop('pc_atts')
    node.pop('attachments')

    new_progs = []
    to_remove = []
    to_head = []
    
    for group in valid_ref_syms:
        new_prog = {
            'name': 'ref_sym_sub_prog',
            'children': [],
            'children_names': [],
            'cubes': [],
        }
        to_head.append(group[0][0])
        for i in group[0]:
            to_remove.append(i)
            new_prog['children'].append(node['children'][i])
            new_prog['children_names'].append(node['children_names'][i])
            new_prog['cubes'].append(node['cubes'][i])

        bbox = jp.getOBB(new_prog['cubes'])

        new_prog['children'] = [{}] + new_prog['children']
        new_prog['children_names'] = ["bbox"] + new_prog['children_names']
        new_prog['cubes'] = [bbox] + new_prog['cubes']

        ind_to_pc, scene_geom = inter.samplePC(new_prog['cubes'], split_bbox=True)
        inters = inter.findInters(ind_to_pc, scene_geom)
        new_prog['attachments'] = inter.calcAttachments(inters, scene_geom, ind_to_pc)
        
        for o in group[1]:
            to_remove.append(o)

        new_progs.append(new_prog)
     
    for ind, new_prog in zip(to_head, new_progs):
        node['children'][ind] = new_prog
        node['cubes'][ind] = new_prog['cubes'][0]
        node['children_names'][ind] = new_prog['name']

    cube_map = {}
    count = 0

    for i in range(len(node['cubes'])):
        if i not in (set(to_remove) - set(to_head)):
            cube_map[i] = count
            count += 1
    
    node['syms'] = [[cube_map[th], 'ref_X'] for th in to_head]
    
    to_remove.sort(reverse=True)
    to_head = set(to_head)
    
    for tr in to_remove:
        if tr in to_head:
            continue
        for key in ['children', 'cubes', 'children_names']:
            node[key].pop(tr)

    to_remove = set(to_remove) - to_head

    ind_to_pc, scene_geom = inter.samplePC(node['cubes'], split_bbox=True)
    inters = inter.findInters(ind_to_pc, scene_geom)
    node['attachments'] = inter.calcAttachments(inters, scene_geom, ind_to_pc)    
    
def addSymSubPrograms(node):    
    addRefSymSubPrograms(node)    
    
def addRefSymSubPrograms(node):
    
    for child in node['children']:
        if len(child) > 0:
            addRefSymSubPrograms(child)
    
    node['pc_atts'] = getPerCubeAtts(node)
    
    pot_ref_syms = getRefSyms(node)

    if len(pot_ref_syms) == 0:
        return
    
    ref_ccs = getRefConnComps(pot_ref_syms, node)

    if len(ref_ccs) == 0:
        return
       
    valid_ref_syms = checkValidRefAtts(ref_ccs, node)

    if len(valid_ref_syms) == 0:
        return
    
    createRefSubPrograms(valid_ref_syms, node)
