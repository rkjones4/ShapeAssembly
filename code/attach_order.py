import numpy as np
import re
import networkx as nx
import itertools
import sys
from copy import deepcopy, copy
import ShapeAssembly as ex
import torch

MAX_PERMS = 10
THRESH = .99
MAX_DUAL_VARS = 5

AA_THRESH = .995
SQUARE_THRESH = .1

BOT_TO_TOP = False

def isSquare(a, b, c):
    v = (a - b).abs()/max(a.item(), b.item(), c.item()) < SQUARE_THRESH
    return v

def isSpecAxisAligned(cdir, axis):
    cdir /= cdir.norm()

    if cdir.dot(axis) >= AA_THRESH:
        return True
    else:
        return False


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


def getDistToBBox(attachments):
    g = nx.Graph()
    max_depth = 0
    
    for (i1, i2, at1, _) in attachments:
        if BOT_TO_TOP:
            assert i2 != 0, 'Intersting fix this'
            if (i1 == 0 and at1[1] > .5):
                continue                    
        
        g.add_node(i1)
        g.add_node(i2)
        g.add_edge(i1, i2)
        
    dists = {}
    for (i1,i2, _, _) in attachments:
        dists[i1] = nx.shortest_path_length(g, 0, i1)
        dists[i2] = nx.shortest_path_length(g, 0, i2)
        max_depth = max(max(dists[i1], dists[i2]), max_depth)
        
    return dists, max_depth


def maybe_flip_pair(m1, m2):
    if m1[0] > m2[0]:
        return (m1, m2)
    else:
        return (m2, m1)


def getCuboidFromProps(gt, name):
    c = ex.Cuboid(name)
    c.dims = torch.stack((gt['xd'], gt['yd'], gt['zd']))
    c.rfnorm = gt['xdir']
    c.tfnorm = gt['ydir']
    c.ffnorm = gt['zdir']
    c.pos = gt['center']
    return c


# takes in a set of members, the ground-truth, the previous members, and the face attpoints of the program
# returns the ordering that recreates the geometry best, then ordered semantically
# todo still might be un-needed
def getBestOrderForCuboid(members, ground_truth, prev_members):
    cube = members[0][0][1]
    acubes = [m[1][1] for m in members]
    gt = ground_truth[cube]

    target = getCuboidFromProps(gt, 'gt').getParams()
    
    agt = [ground_truth[c] for c in acubes]
    
    aps = [
        ((m[0]), (m[1]))
        for m in members
    ]
    
    constraints = []
    
    for ((l1, c1, ap1), (l2, c2, ap2)) in prev_members:
        if c1 == cube:
            constraints.append(((l1,c1, ap1), (l2, c2, ap2)))

        if c2 == cube:
            constraints.append(((l2,c2, ap2), (l1, c1, ap1)))
            
    perms = list(itertools.permutations(aps))
    
    P = ex.Program()

    P.cuboids['bbox'] = getCuboidFromProps(ground_truth[0], 'bbox')
    
    for ac, agt in zip(acubes, agt):
        if ac == 0:
            continue
        
        P.cuboids[ac] = getCuboidFromProps(agt, ac)
        

    for c in constraints:        
        c_ind = c[1][1] if c[1][1] != 0 else 'bbox'
        P.cuboids[c_ind] = getCuboidFromProps(ground_truth[c[1][1]], c_ind)                
        
    best_perm = None
    best_match = 1e8
        
    for i, perm in enumerate(perms):
        cub = ex.Cuboid(cube)
        cub.dims = torch.tensor([gt['xd'], gt['yd'], gt['zd']], dtype = torch.float)
        P.cuboids[cube] = cub
        
        for c in constraints:
            c_ind = c[1][1] if c[1][1] != 0 else 'bbox'
            c_pos = P.cuboids[c_ind].getPos(
                torch.tensor(c[1][2][0]),
                torch.tensor(c[1][2][1]),
                torch.tensor(c[1][2][2])
            )
            c_fap = ex.AttPoint(
                P.cuboids[c[0][1]],
                torch.tensor(c[0][2][0]),
                torch.tensor(c[0][2][1]),
                torch.tensor(c[0][2][2])
            )
            P.attach(c_fap, c_pos, 'd')

        for a in perm:
            a_ind = a[1][1] if a[1][1] != 0 else 'bbox'
            
            a_pos = P.cuboids[a_ind].getPos(
                torch.tensor(a[1][2][0]),
                torch.tensor(a[1][2][1]),
                torch.tensor(a[1][2][2])
            )
            
            a_fap = ex.AttPoint(
                P.cuboids[a[0][1]],
                torch.tensor(a[0][2][0]),
                torch.tensor(a[0][2][1]),
                torch.tensor(a[0][2][2])
            )

            P.attach(a_fap, a_pos, 'd')

        match = (cub.getParams() - target).abs().sum()
        
        if match < best_match:
            best_match = match
            best_perm = perm
            
        P.cuboids.pop(cube)

    free_members = []
    set_members = []
    
    for (m0, m1) in best_perm:
        set_members.append((m0,m1))

    for (m0, m1) in aps:
        if (m0, m1) not in set_members:
            free_members.append((m0, m1))
                    
    free_members = [getMemRank([f]) for f in free_members]
    free_members.sort()
    free_members = [m[0] for _, m in free_members]
    return set_members + free_members


right = torch.tensor([1.0,0.5,0.5])
left = torch.tensor([0.0,0.5,0.5])
top = torch.tensor([0.5,1.0,0.5])
bot = torch.tensor([0.5,0.0,0.5])
front = torch.tensor([0.5,0.5,1.0])
back = torch.tensor([0.5,0.5,0.0])

def getFaceIndex(ap):
    if (ap-right).abs().sum() < .05:
        return 0
    if (ap-left).abs().sum() < .05:
        return 1
    if (ap-top).abs().sum() < .05:
        return 2
    if (ap-bot).abs().sum() < .05:
        return 3
    if (ap-front).abs().sum() < .05:
        return 4
    if (ap-back).abs().sum() < .05:
        return 5
    return 6


# takes in a list of members
def getMemRank(ml):
 
    rank = torch.ones(10) * 1e8
    
    for ((_, c1, ap1), (_, c2, ap2)) in ml:
        n_rank = torch.tensor([
            c1,
            c2,
            getFaceIndex(torch.tensor(ap1)),
            getFaceIndex(torch.tensor(ap2)),
            ap1[0],
            ap2[0],
            ap1[1],
            ap2[1],
            ap1[2],
            ap2[2]
        ])
        rank = torch.stack((
            rank, n_rank
        )).min(dim=0).values
        
    return (rank.tolist(), ml)
    
# fm is a list of free members, gm is a list of grouped members in a specific order
def orderMembers(fm, gm, keep_gm_order):
    # make two stacks, one for fm, one for gm
    # keep next item on top by semantic order
    # to make lines, just keep popping of the top of each list, and adding to queue
    
    fm_stack = [getMemRank([f]) for f in fm]
    fm_stack.sort(reverse = True)

    gm_stack = [getMemRank(g) for g in gm]

    if not keep_gm_order:
        gm_stack.sort(reverse = True)
    else:
        gm_stack.reverse()
        
    order = []

    if len(fm_stack) > 0 and len(gm_stack) > 0:
                
        while len(fm_stack) > 0 and len(gm_stack) > 0:

            fmr, t_fm = fm_stack[-1]
            gmr, t_gm = gm_stack[-1]
            
            if fmr < gmr:
                for m in t_fm:
                    order.append(m)
                fm_stack.pop()
                
            else:
                for m in t_gm:
                    order.append(m)
                gm_stack.pop()
                    
    while(len(fm_stack) > 0):
        _, ml = fm_stack.pop()
        for m in ml:
            order.append(m)

    while(len(gm_stack) > 0):
        _, ml = gm_stack.pop()
        for m in ml:
            order.append(m)
        
    return order


def findDualBestOrder(members, prev_members, ground_truth):

    c_set = set()

    for ((_,c1,_),(_,c2,_)) in members:
        c_set.add(c1)
        c_set.add(c2)

    g = nx.Graph()
    att_counts = {}

    for c in c_set:
        att_counts[c] = 0.
        g.add_node(c)
        
    prev_atts = {}
    
    for ((_,c1, ap1), (_, c2, ap2)) in prev_members:
        if c1 in att_counts:
            att_counts[c1] += 1.
        if c1 in prev_atts:
            prev_atts[c1].append([c2, ap1, ap2])
        else:
            prev_atts[c1] = [[c2, ap1, ap1]]
                   
    is_aa = {c: isAxisAligned(ground_truth[c], ground_truth[0]) for c in c_set}
        
    if len(members) == 1:        
        if is_aa[members[0][0][1]] or att_counts[members[0][0][1]] > 1:
            return [(members[0][1], members[0][0])]
        elif is_aa[members[0][1][1]] or att_counts[members[0][1][1]] > 1:
            return members

        if getFaceIndex(torch.tensor(members[0][0][2])) != 6:
            return members
        elif getFaceIndex(torch.tensor(members[0][1][2])) != 6:
            return [(members[0][1], members[0][0])]

        if members[0][0][1] < members[0][1][1]:
            return members
        else:
            return [(members[0][1], members[0][0])]
        
    for m in members:
        ((_, c1, _), (_, c2, _)) = m                
        g.add_edge(c1, c2, mem = m)
            
    group_members = []

    while len(g.edges) > 0:

        # Check if any leaves that are COA

        snodes = list(g.nodes())
        snodes.sort()

        found_coa_leaf = False
                
        for n in snodes:
            if g.degree(n) == 1 and (is_aa[n] or att_counts[n] > 2):
                par = list(g[n].keys())[0]
                mem = g[n][par]['mem']
                
                if mem[0][1] == n:
                    nmem = (mem[1], mem[0])
                else:
                    nmem = mem

                group_members.append([nmem])
                g.remove_node(n)
                att_counts[par] += 1.

                found_coa_leaf = True
                break

        if found_coa_leaf:
            continue
                        
        # Check if any coa nodes

        found_coa_node = False
        
        for n in snodes:
            if is_aa[n] or att_counts[n] > 2:
                neighs = list(g[n].keys())

                cur_group = []
                
                for i in range(len(neighs)):
                    mem = g[n][neighs[i]]['mem']

                    if mem[0][1] == n:
                        nmem = (mem[1], mem[0])
                    else:
                        nmem = mem

                    cur_group.append(nmem)
                    att_counts[neighs[i]] += 1.

                group_members.append(cur_group)
                g.remove_node(n)
                found_coa_node = True
                break

        if found_coa_node:
            continue
        
        degs = [(d, n) for n, d in g.degree()]
        degs.sort()

        n = degs[0][1]

        neighs = list(g[n].keys())

        cur_group = []
                
        for i in range(len(neighs)):
            mem = g[n][neighs[i]]['mem']

            if mem[0][1] != n:
                nmem = (mem[1], mem[0])
            else:
                nmem = mem
                
            cur_group.append(nmem)
            att_counts[neighs[i]] += 1.

        if len(cur_group) == 0:
            pass            
        elif len(cur_group) == 1:
            group_members.append(cur_group)            
        else:
            group_members.append(getBestOrderForCuboid(cur_group, ground_truth, prev_members))
            
        g.remove_node(n)        
        
    return orderMembers([], group_members, True)
    

def findBestOrder(members, prev_members, ground_truth):
    
    
    i = members[0][0][0]
    j = members[0][1][0]

    if i == j:
        return findDualBestOrder(members, prev_members, ground_truth)    

    if len(members) == 1:
        return members
    
    counts = {}

    for ((_, c1, _), (_, c2, _)) in members:
        if c1 in counts:
            counts[c1] += 1
        else:
            counts[c1] = 1
    
    free_members = []
    case2_members = {}
    
    om = orderMembers(members, [], False)
    
    return om


def filterTopAtts(att_pairs):
    atts = []
    end_pairs = []
    for ((d1, c1, ap1), (d2, c2, ap2)) in att_pairs:
        if c1 == 0 and ap1[1] > .5:
            end_pairs.append(((-1, c2,ap2), (-1, c1, ap1)))
        else:
            atts.append(((d1, c1, ap1), (d2, c2, ap2)))

    end_pairs.sort()
            
    return end_pairs, atts
    
    
def getAttOrder(attachments, ground_truth):

    dists, max_depth = getDistToBBox(attachments)
    
    att_pairs = [
        ((dists[c1], c1, ap1), (dists[c2], c2, ap2)) for
        (c1, c2, ap1, ap2) in attachments
    ]

    end_pairs = []

    if BOT_TO_TOP:
        end_pairs, att_pairs = filterTopAtts(att_pairs)
    
    att_pairs = [maybe_flip_pair(m1, m2) for (m1, m2) in att_pairs]
    
    groups = []

    for i in range(1, max_depth+1):
        for j in [1, 0]:
            members = []
            for pair in att_pairs:
                if pair[0][0] == i and pair[1][0] == i-j:                    
                    members.append(pair)
            if len(members) > 0:
                groups.append(members)

    order = []
    
    for gi in range(len(groups)):
        members = groups[gi]
        prev_members = []
        if gi-1 >= 0:
            prev_members = groups[gi-1]
        bo = findBestOrder(members, prev_members, ground_truth)
        order += bo

    order = order + end_pairs
                
    assert len(order) == len(attachments), 'something has gone terribly wrong'
    
    return order
        
