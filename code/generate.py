import numpy as np
import re
import json_parse as jp
import itertools
import sys
from copy import deepcopy, copy
import attach_order as att
import torch
import ShapeAssembly as sa
import utils

# Round Details to this Precision
PREC = 3
VERBOSE = False
DO_SYM = True

MAX_ATTS = 2
CLOSE_THRESH = 0.1
AA_THRESH = 0.995

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

def make_cuboid_function(cube, aligned):
    
    x_dim = cube['xd']
    y_dim = cube['yd']
    z_dim = cube['zd']

    x_dim = round(x_dim.item(), PREC)
    z_dim = round(z_dim.item(), PREC)
    y_dim = round(y_dim.item(), PREC)
    
    return make_function('Cuboid', args=[x_dim, y_dim, z_dim, str(aligned)])

def make_function(name, args):
    args = [str(arg) for arg in args]
    return '{}({})'.format(name, ", ".join(args))

def assign(var_name, value):
    return '{} = {}'.format(var_name, value)

def getSymLines(node, names):
    sym_lines = []

    for sym in node['syms']:
        name = names[sym[0]]
        st = sym[1].split('_')[0]
        axis = sym[1].split('_')[1]
        if st == 'ref':
            sym_lines.append(make_function(
                'reflect',
                [name, axis]
            ))

        elif st == 'tr':
            n = sym[2]
            s = round(sym[3], PREC)
            sym_lines.append(make_function(
                'translate',
                [name, axis, n, s]
            ))                    

        elif st == 'rot':
            n = sym[3]
            ang = sym[2]
            sym_lines.append(make_function(
                'rotate',
                [name, n, ang]
            ))
            
    return sym_lines

def getCubeIndex(c):
    if 'cube' in c:
        return int(c[4:]) + 1 
    elif 'bbox' in c:
        return 0

def spec_add(m, k, v):
    if k in m:
        m[k].append(v)
    else:
        m[k] = [v]

right = torch.tensor([1.0,0.5,0.5])
left = torch.tensor([0.0,0.5,0.5])
top = torch.tensor([0.5,1.0,0.5])
bot = torch.tensor([0.5,0.0,0.5])
front = torch.tensor([0.5,0.5,1.0])
back = torch.tensor([0.5,0.5,0.0])

def isFaceAtt(face):
    if (face-right).abs().sum() < .05:
        return True
    if (face-left).abs().sum() < .05:
        return True
    if (face-top).abs().sum() < .05:
        return True
    if (face-bot).abs().sum() < .05:
        return True
    if (face-front).abs().sum() < .05:
        return True
    if (face-back).abs().sum() < .05:
        return True
    return False

        
def filterAttLines(att_lines, caligned):
    P = sa.Program()
    fatt_lines = []

    prev_atts = {}

    for line in att_lines:
        parse = P.parseAttach(line)

        c1 = getCubeIndex(parse[0])
        c2 = getCubeIndex(parse[1])

        at1 = torch.stack(parse[2:5])
        at2 = torch.stack(parse[5:])

        is_valid = True
        
        if c1 in prev_atts:
            for patt in prev_atts[c1]:
                dist = (at1-patt).abs().sum()
                if dist < CLOSE_THRESH:
                    is_valid = False
        
        if not is_valid:
            continue

        do_flip = False
        if c1 in prev_atts and caligned[c1] and not caligned[c2]:
            do_flip = True

        if c1 in prev_atts and c2 in prev_atts and caligned[c1] and caligned[c2]:
            if isFaceAtt(at2) and not isFaceAtt(at1):
                do_flip = True
                     
        if do_flip:
            line = make_function(
                'attach',
                [parse[1], parse[0]] + \
                [t.item() for t in list(parse[5:])] + \
                [t.item() for t in list(parse[2:5])]
            )
            c_ind = c2
        else:
            c_ind = c1
        
        if c_ind in prev_atts and len(prev_atts[c_ind]) >= MAX_ATTS:
            continue                
        
        fatt_lines.append(line)        
        spec_add(prev_atts, c1, at1)
        spec_add(prev_atts, c2, at2)
        
    return fatt_lines
    
    

def generate_program_for_node(node):

    c_lines = []
    ap_lines = []
    att_pairs = []

    attachment_count = {}

    names = []
    caligned = []
    bbox = node['cubes'][0]
    
    for i, cube in enumerate(node['cubes']):
        if i == 0:
            name = 'bbox'
        else:
            name = f'cube{i-1}'            

        names.append(name)
        aligned = isAxisAligned(cube, bbox)
        caligned.append(aligned)
        c_lines.append(assign(name, make_cuboid_function(cube, aligned)))

    att_order = att.getAttOrder(
        node['attachments'],
        deepcopy(node['cubes'])
    )

    att_lines = []

    for ((_, c1, ap1), (_, c2, ap2)) in att_order:               
        att_lines.append(make_function(
            'attach', 
            [names[c1], names[c2]] + \
            [round(e, PREC) for e in ap1] + \
            [round(e, PREC) for e in ap2]
        ))

    
    fatt_lines = filterAttLines(
        att_lines,
        caligned
    )

    sym_lines = []
    if DO_SYM and 'syms' in node:
        if len(node['syms']) > 0:
            sym_lines = getSymLines(node, names)
        
    return c_lines + fatt_lines + sym_lines

def generate_program(node):
    
    node['prog'] = generate_program_for_node(node)    
    
    if VERBOSE:
        print(f"Program for {node['name']}")

        for line in node['prog']:
            print(f'"{line}",')
    
    for child in node['children']:
        if len(child) > 0:
            generate_program(child)
    

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
