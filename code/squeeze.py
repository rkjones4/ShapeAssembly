import sys
import utils
import torch
from ShapeAssembly import Program
import os
import re

device = torch.device("cuda")
fscore = utils.FScore(device)

def make_function(name, args):
    args = [str(arg) for arg in args]
    return '{}({})'.format(name, ", ".join(args))

FTHRESH = 0.05
OTHRESH = 0.15

right = torch.tensor([1.0,0.5,0.5])
left = torch.tensor([0.0,0.5,0.5])
top = torch.tensor([0.5,1.0,0.5])
bot = torch.tensor([0.5,0.0,0.5])
front = torch.tensor([0.5,0.5,1.0])
back = torch.tensor([0.5,0.5,0.0])

def orderSqueeze(lines):
    ord_lines = []
    q = []
    at_q = []
    grounded = set(['bbox'])
    P = Program()
    for line in lines:
        if "attach(" in line:            
            parse = P.parseAttach(line)
            if parse[1] in grounded:
                grounded.add(parse[0])
                ord_lines.append(line)
            else:
                at_q.append((parse[0], parse[1], line))
        elif "squeeze(" in line:
            parse = P.parseSqueeze(line)
            if parse[1] in grounded and parse[2] in grounded:
                ord_lines.append(line)
                grounded.add(parse[0])
            else:
                q.append((parse[0], parse[1], parse[2], line))

        torm = []
        for i, (c, o1, o2, sl) in enumerate(q):
            if o1 in grounded and o2 in grounded:
                grounded.add(c)
                torm.append(i)
                ord_lines.append(sl)

        torm.sort(reverse=True)
        for i in torm:
            q.pop(i)

        atorm = []
        for i, (c, o, al) in enumerate(at_q):
            if o in grounded:
                grounded.add(c)
                atorm.append(i)
                ord_lines.append(al)

        atorm.sort(reverse=True)
        for i in atorm:
            at_q.pop(i)


            
    assert len(q) == 0, 'Some squeezes are ungrounded'
    assert len(at_q) == 0, 'Some attaches are ungrounded'

    return ord_lines
    
def getOppFace(face):
    of = {
        'right': 'left',
        'left': 'right',
        'top': 'bot',
        'bot': 'top',
        'front': 'back',
        'back': 'front',
    }
    return of[face]

def getFace(at1, at2, ocube):
    if ocube == 'bbox':
        at2[1] = 1-at2[1]

    for name, face, oface, ind, oind in [
            ('right', right, left, 0, torch.tensor([1,2]).long()),
            ('left', left, right, 0, torch.tensor([1,2]).long()),
            ('top', top, bot, 1, torch.tensor([0,2]).long()),
            ('bot', bot, top, 1, torch.tensor([0,2]).long()),
            ('front', front, back, 2, torch.tensor([0,1]).long()),
            ('back', back, front, 2, torch.tensor([0,1]).long()),
    ]:        
        if (at1 - face).abs().sum() < FTHRESH and (at2[ind] - oface[ind]).abs().sum() < OTHRESH:
            return name, at2[oind]

    return None, None

def spec_add(m, k, v):
    if k not in m:
        m[k] = v
    else:
        m[k].update(v)
        
def getSqueezeProg(prog):
    P = Program()
    cuboid_lines = []
    sym_lines = []
    attach_lines = []
    
    face_info = {}
    
    for line in prog:
        if "Cuboid(" in line:
            cuboid_lines.append(line)
        elif "reflect(" in line or "translate(" in line:
            sym_lines.append(line)
        elif "attach(" in line:
            attach_lines.append(line)            
            parse = P.parseAttach(line)
            face, uv = getFace(torch.stack(parse[2:5]), torch.stack(parse[5:8]), parse[1])            
            if face is not None:
                spec_add(face_info, parse[0], {face : (uv, parse[1])})
                

    squeeze_lines = []

    for att_line in attach_lines:
        parse = P.parseAttach(att_line)            
        face, uv = getFace(torch.stack(parse[2:5]), torch.stack(parse[5:8]), parse[1])
        
        # not a face att so just skip
        if face is None:
            squeeze_lines.append(att_line)
            continue
                
        oface = getOppFace(face)
        cube = parse[0]
        
        # if this is missing it must have been removed previously by a squeeze attach
        if face not in face_info[cube]:
            continue
        
        if oface in face_info[cube] and \
           (face_info[cube][oface][0] - uv).abs().sum() < OTHRESH:
            squeeze_lines.append(
                make_function('\tsqueeze', (cube, parse[1], face_info[cube][oface][1], face, uv[0].item(), uv[1].item()))
            )

            face_info[cube].pop(oface)
            face_info[cube].pop(face)

        else:
            squeeze_lines.append(att_line)

    ord_squeeze_lines = orderSqueeze(squeeze_lines)

    return cuboid_lines + ord_squeeze_lines + sym_lines

def addSqueeze(node):
    raw_prog = node.pop('prog')
    node['prog'] = getSqueezeProg(raw_prog)

    nsq = 0.
    
    for line in node['prog']:
        if 'squeeze' in line:
            nsq += 1.
    
    for c in node['children']:
        if len(c) > 0:
            nsq += addSqueeze(c)

    return nsq

