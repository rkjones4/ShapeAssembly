import torch
import torch.nn as nn
import numpy as np
import generate as gen
import ShapeAssembly as ex
from copy import deepcopy

MAX_CUBES = 10
PREC = 4

def loadObj(infile):
    tverts = []
    ttris = []
    with open(infile) as f:
        for line in f:
            ls = line.split()
            if len(ls) == 0:
                continue
            if ls[0] == 'v':
                tverts.append([
                    float(ls[1]),
                    float(ls[2]),
                    float(ls[3])
                ])
            elif ls[0] == 'f':
                ttris.append([
                    int(ls[1].split('//')[0])-1,
                    int(ls[2].split('//')[0])-1,
                    int(ls[3].split('//')[0])-1
                ])

    return tverts, ttris


def vector_cos(norm1, norm2):
    norm1 = np.asarray(norm1)
    norm2 = np.asarray(norm2)
    dot = np.dot(norm1, norm2)
    magnitude = np.linalg.norm(norm1) * np.linalg.norm(norm2)
    if magnitude == 0.:
        return 0.
    return dot / float(magnitude)


def orientProps(center, xd, yd, zd, xdir, ydir, zdir):

    rt = np.asarray([1., 0., 0.])
    up = np.asarray([0., 1., 0.])
    fwd = np.asarray([0., 0., 1.])

    l = [
        (xdir, xd, 0),
        (ydir, yd, 1),
        (zdir, zd, 2),
        (-1 * xdir, xd, 3),                
        (-1 * ydir, yd, 4),        
        (-1 * zdir, zd, 5)
    ]

    rtdir, rtd, rind = sorted(deepcopy(l), key=lambda x: vector_cos(rt, x[0]))[-1]

    if rind >= 3:
        l.pop(rind)
        l.pop((rind+3)%6)    
    else:
        l.pop((rind+3)%6)    
        l.pop(rind)
        
    for i in range(0, 4):
        p_ind = l[i][2]
        if p_ind > max(rind, (rind+3)%6):
            l[i] = (l[i][0], l[i][1], l[i][2] - 2)
        elif p_ind > min(rind, (rind+3)%6):
            l[i] = (l[i][0], l[i][1], l[i][2] - 1)
                
    updir, upd, upind = sorted(deepcopy(l), key=lambda x: vector_cos(up, x[0]))[-1]

    if upind >= 2:    
        l.pop(upind)
        l.pop((upind+2)%4)    
    else:
        l.pop((upind+2)%4)    
        l.pop(upind)
        
    fwdir, fwd, _ = sorted(l, key=lambda x: vector_cos(fwd, x[0]))[-1]
        
    return {
        'center': torch.tensor(center).float(),
        'xd': torch.tensor(rtd).float(),
        'yd': torch.tensor(upd).float(),
        'zd': torch.tensor(fwd).float(),
        'xdir': torch.tensor(rtdir).float(),
        'ydir': torch.tensor(updir).float(),
        'zdir': torch.tensor(fwdir).float()
    }

def locallyClean(prog):
    cube_count = -1
    switches = []
    for line in prog:
        if 'Cuboid' in line:
            if 'Program_' in line:
                switches.append((
                    f'cube{cube_count}', line.split()[0]
                ))
            
            cube_count += 1
            
    for a, b in switches:
        prog = [line.replace(b,a) for line in prog]

    clines = []
    
    P = ex.Program()
    for line in prog:
        if "Cuboid(" in line:
            parse = P.parseCuboid(line)
            name = parse[0]                                
            x = float(max(parse[1].item(), 0.01))
            y = float(max(parse[2].item(), 0.01))
            z = float(max(parse[3].item(), 0.01))
            aligned = str(parse[4])
            clines.append("\t" + gen.assign(
                name, gen.make_function('Cuboid', [x,y,z,aligned])))            
            
        if "attach(" in line:
            parse = P.parseAttach(line)
            clines.append("\t" + gen.make_function(
                "attach",
                [parse[0], parse[1]] + [
                    torch.clamp(co, 0.0, 1.0).item() for co in parse[2:]
                ]
            ))

        if "squeeze(" in line:
            parse = P.parseSqueeze(line)
            clines.append("\t" + gen.make_function(
                "squeeze",
                [parse[0], parse[1], parse[2], parse[3]] + [max(min(co, 1.0), 0.0) for co in parse[4:]]
            ))
            
        if "reflect(" in line:
            parse = P.parseReflect(line)
            clines.append("\t" + gen.make_function(
                "reflect",
                [parse[0], parse[1]]
            ))

        if "translate(" in line:
            parse = P.parseTranslate(line)
            clines.append("\t" + gen.make_function(
                "translate",
                [
                    parse[0],
                    parse[1],
                    max(int(parse[2]), 1),
                    float(min(max(parse[3], 0.0), 1.0))
                ]
            ))
            
                
    return clines
            
def fillHP(name, progs, children):
    hp = {'name': name}
    hp['prog'] = locallyClean(progs[name])
    hp['children'] = [
        fillHP(cn, progs, children) if cn is not None else {} for cn in children[name]
    ]
    if len(hp['children']) > MAX_CUBES:
        return {}
    return hp


def loadHPFromFile(progfile):

    progs = {}
    children = {}
    
    prog_num = -1
    cur_prog = []
    cur_children = []
    
    with open(progfile) as f:
        for line in f:
            ls = line.split()
            
            if ls[0] == 'Assembly':
                prog_num = int(ls[1].split('_')[1])

            elif ls[0] == '}':
                
                progs[prog_num] = cur_prog
                children[prog_num] = cur_children
                cur_prog = []
                cur_children = []
                
            elif 'Cuboid' in line:
                if 'Program_' in line:
                    cur_children.append(int(ls[0].split('_')[1]))
                else:
                    cur_children.append(None)

                cur_prog.append(line[1:-1])

            elif 'attach' in line:
                cur_prog.append(line[1:-1])

            elif 'reflect' in line:
                cur_prog.append(line[1:-1])

            elif 'translate' in line:
                cur_prog.append(line[1:-1])

            elif 'squeeze' in line:
                cur_prog.append(line[1:-1])
                
    return fillHP(0, progs, children)


def locallyNormClean(prog, max_val):
    cube_count = -1
    switches = []
    for line in prog:
        if 'Cuboid' in line:
            if 'Program_' in line:
                switches.append((
                    f'cube{cube_count}', line.split()[0]
                ))
            
            cube_count += 1
            
    for a, b in switches:
        prog = [line.replace(b,a) for line in prog]

    clines = []
    
    P = ex.Program()
    for line in prog:
        if "Cuboid(" in line:
            parse = P.parseCuboid(line)
            name = parse[0]                                
            x = float(min(max(parse[1].item() / max_val, 0.01), 1.0))
            y = float(min(max(parse[2].item() / max_val, 0.01), 1.0))
            z = float(min(max(parse[3].item() / max_val, 0.01), 1.0))

            clines.append("\t" + gen.assign(
                name, gen.make_function('Cuboid', [x,y,z])))            
            
        if "attach(" in line:
            parse = P.parseAttach(line)
            clines.append("\t" + gen.make_function(
                "attach",
                [parse[0], parse[1]] + [
                    torch.clamp(co, 0.0, 1.0).item() for co in parse[2:]
                ]
            ))

        if "reflect(" in line:
            parse = P.parseReflect(line)
            clines.append("\t" + gen.make_function(
                "reflect",
                [parse[0], parse[1]]
            ))

        if "translate(" in line:
            parse = P.parseTranslate(line)
            clines.append("\t" + gen.make_function(
                "translate",
                [
                    parse[0],
                    parse[1],
                    max(int(parse[2]), 1),
                    float(min(max(parse[3], 0.0), 1.0))
                ]
            ))
                
    return clines
            
def fillNormHP(name, progs, children, max_val = None):
    hp = {'name': name}
    if max_val is None:
        max_val = max([float(d) for d in progs[name][0][:-1].split('(')[1].split(',')])

    hp['prog'] = locallyNormClean(progs[name], max_val)
    hp['children'] = [
        fillNormHP(cn, progs, children, max_val) if cn is not None else {} for cn in children[name]
    ]
    if len(hp['children']) > MAX_CUBES:
        return {}
    return hp

def loadNormHPFromFile(progfile):

    progs = {}
    children = {}
    
    prog_num = -1
    cur_prog = []
    cur_children = []
    
    with open(progfile) as f:
        for line in f:
            ls = line.split()
            
            if ls[0] == 'Assembly':
                prog_num = int(ls[1].split('_')[1])

            elif ls[0] == '}':
                
                progs[prog_num] = cur_prog
                children[prog_num] = cur_children
                cur_prog = []
                cur_children = []
                
            elif 'Cuboid' in line:
                if 'Program_' in line:
                    cur_children.append(int(ls[0].split('_')[1]))
                else:
                    cur_children.append(None)

                cur_prog.append(line[1:-1])

            elif 'attach' in line:
                cur_prog.append(line[1:-1])

            elif 'reflect' in line:
                cur_prog.append(line[1:-1])

            elif 'translate' in line:
                cur_prog.append(line[1:-1])
                
    return fillNormHP(0, progs, children)
                
def getHierProgLines(root):
    prog_count = 0
    root["prog_num"] = prog_count
    lines = []
    q = [root]
    P = ex.Program()
    while(len(q) > 0):

        node = q.pop(0)
        
        lines.append(f"Assembly Program_{node['prog_num']}" +" {")    

        NAME_DICT = {}
        
        c = 0
        for line in node["prog"]:
            if "Cuboid(" in line:
                parse = P.parseCuboid(line)
                if len(node["children"][c]) > 0:                
                    prog_count += 1
                    name = f"Program_{prog_count}"
                    node["children"][c]["prog_num"] = prog_count
                else:
                    name = parse[0]

                NAME_DICT[parse[0]] = name
                    
                x = round(float(parse[1]), PREC)
                y = round(float(parse[2]), PREC)
                z = round(float(parse[3]), PREC)
                aligned = str(parse[4])
                lines.append("\t" + gen.assign(
                    name, gen.make_function('Cuboid', [x,y,z,aligned]))
                )               
                c += 1
            
            if "attach(" in line:
                parse = P.parseAttach(line)
                lines.append("\t" + gen.make_function(
                    "attach",
                    [NAME_DICT[parse[0]], NAME_DICT[parse[1]]] + [round(co.item(), PREC) for co in parse[2:]]
                ))

            if "reflect(" in line:
                parse = P.parseReflect(line)
                lines.append("\t" + gen.make_function(
                    "reflect",
                    [NAME_DICT[parse[0]], parse[1]]
                ))

            if "translate(" in line:
                parse = P.parseTranslate(line)
                lines.append("\t" + gen.make_function(
                    "translate",
                    [NAME_DICT[parse[0]], parse[1], int(parse[2]), round(float(parse[3]), PREC)]
                ))

            if "rotate(" in line:
                parse = P.parseRotate(line)
                lines.append("\t" + gen.make_function(
                    "rotate",
                    [NAME_DICT[parse[0]], int(parse[1]), round(float(parse[2]), PREC)]
                ))

            if "squeeze(" in line:
                parse = P.parseSqueeze(line)
                lines.append("\t" + gen.make_function(
                    "squeeze",
                    [NAME_DICT[parse[0]], NAME_DICT[parse[1]], NAME_DICT[parse[2]], parse[3]] + [round(co, PREC) for co in parse[4:]]
                ))
        
        lines.append("}")
        
        for c in node["children"]:
            if c is not None and len(c) > 0:
                if "prog_num" in c:
                    q.append(c)

    return lines            

def writeHierProg(hier_prog, outfile):
    lines = getHierProgLines(hier_prog)
    with open(outfile, 'w') as f:
        for line in lines:
            f.write(f'{line}\n')


def log_print(s, of):
    with open(of, 'a') as f:
        f.write(f"{s}\n")
    print(s)

    
def writeObj(verts, faces, outfile):
    faces = faces.clone()
    faces += 1
    with open(outfile, 'w') as f:
        for a, b, c in verts.tolist():
            f.write(f'v {a} {b} {c}\n')

        for a, b, c in faces.tolist():
            f.write(f"f {a} {b} {c}\n")

def writePC(pc, fn):
    with open(fn, 'w') as f:
        for a,b,c,_,_,_ in pc:
            f.write(f'v {a} {b} {c} \n')

def writeSPC(pc, fn):
    with open(fn, 'w') as f:
        for a,b,c in pc:
            f.write(f'v {a} {b} {c} \n')

def face_areas_normals(faces, vs):
    face_normals = torch.cross(
        vs[:, faces[:, 1], :] - vs[:, faces[:, 0], :],
        vs[:, faces[:, 2], :] - vs[:, faces[:, 1], :],
        dim=2,
    )
    face_areas = torch.norm(face_normals, dim=2) + 1e-8
    face_normals = face_normals / face_areas[:, :, None]
    face_areas = 0.5 * face_areas
    return face_areas, face_normals


def sample_surface(faces, vs, count, return_normals=True):
    """
    sample mesh surface
    sample method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Args
    ---------
    vs: vertices (batch x nvs x 3d coordinate)
    faces: triangle faces (torch.long) (num_faces x 3)
    count: number of samples
    Return
    ---------
    samples: (count, 3) points in space on the surface of mesh
    face_index: (count,) indices of faces for each sampled point
    """
    if torch.isnan(faces).any() or torch.isnan(vs).any():
        assert False, 'saw nan in sample_surface'

    device = vs.device
    bsize, nvs, _ = vs.shape
    area, normal = face_areas_normals(faces, vs)
    area_sum = torch.sum(area, dim=1)

    assert not (area <= 0.0).any().item(), "Saw negative probability while sampling"
    assert not (area_sum <= 0.0).any().item(), "Saw negative probability while sampling"
    assert not (area > 1000000.0).any().item(), "Saw inf"
    assert not (area_sum > 1000000.0).any().item(), "Saw inf"

    dist = torch.distributions.categorical.Categorical(probs=area / (area_sum[:, None]))
    face_index = dist.sample((count,))

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vs[:, faces[:, 0], :]
    tri_vectors = vs[:, faces[:, 1:], :].clone()
    tri_vectors -= tri_origins.repeat(1, 1, 2).reshape((bsize, len(faces), 2, 3))

    # pull the vectors for the faces we are going to sample from
    face_index = face_index.transpose(0, 1)
    face_index = face_index[:, :, None].expand((bsize, count, 3))
    tri_origins = torch.gather(tri_origins, dim=1, index=face_index)
    face_index2 = face_index[:, :, None, :].expand((bsize, count, 2, 3))
    tri_vectors = torch.gather(tri_vectors, dim=1, index=face_index2)

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = torch.rand(count, 2, 1, device=vs.device, dtype=tri_vectors.dtype)

    # points will be distributed on a quadrilateral if we use 2x [0-1] samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(dim=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = torch.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths[None, :]).sum(dim=2)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    if return_normals:
        samples = torch.cat((samples, torch.gather(normal, dim=1, index=face_index)), dim=2)
        return samples
    else:
        return samples
