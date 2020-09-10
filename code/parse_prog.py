import torch

from ShapeAssembly import *
from generate import *

INPUT_DIM = 63

def getCubeIndex(c):
    if 'cube' in c:
        return int(c[4:]) + 1 
    elif 'bbox' in c:
        return 0

def indToCube(c):
    if c ==  0:
        return 'bbox'
    else:
        return 'cube' + str(c-1)
    
def getAxisIndex(a):
    if a == 'X':
        return 0
    elif a == 'Y':
        return 1
    elif a == 'Z':
        return 2
    else:
        assert False, 'bad axis'

def indToAxis(i):
    if i == 0:
        return 'X'
    elif i == 1:
        return 'Y'
    elif i == 2:
        return 'Z'


def getFaceIndex(f):
    fm = {
        'right': 0,
        'left': 1,
        'top': 2,
        'bot': 3,
        'front': 4,
        'back': 5,
    }
    return fm[f]

def indToFace(i):
    fm = {
        0 : 'right',
        1 : 'left',
        2 : 'top',
        3 : 'bot',
        4 : 'front',
        5 : 'back', 
    }
    return fm[i]
    
def getCuboidTargets(P, line):
    params = P.parseCuboid(line)
    return [torch.tensor([params[1], params[2], params[3]]), params[4]]


def getReflectTargets(P, line):
    cub = torch.zeros(11, dtype=torch.float)
    axis = torch.zeros(3, dtype=torch.float)
    parse = P.parseReflect(line)
    cub[getCubeIndex(parse[0])] = 1
    axis[getAxisIndex(parse[1])] = 1

    coords = torch.tensor([float(p) for p in parse[2:]])
    
    return [cub, axis]


def getTranslateTargets(P, line):
    cub = torch.zeros(11, dtype=torch.float)
    axis = torch.zeros(3, dtype=torch.float)
    parse = P.parseTranslate(line)
    cub[getCubeIndex(parse[0])] = 1
    axis[getAxisIndex(parse[1])] = 1
    prms = torch.tensor([float(parse[2]), parse[3]])
    
    return [cub, axis, prms]


def getAttachTargets(P, line):
    cub1 = torch.zeros(11, dtype=torch.float)
    cub2 = torch.zeros(11, dtype=torch.float)
    parse = P.parseAttach(line)
    cub1[getCubeIndex(parse[0])] = 1
    cub2[getCubeIndex(parse[1])] = 1
    coords = torch.tensor([float(p) for p in parse[2:]])
    
    return [cub1, cub2, coords]

def getSqueezeTargets(P, line):
    cub1 = torch.zeros(11, dtype=torch.float)
    cub2 = torch.zeros(11, dtype=torch.float)
    cub3 = torch.zeros(11, dtype=torch.float)
    face = torch.zeros(6, dtype=torch.float)
    parse = P.parseSqueeze(line)
    cub1[getCubeIndex(parse[0])] = 1
    cub2[getCubeIndex(parse[1])] = 1
    cub3[getCubeIndex(parse[2])] = 1
    face[getFaceIndex(parse[3])] = 1
    uv = torch.tensor([float(p) for p in parse[4:]])
    
    return [cub1, cub2, cub3, face, uv]


# Takes in a program, returns the target values for each line in the program 
def progToTarget(program):
    target = []
    P = Program()
    weight_info = []
    for line in program:
        tl = torch.zeros(INPUT_DIM, dtype = torch.float)
        if "Cuboid(" in line:
            tl[1] = 1.0
            t = getCuboidTargets(P, line)
            tl[40:43] = t[0]
            tl[62] = t[1]
            wi = [40,41,42]
            
        elif "attach(" in line:
            tl[2] = 1.0
            t = getAttachTargets(P, line)
            tl[7:18] = t[0]
            tl[18:29] = t[1]
            tl[43:49] = t[2]
            wi = [43,44,45,46,47,48]

        elif "reflect(" in line:
            tl[3] = 1.0
            t = getReflectTargets(P, line)
            tl[7:18] = t[0]
            tl[49:52] = t[1]
            wi = []
            
        elif "translate(" in line:
            tl[4] = 1.0
            t = getTranslateTargets(P, line)
            tl[7:18] = t[0]
            tl[49:52] = t[1]
            tl[52:54] = t[2]
            wi = [52, 53]
        elif "squeeze(" in line:
            tl[5] = 1.0
            t = getSqueezeTargets(P, line)
            tl[7:18] = t[0]
            tl[18:29] = t[1]
            tl[29:40] = t[2]
            tl[54:60] = t[3]
            tl[60:62] = t[4]
            wi = [60, 61]
            
        target.append(tl)
        weight_info.append(wi)
        
    start = torch.zeros(INPUT_DIM, dtype = torch.float)
    end = torch.zeros(INPUT_DIM, dtype = torch.float)

    start[0] = 1.0
    end[6] = 1.0

    bb_dims = target[0][40:43]
        
    weights = getWeights(torch.stack([t for t in target]), weight_info)
    
    inp = torch.stack([t for t in [start] + target])
    tar = torch.stack([t for t in target + [end]])
    
    return inp, tar, weights, bb_dims


def getCuboidLine(out, c):
    name = indToCube(c)
    dims = out[40:43].tolist()
    aligned = out[62].item() > 0.
    return assign(name, make_function("Cuboid", dims + [str(aligned)]))

def getAttachLines(out):
    cube1 = indToCube(out[7:18].max(0).indices.item())
    cube2 = indToCube(out[18:29].max(0).indices.item())
    coords = out[43:49].tolist()
    return make_function('attach', [cube1, cube2] + coords)

def getReflectLine(out):
    cube = indToCube(out[7:18].max(0).indices.item())
    axis = indToAxis(out[49:52].max(0).indices.item())
    return make_function('reflect', [cube, axis])

def getTranslateLine(out):
    cube = indToCube(out[7:18].max(0).indices.item())
    axis = indToAxis(out[49:52].max(0).indices.item())
    num = int(round(out[52].item()))
    scale = out[53].item()
    return make_function('translate', [cube, axis, num, scale])

def getSqueezeLine(out):
    cube1 = indToCube(out[7:18].max(0).indices.item())
    cube2 = indToCube(out[18:29].max(0).indices.item())
    cube3 = indToCube(out[29:40].max(0).indices.item())
    face = indToFace(out[54:60].max(0).indices.item())
    uv = out[60:62].tolist()
    return make_function('squeeze', [cube1, cube2, cube3, face] + uv)

# Takes predictions in 58 tensor format and converts to program syntax
def predToProg(pred):
    prog = []
    
    c = 0 
    
    for out in pred:
        command = out[:7].max(0).indices
        if command == 0:
            prog.append("<START>")

        elif command == 1:
            line = getCuboidLine(out, c)
            prog.append(line)
            c += 1
            
        elif command == 2:
            line = getAttachLines(out)
            prog.append(line)

        elif command == 3:
            line = getReflectLine(out)
            prog.append(line)

        elif command == 4:
            line = getTranslateLine(out)
            prog.append(line)            

        elif command == 5:
            line = getSqueezeLine(out)
            prog.append(line)
            
        elif command == 6:
            prog.append("<END>")

    return prog

def getWeights(target, weight_info):
    weights = torch.zeros(len(weight_info)+1, INPUT_DIM)
    for i, wi in enumerate(weight_info):
        for w in wi:
            weights[i][w] = 1.
            
    return weights


def getCuboidParams(out):
    dims = out[40:43]
    return [dims[0], dims[1], dims[2]]


def getAttachParams(out):
    coords = out[43:49]
    return [coords[0], coords[1], coords[2], coords[3], coords[4], coords[5]]  


def getSqueezeParams(out):
    cube2 = indToCube(out[18:29].max(0).indices.item())
    cube3 = indToCube(out[29:40].max(0).indices.item())
    face = indToFace(out[54:60].max(0).indices.item())
    
    uv = out[60:62]
        
    P = Program()
    oface = P.getOppFace(face)
    
    atc1, ato1 = P.getSqueezeAtt(
        face, uv[0], uv[1], cube2 == 'bbox'
    )

    atc2, ato2 = P.getSqueezeAtt(
        oface, uv[0], uv[1], cube3 == 'bbox'
    )
    params = []

    for p in atc1 + ato1 + atc2 + ato2:
        if isinstance(p, torch.Tensor):
            params.append(p)
        else:
            params.append(torch.tensor(p))
            
    return params
    

def predToParams(pred):
    params = []
        
    for out in pred:
        command = out[:7].max(0).indices

        if command == 1:
            _params = getCuboidParams(out)
                        
        elif command == 2:
            _params = getAttachParams(out)

        elif command == 5:
            _params = getSqueezeParams(out)
        else:
            continue
            
        params += _params
            
    return params
