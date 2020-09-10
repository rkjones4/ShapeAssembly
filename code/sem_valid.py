import torch
from ShapeAssembly import Program
from parse_prog import getCuboidLine, getAttachLines, getReflectLine, getTranslateLine, getSqueezeLine
from copy import deepcopy

BE = 1.5
VERBOSE = False
MASK_BAD_OUT = True
BAD_DEC = 1000.
DO_REJECT = True
MAX_REJECT = 10
INPUT_DIM = 63
device = torch.device("cuda")

"""
Contains logic for performing semantically valid generation of ShapeAssembly programs. semValidGen is the main entry-point.

"""

def cleanLine(out):
    assert out.shape[1] == 1, 'bad input to clean struct'
    out = out.squeeze()
    new = torch.zeros(INPUT_DIM, dtype=torch.float).to(device)

    c = torch.argmax(out[:7])
    new[c] = 1.0

    if c == 1:
        new[40:43] = torch.clamp(out[40:43], 0.01, 10.0)
        new[62] = 1.0 if out[62] > 0. else 0.
        
    elif c == 2:
        c1 = torch.argmax(out[7:18])
        c2 = torch.argmax(out[18:29])

        new[7+c1] = 1.0
        new[18+c2] = 1.0
        new[43:49] = torch.clamp(out[43:49], 0, 1)

    elif c == 3:
        c1 = torch.argmax(out[7:18])
        new[7+c1] = 1.0
        axis = torch.argmax(out[49:52])
        new[49+axis] = 1.0

    elif c == 4:
        c1 = torch.argmax(out[7:18])
        new[7+c1] = 1.0
        axis = torch.argmax(out[49:52])
        new[49+axis] = 1.0
        
        new[52] = max(round(out[52].item()), 1)
        new[53] = torch.clamp(out[53], 0, 1)

    elif c == 5:
        c1 = torch.argmax(out[7:18])
        c2 = torch.argmax(out[18:29])
        c3 = torch.argmax(out[29:40])
        f = torch.argmax(out[54:60])

        new[7+c1] = 1.0
        new[18+c2] = 1.0
        new[29+c3] = 1.0
        new[54+f] = 1.0
        new[60:62] = torch.clamp(out[60:62], 0, 1)
        
    new = new.unsqueeze(0).unsqueeze(0)
            
    return new


def clean_forward(net, inp_seq, h, h_start, bb_dims, hier_ind, input_dim, device, P):

    assert inp_seq.shape[1] == 1, "Clean forward handles one lines at a time"
    
    good_geom = True
    
    bb_dims = bb_dims.unsqueeze(0).unsqueeze(0).repeat(1,inp_seq.shape[1],1)
            
    hier_oh = torch.zeros(1, inp_seq.shape[1], 3).to(device)
    hier_oh[0, :, min(hier_ind,2)] = 1.0
                
    inp = net.inp_net(
        torch.cat(
            (inp_seq, bb_dims, hier_oh), dim=2)
    )
            
    gru_out, h = net.gru(inp, h)
        
    cmd_out = net.cmd_net(gru_out)

    bad_inds = []
    
    if P.mode == 'cuboid':
        bad_inds = [0, 3, 4]

    if P.mode == 'attach':
        bad_inds = [0, 1]

    if P.mode == 'sym':
        bad_inds = [0, 1, 2, 5]
        
    for b in bad_inds:
        cmd_out[0, 0, b] -= BAD_DEC
    
    command = torch.argmax(cmd_out[0,0,:]).item()
                
    clean_cmd = torch.zeros(1, 1, 7, dtype=torch.float, device = device)
    clean_cmd[0, 0, command] = 1.0
                
    cube_out = net.cube_net(
        torch.cat((gru_out, clean_cmd), dim = 2)
    )

    clean_cube = torch.zeros(1, 1, 33, dtype=torch.float, device = device)
    clean_axis = torch.zeros(1, 1, 3, dtype=torch.float, device = device)
    clean_face = torch.zeros(1, 1, 6, dtype=torch.float, device = device)
    
    if command == 2:
        try:
            cub1, cub2 = getSVAtt(cube_out[0,0,:], P)
            clean_cube[0,0,cub1] = 1.0
            clean_cube[0,0,11 + cub2] = 1.0
            
        except Exception as e:
            if VERBOSE:
                print(f"Failed fixing attach with {e}")
            good_geom = False

    elif command == 3:
        try:
            cub1 = getSVRefCube(cube_out[0,0,:], P)
            clean_cube[0,0,cub1] = 1.0
            
        except Exception as e:
            if VERBOSE:
                print(f"Failed fixing ref cube with {e}")
            good_geom = False
            
    elif command == 4:
        try:
            cub1 = getSVTransCube(cube_out[0,0,:], P)
            clean_cube[0,0,cub1] = 1.0
            
        except Exception as e:
            if VERBOSE:
                print(f"Failed fixing trans cube with {e}")
            good_geom = False
            
    elif command == 5:
        try:
            cub1, cub2, cub3 = getSVSqueezeCube(cube_out[0,0,:], P)
            clean_cube[0,0,cub1] = 1.0
            clean_cube[0,0,11 + cub2] = 1.0
            clean_cube[0,0,22 + cub3] = 1.0
            
        except Exception as e:
            if VERBOSE:
                print(f"Failed fixing squeeze cube with {e}")
            good_geom = False


    dim_out = net.dim_net(
        torch.cat((gru_out, bb_dims), dim = 2)
    )
            
    att_out = net.att_net(
        torch.cat((gru_out, clean_cube[:,:,:22]), dim = 2)
    )

    sym_out = net.sym_net(
        torch.cat((gru_out, clean_cube[:,:,:11], bb_dims), dim = 2)
    )

    squeeze_out = net.squeeze_net(
        torch.cat((gru_out, clean_cube), dim=2)
    )
    
    align_out = net.align_net(
        torch.cat((gru_out, bb_dims), dim = 2)
    )

    clean_dim = torch.zeros(1, 1, 3, dtype=torch.float, device = device)
    clean_att = torch.zeros(1, 1, 6, dtype=torch.float, device = device)
    clean_sym = torch.zeros(1, 1, 5, dtype=torch.float, device = device)
    clean_squeeze = torch.zeros(1, 1, 8, dtype=torch.float, device = device)
    clean_align = torch.zeros(1, 1, 1, dtype=torch.float, device = device)

    if command == 1:
        clean_dim[0, 0, 0] = torch.clamp(dim_out[0, 0, 0], 0.01, bb_dims[0,0,0])
        clean_dim[0, 0, 1] = torch.clamp(dim_out[0, 0, 1], 0.01, bb_dims[0,0,1])
        clean_dim[0, 0, 2] = torch.clamp(dim_out[0, 0, 2], 0.01, bb_dims[0,0,2])
        clean_align[0,0,0] = 1.0 if align_out[0,0,0] > 0. else 0.0

    if command == 2:
        clean_att[0,0,:] = torch.clamp(att_out[0,0,:], 0., 1.0)
        if cub2 == 0:
            if clean_att[0,0,4] <= .5 and clean_att[0,0,4] > .1:
                clean_att[0,0,4] = 0.1
                
            elif clean_att[0,0,4] >= .5 and clean_att[0,0,4] <.9:
                clean_att[0,0,4] = 0.9

    if command == 3:

        try:
            axis = getSVRefAxis(sym_out[0, 0, :3], cub1, P)
            clean_sym[0, 0, axis] = 1.0
            
        except Exception as e:
            if VERBOSE:
                print(f"Failed Ref axis with {e}")
                
            good_geom = False
        
        clean_sym[0, 0, axis] = 1.0
        
    if command == 4:
        axis = torch.argmax(sym_out[0,0,:3])
        clean_sym[0, 0, axis] = 1.0
        clean_sym[0, 0, 3] = max(round(sym_out[0,0,3].item()), 1)
        clean_sym[0, 0, 4] = torch.clamp(sym_out[0,0,4], 0., 1.0)

    if command == 5:
        try:
            face = getSVSqueezeFace(squeeze_out[0,0,:6], cub2, cub3, P)        
            clean_squeeze[0, 0, face] = 1.0

        except Exception as e:
            if VERBOSE:
                print(f"Failed fixing face with {e}")                
            good_geom = False
            
        clean_squeeze[0, 0, 6] = torch.clamp(squeeze_out[0,0,6], 0.0, 1.0)
        clean_squeeze[0, 0, 7] = torch.clamp(squeeze_out[0,0,7], 0.0, 1.0)
        
    clean_out = torch.cat(
        (clean_cmd, clean_cube, clean_dim, clean_att, clean_sym, clean_squeeze, clean_align), dim=2
    )

    CP = deepcopy(P)
    line = None
    try:
        if command == 2:
            line = getAttachLines(clean_out.flatten())
        elif command == 3:
            line = getReflectLine(clean_out.flatten())
        elif command == 4:
            line = getTranslateLine(clean_out.flatten())
        elif command == 5:
            line = getSqueezeLine(clean_out.flatten())
            
        if line is not None:
            CP.execute(line)
            bb_geom = insideBBox(
                CP
            )
            good_geom = good_geom and bb_geom
            if not bb_geom:
                if VERBOSE:
                    print("Failed to keep geom inside bbox")
                    
    except Exception as e:
        good_geom = False
        if VERBOSE:
            print(f"In Sem Valid, Failed generating line with {e}")        
    
    next_out = net.next_net(
        torch.cat((
            gru_out, h_start.repeat(1, gru_out.shape[1], 1)
        ), dim = 2)
    )

    leaf_out = net.leaf_net(
        torch.cat((
            gru_out, h_start.repeat(1, gru_out.shape[1], 1)
        ), dim = 2)
    )

    #if P.mode == 'start':
    #    clean_out = bboxLine(bb_dims, input_dim, device)
    #    good_geom = True
        
    return clean_out, next_out, leaf_out, h, good_geom


def insideBBox(P):
    bbox_corners = P.cuboids[f"bbox"].getCorners()
    maxb = bbox_corners.max(dim=0).values * BE
    minb = bbox_corners.min(dim=0).values * BE

    for ci in P.cuboids:
        if ci == 'bbox':
            continue
        corners = P.cuboids[ci].getCorners()
    
        maxc = corners.max(dim=0).values
        minc = corners.min(dim=0).values
    
        if (maxc >= maxb).any():
            return False

        if (minc <= minb).any():
            return False

    return True

def cuboid_line_clean(preds, prog_out, children, P):
    cpreds = []
    cprog_out = []
    cchildren = []

    grounded = P.grounded

    c = 0
    ri = 0
    
    cube_map = {}
    
    for pred, po in zip(preds, prog_out):
        command = torch.argmax(pred[:7])
        if command == 1:
            if ri in grounded:
                cube_map[ri] = c
                c += 1
                cpreds.append(pred)
                cprog_out.append(po)
                cchildren.append(children[ri])
                
            ri += 1
            
    for pred, po in zip(preds, prog_out):

        command = torch.argmax(pred[:7])
        if command == 1:
            continue
        
        if command == 2:
            c1 = torch.argmax(pred[7:18]).item()
            c2 = torch.argmax(pred[18:29]).item()

            if c1 not in cube_map or c2 not in cube_map:
                continue
            
            pred[7+c1] = 0.
            pred[18+c2] = 0.
            pred[7+cube_map[c1]] = 1.0
            pred[18+cube_map[c2]] = 1.0
            
        elif command == 3 or command == 4:
            c1 = torch.argmax(pred[7:18]).item()

            if c1 not in cube_map:
                continue
            
            pred[7+c1] = 0.
            pred[7+cube_map[c1]] = 1.0

        elif command == 5:
            c1 = torch.argmax(pred[7:18]).item()
            c2 = torch.argmax(pred[18:29]).item()
            c3 = torch.argmax(pred[29:40]).item()

            if c1 not in cube_map or c2 not in cube_map or c3 not in cube_map:
                continue
            
            pred[7+c1] = 0.
            pred[18+c2] = 0.
            pred[29+c3] = 0.
            pred[7+cube_map[c1]] = 1.0
            pred[18+cube_map[c2]] = 1.0
            pred[29+cube_map[c3]] = 1.0
            
        cpreds.append(pred)
        cprog_out.append(po)
            
            
    return cpreds, cprog_out, cchildren


def bboxLine(bb_dims, input_dim, device):
    new = torch.zeros(input_dim, dtype=torch.float).to(device)
    new[1] = 1.0
    new[40:43] = torch.clamp(bb_dims, 0.01, 10.0)
    new[62] = 1.0
    return new.view(1,1,-1)


def getSVAtt(out, P):    
    num_cubes = len(P.cuboids)

    grounded = P.grounded
    atts = P.atts
    
    bi1 = set([0])
    bi2 = set()

    for i in range(num_cubes, 11):
        bi1.add(i)
        bi2.add(i)

    for i in range(min(num_cubes, 11)):
        if i not in grounded:
            bi2.add(i)

    bi1 = list(bi1)
    bi2 = list(bi2)

    out[:11][torch.tensor(bi1).long()] -= BAD_DEC
    out[11:22][torch.tensor(bi2).long()] -= BAD_DEC

    c1 = torch.argmax(out[:11]).item()
    c2 = torch.argmax(out[11:22]).item()

    while(
            (c1 in atts and c2 in atts[c1] and c2 != 0) or
            (c1 == c2) or
            (c1 in atts and c2 == 0 and atts[c1].count(c2) == 2)
    ):
        v1 = out[c1] 
        v2 = out[11+c2]

        if v1 < -10 or v2 < -10:
            valid = False
            c1 = 0
            c2 = 0
            break
            
        if v1 < v2:
            out[c1] -= BAD_DEC
            c1 = torch.argmax(out[:11]).item()
        else:
            out[11+c2] -= BAD_DEC
            c2 = torch.argmax(out[11:22]).item()

    return c1, c2


def getSVRefCube(out, P):
            
    num_cubes = len(P.cuboids)
    grounded = P.grounded
    
    bi1 = set([0])

    for i in range(num_cubes, 11):
        bi1.add(i)

    for i in range(min(num_cubes, 11)):
        if i not in grounded:
            bi1.add(i)

    bi1 = list(bi1)
            
    out[:11][torch.tensor(bi1).long()] -= BAD_DEC

    c1 = torch.argmax(out[1:11]).item() + 1

    while( c1 != 0 and \
           f'cube{c1-1}' in P.cuboids and \
           (P.cuboids[f'cube{c1-1}'].pos > 0.0).all()):

        v1 = out[c1] 

        if v1 < -10:
            c1 = 0
            break
            
        out[c1] -= BAD_DEC
        c1 = torch.argmax(out[1:11]).item() + 1

    return c1

def getSVRefAxis(out, c1, P):        

    a = torch.argmax(out).item()
        
    while(c1 != 0 and \
          f'cube{c1-1}' in P.cuboids and \
          P.cuboids[f'cube{c1-1}'].pos[a] > 0.0):
        v = out[a]
        if v < -10:
            break

        out[a] -= BAD_DEC
        a = torch.argmax(out).item()
           
    return a


def getSVTransCube(out, P):
    
    num_cubes = len(P.cuboids)
    grounded = P.grounded
    
    bi1 = set([0])

    for i in range(num_cubes, 11):
        bi1.add(i)

    for i in range(min(num_cubes,11)):
        if i not in grounded:
            bi1.add(i)

    bi1 = list(bi1)
    out[:11][torch.tensor(bi1).long()] -= 100.0
    c1 = torch.argmax(out[1:11]).item() + 1
           
    return c1


def getSVSqueezeCube(out, P):    

    num_cubes = len(P.cuboids)

    grounded = P.grounded
    atts = P.atts
    
    bi1 = set([0])
    bi2 = set()
    bi3 = set()

    for i in range(num_cubes, 11):
        bi1.add(i)
        bi2.add(i)
        bi3.add(i)

    for i in range(min(num_cubes, 11)):
        if i not in grounded:
            bi2.add(i)
            bi3.add(i)

    bi1 = list(bi1)
    bi2 = list(bi2)
    bi3 = list(bi3)
            
    out[:11][torch.tensor(bi1).long()] -= BAD_DEC
    out[11:22][torch.tensor(bi2).long()] -= BAD_DEC
    out[22:][torch.tensor(bi3).long()] -= BAD_DEC

    c1 = torch.argmax(out[:11]).item()
    c2 = torch.argmax(out[11:22]).item()
    c3 = torch.argmax(out[22:]).item()
    
    while(
        (c1 in atts and c2 in atts[c1] and c2 != 0) or
        (c1 == c2) or
        (c1 in atts and c2 == 0 and atts[c1].count(c2) == 2)
    ):
        v1 = out[c1] 
        v2 = out[11+c2]

        if v1 < -10 or v2 < -10:
            c1 = 0
            c2 = 0
            break
            
        if v1 < v2:
            out[c1] -= BAD_DEC
            c1 = torch.argmax(out[:11]).item()
        else:
            out[11+c2] -= BAD_DEC
            c2 = torch.argmax(out[11:22]).item()

    while(
            (c1 in atts and c3 in atts[c1] and c3 != 0) or
            (c1 == c3) or
            (c1 in atts and c3 == 0 and atts[c1].count(c3) == 2)
    ):        
        v3 = out[22+c3]

        if v3 < -10:
            c1 = 0
            c2 = 0
            c3 = 0
            break
            
        out[22+c3] -= BAD_DEC
        c3 = torch.argmax(out[22:]).item()

    return c1, c2, c3

def getSVSqueezeFace(out, c2, c3, P):    
            
    if c2 == 0 or c3 == 0:
        out[0] -= BAD_DEC
        out[1] -= BAD_DEC
        out[4] -= BAD_DEC
        out[5] -= BAD_DEC
        
    f = torch.argmax(out).item()
    
    return f
    

# return cleaned preds, return cleaned output, next things to add to the queue
def semValidGen(prog, rnn, h, hier_ind, max_lines, input_dim, device, gt_prog, rejection_sample):
    q = []

    prog_out = []
    preds = []
    children = []
    out = torch.zeros((1, 1, input_dim), dtype = torch.float).to(device)
    out[0][0][0] = 1.0        
    h_start = h.clone()
    bb_dims = prog["bb_dims"]    
    gt_nc_ind = 0    
    gt_children = []
        
    if gt_prog is not None and len(gt_prog) > 0:
        gt_children = gt_prog["children"]

    P = Program()
    P.mode = 'start'
    P.grounded = set([0])
    P.atts = {}
    c = 0

    stop = False

    loops = 0

    num_rejects = 0
    
    while(not stop and loops < max_lines):
        loops += 1 
        
        prev_out = out.clone().detach()

        out, pnext, pleaf, h, valid = clean_forward(
            rnn,
            out,
            h,
            h_start,
            bb_dims,
            hier_ind,
            input_dim,
            device,
            P
        )
        
        if not valid:
            if rejection_sample and DO_REJECT:
                assert False, "Couldn't clean line"
            num_rejects += 1
            if MASK_BAD_OUT and num_rejects < MAX_REJECT:
                out = prev_out
            else:
                num_rejects = 0
                
            continue
        
        prog_out.append(out)        
        line = out.clone().detach().squeeze()
        preds.append(line)
        
        command = torch.argmax(line[:7])

        pline = None
        
        if command == 1:
            P.mode = 'cuboid'
            pline = getCuboidLine(line, c)
            c += 1

        elif command == 2:
            P.mode = 'attach'
            cub1 = torch.argmax(line[7:18]).item()
            cub2 = torch.argmax(line[18:29]).item()

            P.grounded.add(cub1)
            
            if cub2 in P.atts:
                P.atts[cub2].append(cub1)
            else:
                P.atts[cub2] = [cub1]

            if cub1 in P.atts:
                P.atts[cub1].append(cub2)
            else:
                P.atts[cub1] = [cub2]
                
            pline = getAttachLines(line)

        elif command == 3:
            P.mode = 'sym'
            pline = getReflectLine(line)

        elif command == 4:
            P.mode = 'sym'
            pline = getTranslateLine(line)

        elif command == 5:
            P.mode = 'attach'
            cub1 = torch.argmax(line[7:18]).item()
            cub2 = torch.argmax(line[18:29]).item()
            cub3 = torch.argmax(line[29:40]).item()

            P.grounded.add(cub1)
            
            if cub2 in P.atts:
                P.atts[cub2].append(cub1)
            else:
                P.atts[cub2] = [cub1]

            if cub3 in P.atts:
                P.atts[cub3].append(cub1)
            else:
                P.atts[cub3] = [cub1]

            if cub1 in P.atts:
                P.atts[cub1].append(cub2)
                P.atts[cub1].append(cub3)
            else:
                P.atts[cub1] = [cub2, cub3]
                
            pline = getSqueezeLine(line)
            
        try:
            if pline is not None:
                P.execute(pline)

        except Exception:
            if VERBOSE:
                print("Unexpectedly, failed to execute line")
            pass
                
        # Stop at end token or when we have gone past max lines
    
        if command == 6:
            stop = True
            
        # If make a new Cuboid, use l to decide if it should have a child program or be a leaf
        if command == 1:

            gt_child = None
                
            if gt_nc_ind < len(gt_children):
                gt_child = gt_children[gt_nc_ind]
                gt_nc_ind += 1

            # Skip BBox line
            if pleaf.squeeze().item() < 0 and len(preds) > 1:
                d = {
                    "children": [],
                    "bb_dims": line[40:43]
                }
                children.append(d)
                q.append((pnext, d, hier_ind+1, gt_child))
                
            else:
                children.append({})

    fc_preds, fc_prog_out, fchildren = cuboid_line_clean(preds, prog_out, children, P)
    prog["children"] = fchildren
    return fc_preds, fc_prog_out, q


def train_forward(net, inp_seq, h, h_start, bb_dims, hier_ind, input_dim, device):

    assert inp_seq.shape[1] == 1, "Clean forward handles one lines at a time"
    
    bb_dims = bb_dims.unsqueeze(0).unsqueeze(0).repeat(1,inp_seq.shape[1],1)
            
    hier_oh = torch.zeros(1, inp_seq.shape[1], 3).to(device)
    hier_oh[0, :, min(hier_ind,2)] = 1.0
                
    inp = net.inp_net(
        torch.cat(
            (inp_seq, bb_dims, hier_oh), dim=2)
    )
            
    gru_out, h = net.gru(inp, h)
        
    cmd_out = net.cmd_net(gru_out)

    bad_inds = []
            
    command = torch.argmax(cmd_out[0,0,:]).item()
                
    clean_cmd = torch.zeros(1, 1, 7, dtype=torch.float, device = device)
    clean_cmd[0, 0, command] = 1.0
                
    cube_out = net.cube_net(
        torch.cat((gru_out, clean_cmd), dim = 2)
    )

    clean_cube = torch.zeros(1, 1, 33, dtype=torch.float, device = device)
    clean_axis = torch.zeros(1, 1, 3, dtype=torch.float, device = device)
    clean_face = torch.zeros(1, 1, 6, dtype=torch.float, device = device)
    cub1 = torch.argmax(cube_out[0,0,:11]).item()
    cub2 = torch.argmax(cube_out[0,0,11:22]).item()
    cub3 = torch.argmax(cube_out[0,0,22:33]).item()
    
    if command == 2:        
        clean_cube[0,0,cub1] = 1.0
        clean_cube[0,0,11 + cub2] = 1.0
            
    elif command == 3:
        clean_cube[0,0,cub1] = 1.0
                        
    elif command == 4:
        clean_cube[0,0,cub1] = 1.0
                        
    elif command == 5:
        clean_cube[0,0,cub1] = 1.0
        clean_cube[0,0,11 + cub2] = 1.0
        clean_cube[0,0,22 + cub3] = 1.0
            
    dim_out = net.dim_net(
        torch.cat((gru_out, bb_dims), dim = 2)
    )
            
    att_out = net.att_net(
        torch.cat((gru_out, clean_cube[:,:,:22]), dim = 2)
    )

    sym_out = net.sym_net(
        torch.cat((gru_out, clean_cube[:,:,:11], bb_dims), dim = 2)
    )

    squeeze_out = net.squeeze_net(
        torch.cat((gru_out, clean_cube), dim=2)
    )
    
    align_out = net.align_net(
        torch.cat((gru_out, bb_dims), dim = 2)
    )

    clean_dim = torch.zeros(1, 1, 3, dtype=torch.float, device = device)
    clean_att = torch.zeros(1, 1, 6, dtype=torch.float, device = device)
    clean_sym = torch.zeros(1, 1, 5, dtype=torch.float, device = device)
    clean_squeeze = torch.zeros(1, 1, 8, dtype=torch.float, device = device)
    clean_align = torch.zeros(1, 1, 1, dtype=torch.float, device = device)

    if command == 1:
        clean_dim[0, 0, :] = dim_out[0, 0, :]
        clean_align[0,0,0] = 1.0 if align_out[0,0,0] > 0. else 0.0

    if command == 2:
        clean_att[0,0,:] = att_out[0,0,:]

    if command == 3:
        axis = torch.argmax(sym_out[0,0,:3])
        clean_sym[0, 0, axis] = 1.0
                            
    if command == 4:
        axis = torch.argmax(sym_out[0,0,:3])
        clean_sym[0, 0, axis] = 1.0
        clean_sym[0, 0, 3] = max(round(sym_out[0,0,3].item()), 1)
        clean_sym[0, 0, 4] = torch.clamp(sym_out[0,0,4], 0., 1.0)

    if command == 5:
        face = torch.argmax(squeeze_out[0,0,:6]).item()
        clean_squeeze[0, 0, face] = 1.0
        clean_squeeze[0, 0, 6] = torch.clamp(squeeze_out[0,0,6], 0.0, 1.0)
        clean_squeeze[0, 0, 7] = torch.clamp(squeeze_out[0,0,7], 0.0, 1.0)
        
    clean_out = torch.cat(
        (clean_cmd, clean_cube, clean_dim, clean_att, clean_sym, clean_squeeze, clean_align), dim=2
    )
    
    next_out = net.next_net(
        torch.cat((
            gru_out, h_start.repeat(1, gru_out.shape[1], 1)
        ), dim = 2)
    )

    leaf_out = net.leaf_net(
        torch.cat((
            gru_out, h_start.repeat(1, gru_out.shape[1], 1)
        ), dim = 2)
    )
        
    return clean_out, next_out, leaf_out, h


# return cleaned preds, return cleaned output, next things to add to the queue
def semValidTrain(prog, rnn, h, hier_ind, max_lines, input_dim, device):
    q = []

    prog_out = []
    preds = []
    children = []
    in_line = torch.zeros((1, 1, input_dim), dtype = torch.float).to(device)
    in_line[0][0][0] = 1.0
    h_start = h.clone()
    bb_dims = prog["bb_dims"]

    c = 0

    stop = False
    loops = 0
    num_rejects = 0
    
    while(not stop and loops < max_lines):
        
        loops += 1 
        
        out, pnext, pleaf, h = rnn(
            in_line, h, h_start, bb_dims, hier_ind
        )
                
        prog_out.append(out)        
        in_line = cleanLine(out)
        line = in_line.squeeze()
        preds.append(line)
        
        command = torch.argmax(line[:7])
                
        # Stop at end token or when we have gone past max lines
    
        if command == 6:
            stop = True
            
        # If make a new Cuboid, use l to decide if it should have a child program or be a leaf
        if command == 1:
            
            # Skip BBox line
            if pleaf.squeeze().item() < 0 and len(preds) > 1:
                d = {
                    "children": [],
                    "bb_dims": line[40:43]
                }
                children.append(d)
                q.append((pnext, d, hier_ind+1, None))                
            else:
                children.append({})

    prog["children"] = children
    return preds, prog_out, q

