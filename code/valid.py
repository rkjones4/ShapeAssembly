import argparse
from functools import reduce
import numpy as np
import os
import pandas as pd
import trimesh as tm
from trimesh.collision import CollisionManager
from trimesh.creation import box
import pickle
from tqdm import tqdm
import pybullet as p
import pybullet_data
from trimesh.util import concatenate as meshconcat
import xml.etree.ElementTree as xml
import json
import shutil
import utils
from utils import sample_surface
import torch
from pointnet_classification import main as get_variance
import time

def check_rooted(verts, faces):
    # Load up the mesh
    mesh = tm.Trimesh(vertices=verts, faces=faces)
    # Switch from y-up to z-up
    mesh.vertices = mesh.vertices[:, [0, 2, 1]]
    mesh.fix_normals()
    # Find the height of the ground plane
    z_ground = mesh.bounds[0][2]

    # Extract the individual cuboid parts
    comps = mesh.split().tolist()
    # Also create a thin box for the ground plane
    ground = box(
        extents = [10, 10, 0.01],
        transform = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, z_ground - 0.01/2],
            [0, 0, 0, 1]
        ]
    )
    comps.insert(0, ground)

    # Detect (approximate) intersections between parts
    # collision_dist = 0.005 * mesh.scale
    # collision_dist = 0.01 * mesh.scale
    collision_dist = 0.02 * mesh.scale
    adjacencies = {comp_index : [] for comp_index in range(len(comps))}
    manager = CollisionManager()
    for i in range(len(comps)-1):
        manager.add_object(str(i), comps[i])
        for j in range(i+1, len(comps)):
            dist = manager.min_distance_single(comps[j])
            if (dist < collision_dist):
                adjacencies[i].append(j)
                adjacencies[j].append(i)
        manager.remove_object(str(i))

    # Run a DFS starting from the ground, check if everything is reachable
    visited = [False for comp in comps]
    stack = [0]     # Index of 'ground'
    while len(stack) > 0:
        nindex = stack.pop()
        visited[nindex] = True
        for cindex in adjacencies[nindex]:
            if not visited[cindex]:
                stack.append(cindex)
    return all(visited)


def obj2urdf(verts, faces, output_dir, density=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load up the mesh
    mesh = tm.Trimesh(vertices=verts, faces=faces)
    # Switch from y-up to z-up
    mesh.vertices = mesh.vertices[:, [0, 2, 1]]
    mesh.fix_normals()
    # Extract the individual cuboid parts
    comps = mesh.split().tolist()

    # Detect (approximate) intersections between parts, to use for building joints
    # collision_dist = 0.005 * mesh.scale
    # collision_dist = 0.01 * mesh.scale
    collision_dist = 0.02 * mesh.scale
    adjacencies = {comp_index : [] for comp_index in range(len(comps))}
    manager = CollisionManager()
    for i in range(len(comps)-1):
        manager.add_object(str(i), comps[i])
        for j in range(i+1, len(comps)):
            dist = manager.min_distance_single(comps[j])
            if (dist < collision_dist):
                adjacencies[i].append(j)
                adjacencies[j].append(i)
        manager.remove_object(str(i))

    # Compute connected components
    conn_comps = []
    visited = [False for _ in comps]
    while not all(visited):
        conn_comp = set([])
        start_idx = visited.index(False)
        stack = [start_idx]
        while len(stack) > 0:
            idx = stack.pop()
            visited[idx] = True
            conn_comp.add(idx)
            for nidx in adjacencies[idx]:
                if not visited[nidx]:
                    stack.append(nidx)
        conn_comps.append(list(conn_comp))

    # We export one URDF object file per connected component
    for i,conn_comp in enumerate(conn_comps):

        # Re-center this connected component
        ccmesh = meshconcat([comps[j] for j in conn_comp])
        c = ccmesh.centroid
        transmat = [
            [1, 0, 0, -c[0]],
            [0, 1, 0, -c[1]],
            [0, 0, 1, -c[2]],
            [0, 0, 0, 1]
        ]
        for j in conn_comp:
            comps[j].apply_transform(transmat)
        ccmesh.apply_transform(transmat)
        # Also, record where to start this mesh in the simulation
        # That's the x,y coords of the centroid, and -bbox bottom for the z (so it sits on the ground)
        # And the bbox diagonal (we use this for error thresholding)

        metadata = {
            'start_pos': [c[0], c[1], -ccmesh.bounds[0][2]],
            'volume': ccmesh.volume,
            'height': abs(ccmesh.bounds[0][2] - ccmesh.bounds[1][2]),
            'base': min(abs(ccmesh.bounds[0][0] - ccmesh.bounds[1][0]), abs(ccmesh.bounds[0][1] - ccmesh.bounds[1][1]))
        }
        
        with open(f'{output_dir}/assembly_{i}.json', 'w') as f:
            f.write(json.dumps(metadata))

        # Build a directed tree by DFS
        root_idx = conn_comp[0]
        root = {'id': root_idx, 'children': []}
        fringe = [root]
        visited = set([root['id']])
        while len(fringe) > 0:
            node = fringe.pop()
            for neighbor in adjacencies[node['id']]:
                if not (neighbor in visited):
                    child_node = {'id': neighbor, 'children': []}
                    node['children'].append(child_node)
                    visited.add(child_node['id'])
                    fringe.append(child_node)

        # Build up the URDF data structure
        urdf_root = xml.Element('robot')
        urdf_root.set('name', 'part_graph_shape')
        # Links
        for j in conn_comp:
            comp = comps[j]
            link = xml.SubElement(urdf_root, 'link')
            link.set('name', f'part_{j}')
            visual = xml.SubElement(link, 'visual')
            geometry = xml.SubElement(visual, 'geometry')
            mesh = xml.SubElement(geometry, 'mesh')
            mesh.set('filename', f'{output_dir}/part_{j}.stl')
            material = xml.SubElement(visual, 'material')
            material.set('name', 'gray')
            color = xml.SubElement(material, 'color')
            color.set('rgba', '0.5 0.5 0.5 1')
            collision = xml.SubElement(link, 'collision')
            geometry = xml.SubElement(collision, 'geometry')
            mesh = xml.SubElement(geometry, 'mesh')
            mesh.set('filename', f'{output_dir}/part_{j}.stl')
            inertial = xml.SubElement(link, 'inertial')
            mass = xml.SubElement(inertial, 'mass')
            mass.set('value', str(comp.volume * density))
            inertia = xml.SubElement(inertial, 'inertia')
            inertia.set('ixx', '1.0')
            inertia.set('ixy', '0.0')
            inertia.set('ixz', '0.0')
            inertia.set('iyy', '1.0')
            inertia.set('iyz', '0.0')
            inertia.set('izz', '1.0')
        # Joints
        fringe = [root]
        while len(fringe) > 0:
            node = fringe.pop()
            for child_node in node['children']:
                joint = xml.SubElement(urdf_root, 'joint')
                joint.set('name', f'{node["id"]}_to_{child_node["id"]}')
                joint.set('type', 'fixed')
                parent = xml.SubElement(joint, 'parent')
                parent.set('link', f'part_{node["id"]}')
                child = xml.SubElement(joint, 'child')
                child.set('link', f'part_{child_node["id"]}')
                origin = xml.SubElement(joint, 'origin')
                origin.set('xyz', '0 0 0')
                fringe.append(child_node)

        # Save URDF file to disk
        # Have to make sure to split it into multiple lines, otherwise Bullet's URDF parser will
        #    throw an error trying to load really large files as a single line...
        xmlstring = xml.tostring(urdf_root, encoding='unicode')
        xmlstring = '>\n'.join(xmlstring.split('>'))
        with open(f'{output_dir}/assembly_{i}.urdf', 'w') as f:
            f.write(xmlstring)

    # Write the parts to disk as STL files for the URDF to refer to
    for i,comp in enumerate(comps):
        comp.export(f'{output_dir}/part_{i}.stl')

    

def check_stability(verts, faces, gui=False):
    # First, check if the file is even rooted.
    # If it's not rooted, it can't be stable
    if not check_rooted(verts, faces):
        return False

    # Start up the simulation
    mode = p.GUI if gui else p.DIRECT
    physicsClient = p.connect(mode)
    p.setGravity(0, 0, -9.8)

    # Load the ground plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # print(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")

    # Convert the object to a URDF assembly, load it up
    # There may be more than one URDF, if the object had more than one connected component
    obj2urdf(verts, faces, 'tmp')
    objIds = []
    startPositions = {}
    volumes = {}
    heights = {}
    bases = {}
    
    for urdf in [f for f in os.listdir('tmp') if os.path.splitext(f)[1] == '.urdf']:
        with open(f'tmp/{os.path.splitext(urdf)[0]}.json', 'r') as f:
            data = json.loads(f.read())
        startPos = data['start_pos']
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        objId = p.loadURDF(f"tmp/{urdf}",startPos, startOrientation)
        objIds.append(objId)
        startPositions[objId] = startPos
        volumes[objId] = data['volume']
        heights[objId] = data['height']
        bases[objId] = data['base']
    shutil.rmtree('tmp')

    # Disable collisions between all objects (we only want collisions between objects and the ground)
    # That's because we want to check if the different components are *independently* stable, and
    #    having them hit each other might muck up that judgment
    for i in range(0, len(objIds)-1):
        ni = p.getNumJoints(objIds[i])
        for j in range(i+1, len(objIds)):
            nj = p.getNumJoints(objIds[j])
            for k in range(-1, ni):
                for l in range(-1, nj):
                    p.setCollisionFilterPair(objIds[i], objIds[j], k, l, False)
    import math
    # See if objects are stable under some perturbation
    for objId in objIds:
        s = volumes[objId] 
        b = bases[objId]
        v = s * b
        # 800, 4, 4, 4
        p.applyExternalForce(objId, -1, (0, 0, 1000*v), startPositions[objId], p.WORLD_FRAME)
        p.applyExternalTorque(objId, -1, (0, 4*v, 0), p.WORLD_FRAME)
        p.applyExternalTorque(objId, -1, (4*v, 0, 0), p.WORLD_FRAME)
        p.applyExternalTorque(objId, -1, (0, 0, 80*v), p.WORLD_FRAME)

    # Run simulation
    if gui:
        for i in range(600):
            p.stepSimulation()
            time.sleep(1./600.)
            
    else:
        for i in range(10000):            
            p.stepSimulation()

    for objId in objIds:
        endPos, _ = p.getBasePositionAndOrientation(objId)
        zend = endPos[2]
        zstart = startPositions[objId][2]
        zdiff = abs(zstart - zend)
        if zdiff > abs(0.025 * heights[objId]):

            p.disconnect()
            return False
    p.disconnect()
    return True

