import os
import sys
from numpy import *
import numpy as np


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def read_off(file):
    """
    Reads vertices and faces from an off file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """
    assert os.path.exists(file)
    with open(file, 'r') as fp:
        lines = fp.readlines()
    lines = [line.strip() for line in lines]

    assert lines[0] == 'OFF'

    parts = lines[1].split(' ')
    assert len(parts) == 3

    num_vertices = int(parts[0])
    assert num_vertices > 0

    num_faces = int(parts[1])
    assert num_faces > 0

    vertices = []
    for i in range(num_vertices):
        vertex = lines[2 + i].split(' ')
        vertex = [float(point) for point in vertex]
        assert len(vertex) == 3

        vertices.append(vertex)

    faces = []
    for i in range(num_faces):
        face = lines[2 + num_vertices + i].split(' ')
        face = [int(index) for index in face]

        assert face[0] == len(face) - 1
        for index in face:
            assert index >= 0 and index < num_vertices
        assert len(face) > 1
        faces.append(face[1:4])

    return vertices, faces


