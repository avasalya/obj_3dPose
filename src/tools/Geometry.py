from numpy import *
import numpy as np

class Points(object):
    def __init__(self, vertex, normals):
        self.vertex = vertex  # [(x,y,z)]
        self.nvertex = len(self.vertex)
        self.centroid = np.mean(self.vertex,axis=0)
        self.v_normal = normals

class Mesh(object):

    def __init__(self, vertex, face, principalAxis):

        self.vertex = vertex # [(x,y,z)]
        self.face = face # [(xi,yi,zi)]
        self.nvertex = len(self.vertex)
        self.nface = len(self.face)
        self.f_normal = []
        self.v_normal = []
        self.centroid = np.mean(self.vertex, axis=0)
        self.bbox = np.linalg.norm((np.max(self.vertex, axis =0) -  np.min(self.vertex,axis =0)))
        self.principalAxis = principalAxis
        self.ComputeNormal()

    def ComputeNormal(self):

        u=self.vertex[self.face[:,1],:]-self.vertex[self.face[:,0],:]
        v=self.vertex[self.face[:,2],:]-self.vertex[self.face[:,0],:]
        #normal
        n=np.cross(u,v)
        nl=np.sqrt(np.sum(n*n,axis=1))
        self.f_normal =n/np.c_[nl,nl,nl]

        self.v_normal = np.zeros((self.nvertex, 3))

        for i in range(self.nface):
            for j in range(3):
                self.v_normal[self.face[i, j],: ] += self.f_normal[i,:]

        nl = np.sqrt(np.sum(self.v_normal * self.v_normal, axis=1))
        self.v_normal = self.v_normal / np.c_[nl, nl, nl]

    def Transform(self, R, T):

        self.vertex = np.dot(R, self.vertex.T).T + np.reshape(T, (1,3))
        self.v_normal = np.dot(R, self.v_normal.T).T
        self.centroid = np.mean(self.vertex, axis=0)
        self.principalAxis = np.dot(R, self.principalAxis.T).T
