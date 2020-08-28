from numpy import *
import numpy as np


def calcRotationUnitVecPair(model, target):

    m = np.linalg.norm(model)
    t = np.linalg.norm(target)
    d = sum( model * target ) / (m * t)
    c = real(d)

    theta = np.arccos(c)
    cc = 1 - c
    s = sin(theta)

    ortho = np.cross(model/m, target/t)

    if theta < 0.01:

        r = np.identity(3)

    elif theta > 3.14:

        r = - np.identity(3)

    else:
        u = ortho[0] / (np.linalg.norm(ortho) )
        v = ortho[1] / (np.linalg.norm(ortho) )
        w = ortho[2] / (np.linalg.norm(ortho) )
        r = np.asarray([[c + pow(u,2) * cc, -w * s + u * v * cc, v * s + u * w * cc],
             [w * s + u * v * cc, c + pow(v, 2) * cc, -u * s + v * w * cc],
             [-v * s + u * w * cc, u * s + v * w * cc, c + pow(w, 2) * cc] ])

    return r




def calcTranslation(A, B):

    assert len(A) == len(B)
    N = A.shape[0]  # total points
    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)
    t = - centroid_A.T + centroid_B.T
    return t

def calcRigidTranformationPrincipalAxis(A, B, A_pa):

    assert len(A) == len(B)
    N = A.shape[0] # total points
    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)

    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = dot(linalg.pinv(AA), BB)
    U, S, V = linalg.svd(H.transpose())
    R = dot(U, V)

    # remove refrection
    D = np.diag([1, 1, linalg.det(R)])
    R = dot(U,dot(D, V))

    A_pa_new = np.dot(R,  A_pa.T).T
    R = calcRotationUnitVecPair(A_pa, A_pa_new)
    t = dot(-R , centroid_A.T) + centroid_B.T

    return R, t

def calcRigidTranformation(A, B):

    assert len(A) == len(B)
    N = A.shape[0] # total points
    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)

    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = dot(linalg.pinv(AA), BB)
    U, S, V = linalg.svd(H.transpose())
    R = dot(U, V)

    # remove refrection
    D = np.diag([1, 1, linalg.det(R)])
    R = dot(U,dot(D, V))
    t = dot(-R , centroid_A.T) + centroid_B.T

    return R, t

def calcRigidTranformationPointToPlane(A, B, BN):

    assert len(A) == len(B)
    N = A.shape[0] # total points

    b = np.sum((B-A)* BN, axis = 1)
    A1 = np.cross(A, BN)
    A2 = BN
    A= np.c_[A1,A2]

    print(b.shape)
    x = np.linalg.solve(np.dot(A.transpose(),A), np.dot(A.transpose(),b)) 
    #x= np.linalg.lstsq(A,b)[0]

    #x = np.asarray(x[:,0])
    print(x.shape)
    r = x[:3]
    t = np.reshape(x[3:],(3,1))

    #print(x)
    #print(t)

    Rx = np.array([[1, 0, 0], [0, np.cos(r[0]), -np.sin(r[0])], [0, np.sin(r[0]), np.cos(r[0])]])
    Ry = np.array([[np.cos(r[1]), 0, np.sin(r[1])], [0, 1, 0], [-sin(r[1]), 0, np.cos(r[1])]])
    Rz = np.array([[np.cos(r[2]), -np.sin(r[2]), 0], [np.sin(r[2]), np.cos(r[2]), 0], [0, 0, 1]])

    R= np.dot(Rx, np.dot(Ry,Rz))


    #print(R)
    #print(t)

    return R, t


def Axis2Rot(a):

    theta = np.linalg.norm(a)
    M= np.identity(3)
    w = np.asarray([[0, -a[0,2], a[0,1]], [a[0,2], 0, -a[0,0]] , [- a[0,1],a[0,0], 0]])
    if not theta == 0:
        M = M + (sin(theta) / theta) * w + ((1 - cos(theta)) / pow(theta,2)) * dot(w , w)
    return M
