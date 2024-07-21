import numpy as np


def makemtx_left(mtx_left):
    np.savetxt("C:/Users/11478/Desktop/mtx-left.csv", mtx_left, delimiter=",", fmt='%.08f')


def makemtx_right(mtx_right):
    np.savetxt("C:/Users/11478/Desktop/mtx-right.csv", mtx_right, delimiter=",", fmt='%.08f')


def makedist_left(dist_left):
    np.savetxt("C:/Users/11478/Desktop/dist-left.csv", dist_left, delimiter=",", fmt='%.08f')


def makedist_right(dist_left):
    np.savetxt("C:/Users/11478/Desktop/dist-right.csv", dist_left, delimiter=",", fmt='%.08f')
