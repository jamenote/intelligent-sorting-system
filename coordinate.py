import glob
import cv2
import numpy as np
import math
import argparse
import csv_convert
import test
from getpar import cal_right, cal_left, points, cal_left2, cal_right2


class PNPSolver():
    def __init__(self):
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_LBLUE = (255, 200, 100)
        self.COLOR_GREEN = (0, 240, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_DRED = (0, 0, 139)
        self.COLOR_YELLOW = (29, 227, 245)
        self.COLOR_PURPLE = (224, 27, 217)
        self.COLOR_GRAY = (127, 127, 127)
        self.Points3D = np.zeros((1, 4, 3), np.float32)  # 存放4组世界坐标位置
        self.Points2D = np.zeros((1, 4, 2), np.float32)  # 存放4组像素坐标位置
        self.point2find = np.zeros((1, 2), np.float32)
        self.cameraMatrix = None
        self.distCoefs = None
        self.f = 0

    def rotationVectorToEulerAngles(self, rvecs, anglestype):
        R = np.zeros((3, 3), dtype=np.float64)
        cv2.Rodrigues(rvecs, R)
        sy = math.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        if anglestype == 0:
            x = x * 180.0 / 3.141592653589793
            y = y * 180.0 / 3.141592653589793
            z = z * 180.0 / 3.141592653589793
        elif anglestype == 1:
            x = x
            y = y
            z = z
        print(x)
        return x, y, z

    def CodeRotateByZ(self, x, y, thetaz):  # 将空间点绕Z轴旋转
        x1 = x  # 将变量拷贝一次，保证&x == &outx这种情况下也能计算正确
        y1 = y
        rz = thetaz * 3.141592653589793 / 180
        outx = math.cos(rz) * x1 - math.sin(rz) * y1
        outy = math.sin(rz) * x1 + math.cos(rz) * y1
        return outx, outy

    def CodeRotateByY(self, x, z, thetay):
        x1 = x
        z1 = z
        ry = thetay * 3.141592653589793 / 180
        outx = math.cos(ry) * x1 + math.sin(ry) * z1
        outz = math.cos(ry) * z1 - math.sin(ry) * x1
        return outx, outz

    def CodeRotateByX(self, y, z, thetax):
        y1 = y
        z1 = z
        rx = (thetax * 3.141592653589793) / 180
        outy = math.cos(rx) * y1 - math.sin(rx) * z1
        outz = math.cos(rx) * z1 + math.sin(rx) * y1
        return outy, outz

    def solver(self):
        retval, self.rvec, self.tvec = cv2.solvePnP(self.Points3D, self.Points2D, self.cameraMatrix, self.distCoefs)
        thetax, thetay, thetaz = self.rotationVectorToEulerAngles(self.rvec, 0)
        x = self.tvec[0][0]
        y = self.tvec[1][0]
        z = self.tvec[2][0]
        self.Position_OwInCx = x
        self.Position_OwInCy = y
        self.Position_OwInCz = z
        self.Position_theta = [thetax, thetay, thetaz]
        # print('Position_theta:',self.Position_theta)
        x, y = self.CodeRotateByZ(x, y, -1 * thetaz)
        x, z = self.CodeRotateByY(x, z, -1 * thetay)
        y, z = self.CodeRotateByX(y, z, -1 * thetax)
        self.Theta_W2C = ([-1 * thetax, -1 * thetay, -1 * thetaz])
        self.Position_OcInWx = x * (-1)
        self.Position_OcInWy = y * (-1)
        self.Position_OcInWz = z * (-1)
        self.Position_OcInW = np.array([self.Position_OcInWx, self.Position_OcInWy, self.Position_OcInWz])
        print('Position_OcInW:', self.Position_OcInW)

    def WordFrame2ImageFrame(self, WorldPoints):
        pro_points, jacobian = cv2.projectPoints(WorldPoints, self.rvecs, self.tvecs, self.cameraMatrix, self.distCoefs)
        return pro_points

    def ImageFrame2CameraFrame(self, pixPoints):
        fx = self.cameraMatrix[0][0]
        u0 = self.cameraMatrix[0][2]
        fy = self.cameraMatrix[1][1]
        v0 = self.cameraMatrix[1][2]
        zc = (self.f[0] + self.f[1]) / 2
        xc = (pixPoints[0] - u0) * self.f[0] / fx  # f=fx*传感器尺寸/分辨率
        yc = (pixPoints[1] - v0) * self.f[1] / fy
        point = np.array([xc, yc, zc])
        return point

    def getudistmap(self, cameraMatrix, distCoefs):
        # with open(filename, 'r', newline='') as csvfile:
        #     spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        #     rows = [row for row in spamreader]
        self.cameraMatrix = cameraMatrix
        self.cameraMatrix = np.array(self.cameraMatrix)
        size_w = 23.6
        size_h = 15.2
        #待定
        imageWidth = 1920
        imageHeight = 1080
        self.distCoefs = distCoefs
        self.distCoefs = np.array(self.distCoefs)
        print('dim = %d*%d' % (imageWidth, imageHeight))
        print('Kt = \n', self.cameraMatrix)
        print('Dt = \n', self.distCoefs)
        self.f = [self.cameraMatrix[0][0] * (size_w / imageWidth), self.cameraMatrix[1][1] * (size_h / imageHeight)]
        print('f = \n', self.f)
        return


class GetDistanceOf2linesIn3D():
    def __init__(self):
        print('GetDistanceOf2linesIn3D class')

    def dot(self, ax, ay, az, bx, by, bz):
        result = ax * bx + ay * by + az * bz
        return result

    def cross(self, ax, ay, az, bx, by, bz):
        x = ay * bz - az * by
        y = az * bx - ax * bz
        z = ax * by - ay * bx
        return x, y, z

    def crossarray(self, a, b):
        x = a[1] * b[2] - a[2] * b[1]
        y = a[2] * b[0] - a[0] * b[2]
        z = a[0] * b[1] - a[1] * b[0]
        return np.array([x, y, z])

    def norm(self, ax, ay, az):
        return math.sqrt(self.dot(ax, ay, az, ax, ay, az))

    def norm2(self, one):
        return math.sqrt(np.dot(one, one))

    def SetLineA(self, A1x, A1y, A1z, A2x, A2y, A2z):
        self.a1 = np.array([A1x, A1y, A1z])
        self.a2 = np.array([A2x, A2y, A2z])

    def SetLineB(self, B1x, B1y, B1z, B2x, B2y, B2z):
        self.b1 = np.array([B1x, B1y, B1z])
        self.b2 = np.array([B2x, B2y, B2z])

    def GetDistance(self):
        d1 = self.a2 - self.a1
        d2 = self.b2 - self.b1
        e = self.b1 - self.a1

        cross_e_d2 = self.crossarray(e, d2)
        cross_e_d1 = self.crossarray(e, d1)
        cross_d1_d2 = self.crossarray(d1, d2)

        dd = self.norm2(cross_d1_d2)
        t1 = np.dot(cross_e_d2, cross_d1_d2)
        t2 = np.dot(cross_e_d1, cross_d1_d2)

        t1 = t1 / (dd * dd)
        t2 = t2 / (dd * dd)

        self.PonA = self.a1 + (self.a2 - self.a1) * t1
        self.PonB = self.b1 + (self.b2 - self.b1) * t2
        self.distance = self.norm2(self.PonB - self.PonA)
        print('distance=', self.distance)
        return self.distance


def getpoint():
    print("***************************************")
    print("test example")
    print("***************************************")
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('-file', type=str, default='calibration.csv')
    args = parser.parse_args()
    calibrationfile = args.file
    p4psolver1 = PNPSolver()
    image_path_list = glob.glob("C:/Users/11478/Desktop/par/0-left.jpg")
    cache1, cache2, cache3, cache4 = points(9, 6, image_path_list)
    P11_world = np.array([0, 0, 0])
    P12_world = np.array([180, 0, 0])
    P13_world = np.array([0, 113, 0])
    P14_world = np.array([180, 113, 0])
    p11_img = np.array(cache1)
    p12_img = np.array(cache2)
    p13_img = np.array(cache3)
    p14_img = np.array(cache4)
    p4psolver1.Points3D[0] = np.array([P11_world, P12_world, P13_world, P14_world])
    p4psolver1.Points2D[0] = np.array([p11_img, p12_img, p13_img, p14_img])
    # image_path_list = glob.glob("C:/Users/11478/Desktop/cache-left/*.jpg")
    # mtx_left, dist_left, cache1, cache2, cache3, cache4 = cal_left2(9, 6, image_path_list)
    # cameraMatrix1 = mtx_left
    # distCoefs1 = dist_left
    mtx_left = csv_convert.csv_to_Matrix("C:/Users/11478/Desktop/mtx-left.csv")
    mtx_left = mtx_left.tolist()
    cameraMatrix1 = mtx_left
    dist_left = csv_convert.csv_to_Matrix("C:/Users/11478/Desktop/dist-left.csv")
    dist_left = dist_left.tolist()
    distCoefs1 = dist_left
    img_pos1 = test.decodeDisplay("C:/Users/11478/Desktop/par/0-left.jpg")
    print("要求的中心点是：", img_pos1)
    p4psolver1.point2find = np.array(img_pos1)  #需要检测的点
    p4psolver1.getudistmap(cameraMatrix1, distCoefs1)
    p4psolver1.solver()
    p4psolver2 = PNPSolver()
    image_path_list = glob.glob("C:/Users/11478/Desktop/par/0-right.jpg")
    cache1, cache2, cache3, cache4 = points(9, 6, image_path_list)
    P21_world = np.array([0, 0, 0])
    P22_world = np.array([180, 0, 0])
    P23_world = np.array([0, 113, 0])
    P24_world = np.array([180, 113, 0])
    p21_img = np.array(cache1)
    p22_img = np.array(cache2)
    p23_img = np.array(cache3)
    p24_img = np.array(cache4)
    p4psolver2.Points3D[0] = np.array([P21_world, P22_world, P23_world, P24_world])
    p4psolver2.Points2D[0] = np.array([p21_img, p22_img, p23_img, p24_img])
    mtx_right = csv_convert.csv_to_Matrix("C:/Users/11478/Desktop/mtx-right.csv")
    mtx_right = mtx_right.tolist()
    cameraMatrix2 = mtx_right
    dist_right = csv_convert.csv_to_Matrix("C:/Users/11478/Desktop/dist-right.csv")
    dist_right = dist_right.tolist()
    distCoefs2 = dist_right
    # image_path_list = glob.glob("C:/Users/11478/Desktop/cache-right/*.jpg")
    # mtx_right, dist_right, cache1, cache2, cache3, cache4 = cal_left2(9, 6, image_path_list)
    # cameraMatrix2 = mtx_right
    # distCoefs2 = dist_right
    img_pos2 = test.decodeDisplay("C:/Users/11478/Desktop/par/0-right.jpg")
    print("要求的中心点是：", img_pos2)
    p4psolver2.point2find = np.array(img_pos2)
    p4psolver2.getudistmap(cameraMatrix2, distCoefs2)
    p4psolver2.solver()
    point2find1_CF = p4psolver1.ImageFrame2CameraFrame(p4psolver1.point2find)
    Oc1P_1 = np.array(point2find1_CF)
    print(Oc1P_1)
    Oc1P_1[0], Oc1P_1[1] = p4psolver1.CodeRotateByZ(Oc1P_1[0], Oc1P_1[1], p4psolver1.Theta_W2C[2])
    Oc1P_1[0], Oc1P_1[2] = p4psolver1.CodeRotateByY(Oc1P_1[0], Oc1P_1[2], p4psolver1.Theta_W2C[1])
    Oc1P_1[1], Oc1P_1[2] = p4psolver1.CodeRotateByX(Oc1P_1[1], Oc1P_1[2], p4psolver1.Theta_W2C[0])
    a1 = np.array([p4psolver1.Position_OcInWx, p4psolver1.Position_OcInWy, p4psolver1.Position_OcInWz])
    a2 = a1 + Oc1P_1
    # a2 = (p4psolver1.Position_OcInWx + Oc1P_1[0], p4psolver1.Position_OcInWy + Oc1P_y1, p4psolver1.Position_OcInWz + Oc1P_z1)
    point2find2_CF = p4psolver2.ImageFrame2CameraFrame(p4psolver2.point2find)
    Oc2P_2 = np.array(point2find2_CF)
    print(Oc2P_2)
    Oc2P_2[0], Oc2P_2[1] = p4psolver2.CodeRotateByZ(Oc2P_2[0], Oc2P_2[1], p4psolver2.Theta_W2C[2])
    Oc2P_2[0], Oc2P_2[2] = p4psolver2.CodeRotateByY(Oc2P_2[0], Oc2P_2[2], p4psolver2.Theta_W2C[1])
    Oc2P_2[1], Oc2P_2[2] = p4psolver2.CodeRotateByX(Oc2P_2[1], Oc2P_2[2], p4psolver2.Theta_W2C[0])
    b1 = ([p4psolver2.Position_OcInWx, p4psolver2.Position_OcInWy, p4psolver2.Position_OcInWz])
    b2 = b1 + Oc2P_2
    # b2 = (p4psolver2.Position_OcInWx + Oc2P_x2, p4psolver2.Position_OcInWy + Oc2P_y2, p4psolver2.Position_OcInWz + Oc2P_z2)
    g = GetDistanceOf2linesIn3D()
    g.SetLineA(a1[0], a1[1], a1[2], a2[0], a2[1], a2[2])
    g.SetLineB(b1[0], b1[1], b1[2], b2[0], b2[1], b2[2])
    distance = g.GetDistance()
    pt = (g.PonA + g.PonB) / 2
    print(pt)
    pt = pt.tolist()
    return pt[0], pt[1]


# getpoint()
