import cv2
import numpy as np
import glob
import makecsv


def cal_left(h, w, image_path_list):
    # h, w: 棋盘格角点规格
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    # 获取标定板角点的在世界坐标系位置， 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    '''
    objp:
    [[0. 0. 0.]
    [1. 0. 0.]
    [2. 0. 0.]
    ...
    [4. 4. 0.]
    [5. 4. 0.]
    [6. 4. 0.]]
    '''
    objp = np.zeros((h * w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)
    obj_points = []
    img_points = []
    for image_path in image_path_list:
        img = cv2.imread(image_path)
        u, v = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (h, w), None)

        if ret:
            obj_points.append(objp)
            # 专门用来获取棋盘图上内角点的精确位置的， 即在原角点的基础上寻找亚像素角点
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            if [corners2]:
                img_points.append(corners2)
                # print(img_points)
            else:
                img_points.append(corners)
            cv2.drawChessboardCorners(img, (h, w), corners, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
    # print(img_points)
    # print(len(img_points))
    cv2.destroyAllWindows()
    # calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    makecsv.makemtx_left(mtx)
    makecsv.makedist_left(dist)
    np.set_printoptions(suppress=True)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))
    # print('newcameramtx外参', newcameramtx)
    total_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        total_error += error
    # 单位是像素
    # print("average error: ", total_error / len(img_points2))
    return mtx, dist.tolist(), corners2[0][0], corners2[8][0], corners2[45][0], corners2[53][0]


def cal_right(h, w, image_path_list):
    # h, w: 棋盘格角点规格
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    # 获取标定板角点的在世界坐标系位置， 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    '''
    objp:
    [[0. 0. 0.]
    [1. 0. 0.]
    [2. 0. 0.]
    ...
    [4. 4. 0.]
    [5. 4. 0.]
    [6. 4. 0.]]
    '''
    objp = np.zeros((h * w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)
    obj_points = []
    img_points = []
    for image_path in image_path_list:
        img = cv2.imread(image_path)
        u, v = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (h, w), None)

        if ret:
            obj_points.append(objp)
            # 专门用来获取棋盘图上内角点的精确位置的， 即在原角点的基础上寻找亚像素角点
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            if [corners2]:
                img_points.append(corners2)
                # print(img_points)
            else:
                img_points.append(corners)
            cv2.drawChessboardCorners(img, (h, w), corners, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
    # print(img_points)
    # print(len(img_points))
    cv2.destroyAllWindows()
    # calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    makecsv.makemtx_right(mtx)
    makecsv.makedist_right(dist)
    np.set_printoptions(suppress=True)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))
    # print('newcameramtx外参', newcameramtx)
    total_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        total_error += error
    # 单位是像素
    # print("average error: ", total_error / len(img_points2))
    return mtx, dist.tolist(), corners2[0][0], corners2[8][0], corners2[45][0], corners2[53][0]


def points(h, w, image_path_list):
    # h, w: 棋盘格角点规格
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    # 获取标定板角点的在世界坐标系位置， 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    '''
    objp:
    [[0. 0. 0.]
    [1. 0. 0.]
    [2. 0. 0.]
    ...
    [4. 4. 0.]
    [5. 4. 0.]
    [6. 4. 0.]]
    '''
    objp = np.zeros((h * w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)
    obj_points = []
    img_points = []
    for image_path in image_path_list:
        img = cv2.imread(image_path)
        u, v = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (h, w), None)
        if ret:
            obj_points.append(objp)
            # 专门用来获取棋盘图上内角点的精确位置的， 即在原角点的基础上寻找亚像素角点
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            if [corners2]:
                img_points.append(corners2)
                # print(img_points)
            else:
                img_points.append(corners)
            cv2.drawChessboardCorners(img, (h, w), corners, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
    # print(img_points)
    # print(len(img_points))
    cv2.destroyAllWindows()
    # calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    np.set_printoptions(suppress=True)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))
    total_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        total_error += error
    # 单位是像素
    # print("average error: ", total_error / len(img_points2))
    return corners2[0][0], corners2[8][0], corners2[45][0], corners2[53][0]


def get_parm():
    image_path_list = glob.glob("C:/Users/11478/Desktop/cache-left/*.jpg")
    cal_left(9, 6, image_path_list)

    image_path_list = glob.glob("C:/Users/11478/Desktop/cache-right/*.jpg")
    cal_right(9, 6, image_path_list)


def gggg():
    image_path_list = glob.glob("C:/Users/11478/Desktop/par/0-right.jpg")
    points(9, 6, image_path_list)


def cal_left2(h, w, image_path_list):
    # h, w: 棋盘格角点规格
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    # 获取标定板角点的在世界坐标系位置， 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    '''
    objp:
    [[0. 0. 0.]
    [1. 0. 0.]
    [2. 0. 0.]
    ...
    [4. 4. 0.]
    [5. 4. 0.]
    [6. 4. 0.]]
    '''
    objp = np.zeros((h * w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)
    obj_points = []
    img_points = []
    for image_path in image_path_list:
        img = cv2.imread(image_path)
        u, v = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (h, w), None)

        if ret:
            obj_points.append(objp)
            # 专门用来获取棋盘图上内角点的精确位置的， 即在原角点的基础上寻找亚像素角点
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            if [corners2]:
                img_points.append(corners2)
                # print(img_points)
            else:
                img_points.append(corners)
            cv2.drawChessboardCorners(img, (h, w), corners, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
    # print(img_points)
    # print(len(img_points))
    cv2.destroyAllWindows()
    # calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    # makecsv.makemtx_left(mtx)
    # makecsv.makedist_left(dist)
    np.set_printoptions(suppress=True)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))
    # print('newcameramtx外参', newcameramtx)
    total_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        total_error += error
    # 单位是像素
    # print("average error: ", total_error / len(img_points2))
    return mtx, dist.tolist(), corners2[0][0], corners2[8][0], corners2[45][0], corners2[53][0]


def cal_right2(h, w, image_path_list):
    # h, w: 棋盘格角点规格
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    # 获取标定板角点的在世界坐标系位置， 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    '''
    objp:
    [[0. 0. 0.]
    [1. 0. 0.]
    [2. 0. 0.]
    ...
    [4. 4. 0.]
    [5. 4. 0.]
    [6. 4. 0.]]
    '''
    objp = np.zeros((h * w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)
    obj_points = []
    img_points = []
    for image_path in image_path_list:
        img = cv2.imread(image_path)
        u, v = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (h, w), None)

        if ret:
            obj_points.append(objp)
            # 专门用来获取棋盘图上内角点的精确位置的， 即在原角点的基础上寻找亚像素角点
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            if [corners2]:
                img_points.append(corners2)
                # print(img_points)
            else:
                img_points.append(corners)
            cv2.drawChessboardCorners(img, (h, w), corners, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
    # print(img_points)
    # print(len(img_points))
    cv2.destroyAllWindows()
    # calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    # makecsv.makemtx_right(mtx)
    # makecsv.makedist_right(dist)
    np.set_printoptions(suppress=True)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))
    # print('newcameramtx外参', newcameramtx)
    total_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        total_error += error
    # 单位是像素
    # print("average error: ", total_error / len(img_points2))
    return mtx, dist.tolist(), corners2[0][0], corners2[8][0], corners2[45][0], corners2[53][0]


# get_parm()

