import cv2
import numpy as np

calibrate_board_size = (7, 7)
calibrate_board_circle_gap = 12.5

src_img = cv2.imread('./img/2.jpg', cv2.IMREAD_ANYCOLOR)
src_img = src_img[600:1500, 100:1100]
gray_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)

###找标定板圆心###
retval, image_point = cv2.findCirclesGrid(gray_img, calibrate_board_size, cv2.CALIB_CB_SYMMETRIC_GRID)
cv2.drawChessboardCorners(src_img, calibrate_board_size, image_point, True)
cv2.imshow('img', src_img)

###对圆心进行亚像素精度运算###
stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                 30, 0.001)
image_point = cv2.cornerSubPix(gray_img, image_point, calibrate_board_size, (-1, -1), stop_criteria)

###标定相机###
worldPoint = []
for i in range(calibrate_board_size[0]):
    for j in range(calibrate_board_size[1]):
        worldPoint.append([float(i * calibrate_board_circle_gap), float(j * calibrate_board_circle_gap), float(0)])
worldPoint = np.array(worldPoint, dtype='float32')
image_point = np.reshape(image_point, (49, 2))

worldPoints = []
image_points = []

worldPoints.append(worldPoint)
image_points.append(image_point)

retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(worldPoints, image_points,
                                                                       (gray_img.shape[1], gray_img.shape[0]), None,
                                                                       None)
print('-------相机内参矩阵--------')
print(camera_matrix)
print('-------相机畸变矩阵--------')
print(dist_coeffs)
print('-------旋转矩阵--------')
print(rvecs)
print('-------平移矩阵--------')
print(tvecs)

###对图像进行畸变校正###
new_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, gray_img.shape[::-1], 1)
print(_roi)
_x, _y, _w, _h = _roi
src_img = cv2.undistort(src_img, new_camera_matrix, dist_coeffs)
cv2.imshow('undistort img', src_img[_x:_x + _w, _y:_y + _h])

###计算冲投影误差###
mean_error = 0
for i in range(len(worldPoints)):
    reproject_image_point, jacobian = cv2.projectPoints(worldPoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(image_points[i] - reproject_image_point, cv2.NORM_L2) / len(reproject_image_point)
    print(image_points[i] - reproject_image_point)
    mean_error += error
mean_error /= len(worldPoints)
print("calibrate error:" + str(mean_error))

###通过图像坐标计算世界坐标（在有机械手的项目中经常需要用到这个来计算机械手的抓取点）###
'''
前面相机标定求出 1.相机内参矩阵camera_matrix 2.相机畸变矩阵dist_coeffs  3.旋转矩阵rvecs 4.平移矩阵tvecs
根据coordinate_transform_formula.jpg可以推出从图像坐标到世界坐标的转换公式
注：如果是使用张正友标定法，则world_point_z=0,相当于一个平面到另一个平面的投影，假设旋转矩阵R=[rx,ry,rz,t],world_point=[x,y,z,1],z=0可以推出
R可以写为[rx,ry,t]
下面的只是简单演示如何通过图像坐标反推世界坐标，具体推导过程可以看图calibrate_world_point.jpg
'''
worldPoint_index = 30
x = [image_point[worldPoint_index][0], image_point[worldPoint_index][1], 1]
image_point = np.array(x).reshape(3, 1)
image_point = np.asmatrix(image_point)

rotate_matrix = cv2.Rodrigues(rvecs[0])[0]
translate_matrix = tvecs[0]

rotate_matrix = np.asmatrix(rotate_matrix)
translate_matrix = np.asmatrix(translate_matrix)
camera_matrix = np.asmatrix(camera_matrix)

camera_rotate = camera_matrix * rotate_matrix
camera_translate = camera_matrix * translate_matrix
camera_rotate_inv = np.linalg.inv(camera_rotate)

world_point_z = worldPoint[worldPoint_index][2]
matrix1 = camera_rotate_inv * camera_translate

matrix2 = camera_rotate_inv * image_point

world_point_x = ((matrix1[2][0] + world_point_z) * matrix2[0] / matrix2[2]) - matrix1[0]
world_point_y = ((matrix1[2][0] + world_point_z) * matrix2[1] / matrix2[2]) - matrix1[1]
print(worldPoint[worldPoint_index])
print([world_point_x[0], world_point_y[0], worldPoint[worldPoint_index][2]])
cv2.waitKey()
