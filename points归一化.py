import numpy as np
import os
import cv2

for i in range(192,193):
    root = '../original_data_whole/'
    json_file_path = root+str(i)+'/6.json'
    if not os.path.exists(json_file_path):
        continue

    point_i = np.loadtxt(root + str(i) + '/6_cut.txt',dtype=float)[:,:]
    img = cv2.imread(root+str(i)+'/6_cut.jpg')
    h = img.shape[0]/512
    w = img.shape[1]/512
    cut_points = np.zeros((point_i.shape))
    for j in range(point_i.shape[0]):
        cut_points[j, 0] = int(point_i[j, 0] / w)
        cut_points[j, 1] = int(point_i[j, 1] / h)
        cut_points[j, 2] = int(point_i[j, 2])

    with open(root+str(i)+'/norm_6cut.txt', "w") as f:
        np.savetxt(f,cut_points, fmt='%d')



