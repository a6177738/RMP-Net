import json
import numpy as np
import os
import cv2


for i in range(192,193):
    # 假设json_file_path是您的JSON文件的路径
    json_file_path = '../original_data_whole/'+str(i)+'/6.json'
    if not os.path.exists(json_file_path):
        continue

    # with open('original_data_whole/' + str(i) + '/2.txt', 'r') as f:
    #     points = np.loadtxt(f,dtype=int)
    #     cut_points = np.zeros((points.shape))
    # with open('original_data_whole/' + str(i) + '/2_cut.txt', 'w') as out:
    #     for i in range(points.shape[0]):
    #         cut_points[i][1] = points[i][1]-points[0][1]
    #         cut_points[i][0] = points[i][0]-points[17][0]
    #         cut_points[i][2] = points[i][2]
    #
    #         np.savetxt(out, [cut_points[i]], fmt='%d', delimiter=' ')
    #         out.write('\n')


    # with open('../original_data_whole/' + str(i) + '/3.txt', 'r') as f:
    #     points = np.loadtxt(f,dtype=int)
    #     cut_points = np.zeros((points.shape))
    # with open('../original_data_whole/' + str(i) + '/3_cut.txt', 'w') as out:
    #     for i in range(points.shape[0]):
    #         #if i != 10 and i !=14:
    #             cut_points[i][1] = points[i][1]-points[0][1]
    #             cut_points[i][0] = points[i][0]-points[18][0]
    #             # if i<10:
    #             cut_points[i][2] = points[i][2]-1
    #             # if i>10 and i<14:
    #             #     cut_points[i][2] = points[i][2]-1
    #             # if i>14:
    #             #     cut_points[i][2] = points[i][2]-2
    #
    #             np.savetxt(out, [cut_points[i]], fmt='%d', delimiter=' ')
    #             out.write('\n')


    with open('../original_data_whole/' + str(i) + '/6.txt', 'r') as f:
        points = np.loadtxt(f,dtype=int)
        cut_points = np.zeros((points.shape))
        print(points)
    with open('../original_data_whole/' + str(i) + '/6_cut.txt', 'w') as out:
        for i in range(points.shape[0]):
            cut_points[i][1] = points[i][1]-points[0][1]
            cut_points[i][0] = points[i][0]-(points[5][0]-20)
            #cut_points[i][0] = points[i][0] - (points[7][0] - 125)
            cut_points[i][2] = points[i][2]-1

            np.savetxt(out, [cut_points[i]], fmt='%d', delimiter=' ')
            out.write('\n')
        print(cut_points)