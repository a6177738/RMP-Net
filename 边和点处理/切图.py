import os
import cv2
import numpy as np

# root = '../original_data_whole/'
# for i in range(266,278):
#     json_file_path = root+str(i)+'/3.json'
#     if not os.path.exists(json_file_path):
#         continue
#     print(i)
#     img = cv2.imread(root+str(i)+'/3.jpg')
#     with open(root + str(i) + '/3.txt', 'r') as f:
#         points = np.loadtxt(f, dtype=int)
#         img1 = img[points[0][1]:points[25][1],:,:]
#         img2 = img1[:,points[18][0]:points[24][0],:]
#         cv2.imwrite(root+str(i)+"/3_cut.jpg",img2)

root = '../original_data_whole/'
for i in range(266,278):
    json_file_path = root+str(i)+'/3.json'
    if not os.path.exists(json_file_path):
        continue
    print(i)
    img = cv2.imread(root+str(i)+'/3.jpg')
    with open(root + str(i) + '/3.txt', 'r') as f:
        points = np.loadtxt(f, dtype=int)
        img1 = img[points[0][1]:points[25][1],:,:]
        img2 = img1[:,points[18][0]:points[24][0],:]
        cv2.imwrite(root+str(i)+"/3_cut.jpg",img2)

# for i in range(274,275):
#     json_file_path = root+str(i)+'/6.json'
#     if not os.path.exists(json_file_path):
#         continue
#     print(i)
#     img = cv2.imread(root+str(i)+'/6.jpg')
#     with open(root + str(i) + '/6.txt', 'r') as f:
#         points = np.loadtxt(f, dtype=int)
#         img1 = img[points[0][1]:points[9][1],:,:]
#         img2 = img1[:,(points[5][0]-20):(points[7][0]+125),:]
#         #img2 = img1[:, (points[7][0] - 125):(points[5][0] + 20), :]
#         cv2.imwrite(root+str(i)+"/6_cut.jpg",img2)



