import numpy as np 
import cv2 
import pandas as pd
import glob
import time 
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import imutils

def pyramid(image, scale=1.5, minSize=(30, 30)):
	yield image
	while True:
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		yield image
		
def sliding_window(image, stepSize, windowSize):
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def extract_orb_des(image):
    orb = cv2.ORB_create(nfeatures=350)
    kp, des = orb.detectAndCompute(image, None)
    return des

def read_bounding_box(path):
    boundary_file = open(path, 'r')
    boundary = boundary_file.readline().strip().split(' ')
    boundary = [int(b) for b in boundary]
    boundary_file.close()
    return boundary

def img_preprocess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img = cv2.cvtColor(img,cv2.COLOR_YUV2BGR) 
    return img 
 
def initialize(download_path):
    path_query=download_path+'/query_4186'
    path_query_txt=download_path+'/query_txt_4186'

    path_gallery=download_path+'/gallery_4186'

    name_query=glob.glob(path_query+'/*.jpg')
    num_query=len(name_query)
    name_gallery=glob.glob(path_gallery+'/*.jpg')
    num_gallery=len(name_gallery)
    record_all=np.zeros((num_query,len(name_gallery)))
    query_imgs_no = [x.split('\\')[-1] for x in glob.glob(path_query+'/*.jpg')]
    query_imgs_no = [x[:-4] for x in query_imgs_no]

    gallery_imgs_no = [x.split('\\')[-1] for x in glob.glob(path_gallery+'/*.jpg')]
    gallery_imgs_no = [x[:-4] for x in gallery_imgs_no]
    return path_gallery, path_query, path_query_txt, gallery_imgs_no, query_imgs_no, record_all, num_query, num_gallery

path='C:/Users/IV/Desktop/CityU'

path_gallery, path_query, path_query_txt, gallery_imgs_no, query_imgs_no, record_all, num_query, num_gallery = initialize(path)

gallery_des = [[] for i in range(num_gallery)]

for i, gallery_img_no in tqdm(enumerate(gallery_imgs_no)):
    per_gallery_name = path_gallery+'/'+str(gallery_img_no)+'.jpg'
    per_gallery=cv2.imread(per_gallery_name)

    # Image pre-processing
    per_gallery = img_preprocess(per_gallery)
    # Define the window size
    winW, winH = (500, 500)

    # loop over the image pyramid
    for resized in pyramid(per_gallery, scale=1.5):
        for (x, y, window) in sliding_window(resized, stepSize=100, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            per_gallery_des = extract_orb_des(window)
            gallery_des[i].append(per_gallery_des)

for i, query_img_no in tqdm(enumerate(query_imgs_no[0:20])):
    time_s = time.time()
    
    score_record=[]
    gallery_imgs_no_desc=[]

    per_query_name=path_query+'/'+str(query_img_no)+'.jpg'
    per_query=cv2.imread(per_query_name)
    
    # read boundary from text file
    queryfilename = path_query_txt+'/'+str(query_img_no)+'.txt'
    
    # crop the image
    boundary = read_bounding_box(queryfilename)
    x ,y, w, h = boundary
    query_boundary = per_query[y:y+h, x:x+w]

    # Image pre-processing
    query_boundary = img_preprocess(query_boundary)

    # feature extraction for per query
    # per_query_des = extract_sift_des(query_boundary)
    per_query_des = extract_orb_des(query_boundary)

    
    # the iteration loop for gallery
    for j, gallery_img_no in tqdm(enumerate(gallery_imgs_no)):
        # Compute similarity score
        sim_scores = []
        
        for k in range(len(gallery_des[j])):
            if gallery_des[j][k] is None:
                continue
            des = gallery_des[j][k]
            padded_gallery_des = np.pad(des, ((0, 355 - des.shape[0]), (0, 0)), mode='constant')
            padded_query_des = np.pad(per_query_des, ((0, 355 - per_query_des.shape[0]), (0, 0)), mode='constant')
            sim_score = cosine_similarity(padded_query_des, padded_gallery_des)
            sim_scores.append(sim_score)

        if len(sim_scores) == 0:
            max_score = 0
        else:
            # print(sim_scores)
            max_score = np.max(sim_scores)
        score_record.append(max_score)

    # find the indexes with descending similarity order
    descend_index=sorted(range(len(score_record)), key=lambda k: np.max(score_record[k]),reverse=True)
    # update the results for one query
    for k in range(len(descend_index)):
        gallery_imgs_no_desc.append(np.array(gallery_imgs_no)[descend_index[k]])

    record_all[i,:]= gallery_imgs_no_desc
    time_e = time.time()
    print('retrieval time for query {} is {}s'.format(query_img_no, time_e-time_s))
    query_idx = i
    print(f'For query image No. {query_imgs_no[query_idx]}, the top 10 ranked similar image No. are {gallery_imgs_no_desc[0]} {gallery_imgs_no_desc[1]} {gallery_imgs_no_desc[2]} {gallery_imgs_no_desc[3]} {gallery_imgs_no_desc[4]} {gallery_imgs_no_desc[5]} {gallery_imgs_no_desc[6]} {gallery_imgs_no_desc[7]} {gallery_imgs_no_desc[8]} {gallery_imgs_no_desc[9] }')
    print(f'For query image No. {query_imgs_no[query_idx]}, the similarity scores are {score_record[descend_index[0]]} {score_record[descend_index[1]]} {score_record[descend_index[2]]} {score_record[descend_index[3]]} {score_record[descend_index[4]]} {score_record[descend_index[5]]} {score_record[descend_index[6]]} {score_record[descend_index[7]]} {score_record[descend_index[8]]} {score_record[descend_index[9]] }')

    # filename=path_query+'/'+str(query_imgs_no[query_idx])+'.jpg'
    # image = mpimg.imread(filename)
    # plt.imshow(image)
    # plt.show()
    # for x in range(10):
    #     filename=path_gallery+'/'+str(gallery_imgs_no_desc[x])+'.jpg'
    #     image = mpimg.imread(filename)
    #     plt.imshow(image)
    #     plt.show()
    # plt.close('all')

# write the output file following the example
f=open(r'./rank_list_SIFT.txt','w')
for i in range(num_query):
    f.write('Q'+str(i+1)+': ')
    for j in range(num_gallery):
        f.write(str(np.int32(record_all[i,j]))+' ')
    f.write('\n')
f.close()

for i in range(num_query):
    print('Q'+str(i+1)+': ',end="")
    for j in range(10):
        print(str(np.int32(record_all[i,j]))+' ',end="")
    print('\n')


