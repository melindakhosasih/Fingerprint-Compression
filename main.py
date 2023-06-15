import sys
from fingerprint_image_enhancer import FingerprintImageEnhancer
import cv2
import fingerprint_minutiae_extractor
import os
import tqdm
import fingerprint_singular_extractor
import numpy as np


folder_path = './SOCOFing/Real/'
# get a list of all files in the folder
files = os.listdir(folder_path)
# sort the files based on the first number
files.sort(key=lambda x: (int(x.split('_')[0]), x))

enhanced_dir = "./enhanced/"
if not os.path.exists(enhanced_dir):
    os.mkdir(enhanced_dir)

compressed_enhanced_dir = "./compressed_enhanced/"
if not os.path.exists(compressed_enhanced_dir):
    os.mkdir(compressed_enhanced_dir)

recognized_num_1 = 0
recognized_num_2 = 0
recognized_num_3 = 0
recognized_num_4 = 0
recognized_num_5 = 0
recognized_num_6 = 0
recognized_num_7 = 0
total_score = 0
num_of_errors = 0
# iterate over the last 1000 files
# for i, file in enumerate(tqdm.tqdm(files[5195:])):
for i, file in enumerate(tqdm.tqdm(files)):
    # enhance fingerprints
    image_enhancer = FingerprintImageEnhancer()         # create object called image_enhancer

    folder_path = './SOCOFing/Real/'
    img = cv2.imread(os.path.join(folder_path, file))
    if(len(img.shape)>2):                               # convert image into gray if necessary
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    out = image_enhancer.enhance(img)     # run image enhancer
    image_enhancer.save_enhanced_image('./enhanced/' + file)   # save output

    folder_path = './compressed/'
    img = cv2.imread(os.path.join(folder_path, file))
    if(len(img.shape)>2):                               # convert image into gray if necessary
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    out = image_enhancer.enhance(img)     # run image enhancer
    image_enhancer.save_enhanced_image('./compressed_enhanced/' + file)   # save output


    # extract features from enhanced fingerprints
    file_1 = file
    img_1 = cv2.imread('./enhanced/' + file_1, 0)				# read the input image --> You can enhance the fingerprint image using the "fingerprint_enhancer" library
    FeaturesTerminations_1, FeaturesBifurcations_1, FeaturesDot_1 = fingerprint_minutiae_extractor.extract_minutiae_features(img_1, file_1, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=True)

    file_2 = file
    img_2 = cv2.imread('./compressed_enhanced/' + file_2, 0)				# read the input image --> You can enhance the fingerprint image using the "fingerprint_enhancer" library
    FeaturesTerminations_2, FeaturesBifurcations_2, FeaturesDot_2 = fingerprint_minutiae_extractor.extract_minutiae_features(img_2, file_2, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False)

    #score-based minutiae matching
    term_match, term_total = fingerprint_minutiae_extractor.calculate_score(FeaturesTerminations_1, FeaturesTerminations_2)
    bif_match, bif_total = fingerprint_minutiae_extractor.calculate_score(FeaturesBifurcations_1, FeaturesBifurcations_2)
    dot_match, dot_total = fingerprint_minutiae_extractor.calculate_score(FeaturesDot_1, FeaturesDot_2)



    stacked_img = np.stack((img_1,)*3, axis=-1)
    

    try:
        detect_SP_1 = fingerprint_singular_extractor.walking(img_1)
    except:
        num_of_errors += 1

    if min(detect_SP_1['core'].shape) !=0:
        for i in range(0, detect_SP_1['core'].shape[0]):
            centre = (int(detect_SP_1['core'][i,0]), int(detect_SP_1['core'][i,1]))
            stacked_img = cv2.circle(stacked_img, centre, 10, (0,0,255), 2)

    if min(detect_SP_1['delta'].shape) !=0:
        for j in range(0, detect_SP_1['delta'].shape[0]):
            x = int(detect_SP_1['delta'][j,0])
            y = int(detect_SP_1['delta'][j,1])
            pts = np.array([[x,y-10], [x-9,y+5], [x+9,y+5]])
            stacked_img = cv2.polylines(stacked_img, [pts], True, (0,255,0), 2)

    cv2.imwrite('./singular/' + file_1, stacked_img)

    # print(detect_SP_1)

    stacked_img = np.stack((img_2,)*3, axis=-1)

    try:
        detect_SP_2 = fingerprint_singular_extractor.walking(img_2)
    except:
        num_of_errors += 1

    if min(detect_SP_2['core'].shape) !=0:
        for i in range(0, detect_SP_2['core'].shape[0]):
            centre = (int(detect_SP_2['core'][i,0]), int(detect_SP_2['core'][i,1]))
            stacked_img = cv2.circle(stacked_img, centre, 9, (0,0,255), 2)

    if min(detect_SP_1['delta'].shape) !=0:
        for j in range(0, detect_SP_2['delta'].shape[0]):
            x = int(detect_SP_2['delta'][j,0])
            y = int(detect_SP_2['delta'][j,1])
            pts = np.array([[x,y-10], [x-9,y+5], [x+9,y+5]])
            stacked_img = cv2.polylines(stacked_img, [pts], True, (0,255,0), 2)

    # cv2.imwrite('./singular/' + file_2, stacked_img)

    # print(detect_SP_2)


    # pattern_1 = fingerprint_singular_extractor.classify_fingerprint_pattern(detect_SP_1)
    # pattern_2 = fingerprint_singular_extractor.classify_fingerprint_pattern(detect_SP_2)

    # if pattern_1 != 'None' and pattern_2 != 'None' and pattern_1 == pattern_2:
    #     pattern = pattern_1

    singular_match, singular_total = fingerprint_singular_extractor.calculate_score(detect_SP_1, detect_SP_2, 2)

    total_match = term_match + bif_match + dot_match + singular_match
    total_points = term_total + bif_total + dot_total + singular_total

    threshold_1 = 0.5
    threshold_2 = 0.55
    threshold_3 = 0.6
    threshold_4 = 0.65
    threshold_5 = 0.7
    threshold_6 = 0.75
    threshold_7 = 0.8
    score = total_match / total_points
    print(score)

    if score >= threshold_1:
        recognized_num_1 += 1
    if score >= threshold_2:
        recognized_num_2 += 1
    if score >= threshold_3:
        recognized_num_3 += 1
    if score >= threshold_4:
        recognized_num_4 += 1
    if score >= threshold_5:
        recognized_num_5 += 1
    if score >= threshold_6:
        recognized_num_6 += 1
    if score >= threshold_7:
        recognized_num_7 += 1

    total_score += score

print('Recognized amount:', recognized_num_1)
print('Recognized amount:', recognized_num_2)
print('Recognized amount:', recognized_num_3)
print('Recognized amount:', recognized_num_4)
print('Recognized amount:', recognized_num_5)
print('Recognized amount:', recognized_num_6)
print('Recognized amount:', recognized_num_7)
print(total_score/1000)
print(num_of_errors)

