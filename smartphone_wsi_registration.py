from PIL import Image
import cv2
import glob
import pandas as pd
import os
import numpy as np
import collections
import pickle
import os  
import openslide
import numpy as np
from tqdm import tqdm
from math import ceil
import shutil
import argparse

def tile_saving(path, window_size, save_folder_png, ):
    source_slide = openslide.open_slide(path)
    width, height= source_slide.dimensions[0], source_slide.dimensions[1]
    window_size = window_size
    increment_x = int(ceil(width / window_size))
    increment_y = int(ceil(height / window_size))
    
    for incre_x in range(increment_x):  # have to read the image in patches since it doesn't let me do it for larger things
        for incre_y in range(increment_y):

            top_left_x = window_size * incre_x
            top_left_y = window_size * incre_y
            
            tile_image = source_slide.read_region((top_left_x, top_left_y), level=0, size=(window_size, window_size))

            tile_png= tile_image.convert('RGB')
            tile_png.save(os.path.join(save_folder_png, os.path.splitext(os.path.basename(path))[0][:-4]+"_"+ str(top_left_x)+"_"+str(top_left_y)+'.png'))
            
def find_match(WSI_folder,q):
    q = q
    WSIs = sorted(glob.glob(WSI_folder+"/*.png"))
    goods = []
    records = []

    for j in tqdm(range(len(WSIs))):
        Q =  cv2.resize(cv2.imread(q,0), dsize=None, fx=0.7, fy=0.7)
        query = cv2.rotate(Q, cv2.ROTATE_180)
        target = cv2.resize(cv2.imread(WSIs[j],0),dsize=None, fx=0.35, fy=0.35)
        akaze = cv2.AKAZE_create()
        float_kp, float_des = akaze.detectAndCompute(query, None)
        ref_kp, ref_des = akaze.detectAndCompute(target, None)

        bf = cv2.BFMatcher()
        try:
            matches = bf.knnMatch(float_des, ref_des, k=2)

            good_matches = []

            for m, n in matches:
                if m.distance < 0.70 * n.distance:
                    good_matches.append([m])


            src_pts = np.float32([ float_kp[m[0].queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ ref_kp[m[0].trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            #print(matchesMask)

            h,w = query.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            dst2 = dst*(dst>=0)

            goods.append(len(good_matches))
            records.append((src_pts, dst_pts, dst2, WSIs[j],j))

        except:
            continue
    return goods, records



def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2)),((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                        (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2)
                           
def crop_from_WSI(goods, records, q, result_dir, crop_size):
    """
    Crop the corresponding region from a WSI (Whole Slide Image) 
    based on matching with a given smartphone image.
    
    Parameters:
    goods : list
        List of good matching scores.
    records : list
        List of matching records.
    q : str
        Path to the query smartphone image.
    result_dir : str
        Directory where to save the resulting cropped image.
    crop_size : int
        Size of the final cropped region.
    """
    # Find the best-matching WSI
    match_num = goods
    ps = records
    ps_max = ps[match_num.index(max(match_num))]
    
    src_pts, dst_pts, dst2, WSI_path, _ = ps_max

    # Step 1: Crop the corresponding part from the best-matching WSI
    img = cv2.resize(cv2.imread(WSI_path), dsize=None, fx=0.35, fy=0.35)
    x, y = get_coordinates_from_filename(WSI_path)
    cropped_resized, rect = crop_corresponding_region(img, dst2)

    # Step 2: Register the smartphone image to the WSI
    converted_img = register_images(q, cropped_resized)

    # Save the final registered image
    save_registered_image(converted_img, rect, result_dir, x, y, crop_size)
    

def get_coordinates_from_filename(WSI_path):
    """
    Extracts the coordinates from the filename of a WSI.
    """
    x = int(os.path.splitext(os.path.basename(WSI_path))[0].split("_")[-2])
    y = int(os.path.splitext(os.path.basename(WSI_path))[0].split("_")[-1])
    return x, y


def crop_corresponding_region(img, dst2):
    """
    Crops a region from a WSI that corresponds to the smartphone image.
    """
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.int32(dst2).reshape(1, -1, 2)

    cv2.fillPoly(mask, points, (255))
    res = cv2.bitwise_and(img, img, mask=mask)

    rect = cv2.boundingRect(points)
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    cropped_resized = cv2.resize(cropped, dsize=None, fx=1/0.35, fy=1/0.35)

    return cropped_resized, rect


def register_images(q, ref_img):
    """
    Registers a smartphone image to a WSI.
    """
    # Perform the registration (example for grayscale images)
    float_img = cv2.imread(q, cv2.IMREAD_GRAYSCALE)
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    # Detect features and compute descriptors
    akaze = cv2.AKAZE_create()
    float_kp, float_des = akaze.detectAndCompute(float_img, None)
    ref_kp, ref_des = akaze.detectAndCompute(ref_img_gray, None)

    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(float_des, ref_des, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Compute Homography
    ref_matched_kpts = np.float32([float_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    sensed_matched_kpts = np.float32([ref_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(ref_matched_kpts, sensed_matched_kpts, cv2.RANSAC, 5.0)

    # Warp the image
    color_img = cv2.imread(q)
    converted_img = cv2.warpPerspective(color_img, H, (float_img.shape[1], float_img.shape[0]))

    return converted_img


def save_registered_image(converted_img, rect, result_dir, x, y, crop_size):
    """
    Saves the final registered image.
    """
    left_upper_x = str(int(rect[0] * (1 / 0.35)) + x)
    left_upper_y = str(int(rect[1] * (1 / 0.35)) + y)
    right_lower_x = str(int((rect[0] + rect[2]) * (1 / 0.35)) + x)
    right_lower_y = str(int((rect[1] + rect[3]) * (1 / 0.35)) + y)

    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, f"{left_upper_x}_{left_upper_y}_{right_lower_x}_{right_lower_y}.png")
    Image.fromarray(converted_img).save(save_path, quality=100)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Process WSI images")
    parser.add_argument('--q', default="smartphone.png", type=str, help="Query image filename")
    parser.add_argument('--path', default="GIVH_AT2.svs", type=str, help="Path to WSI file")
    parser.add_argument('--result_dir', default="/save_dir", type=str, help="Directory to save the results")
    parser.add_argument('--crop_size', default=512, type=int, help="Crop size")
    parser.add_argument('--window_size', default=4096, type=int, help="Window size for tiling")
    parser.add_argument('--save_folder_png', default="wsi_patch_4096", type=str, help="Folder to save PNG patches")

    args = parser.parse_args()
    tile_saving(args.path, args.window_size, args.save_folder_png)
    good, records = find_match(args.save_folder_png, args.q)
    crop_from_WSI(good, records, args.q, args.result_dir, args.crop_size)
