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
import glob
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
            
            #重複なし
            top_left_x = window_size * incre_x
            top_left_y = window_size * incre_y
            
            tile_image = source_slide.read_region((top_left_x, top_left_y), level=0, size=(window_size, window_size))

            #PILへ変換
            tile_png= tile_image.convert('RGB')
            #print(type(image_target_trans))
            tile_png.save(os.path.join(save_folder_png, os.path.splitext(os.path.basename(path))[0][:-4]+"_"+ str(top_left_x)+"_"+str(top_left_y)+'.png'))
            #numpyの保存
            #np.save(os.path.join(save_folder_np, os.path.splitext(os.path.basename(path))[0][:-4]+"_"+ str(top_left_x)+"_"+ str(top_left_y)), np.array(tile_png))
            
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

                #import pickle
                #with open(folder+"/"+str('{0:03d}'.format(j))+"_"+str(len(good_matches))+".pickle", mode="wb") as f:
                #    pickle.dump((src_pts, dst_pts, dst2), f)

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
                           
def crop_from_WSI(goods, records,q, result_dir, crop_size):
        WSIs = sorted(glob.glob(WSI_folder+"/*.png"))
        
        match_num=goods
        ps = records
 
        ps_max = ps[match_num.index(max(match_num))]
        WSI_n = WSIs[match_num.index(max(match_num))]
        


        src_pts, dst_pts ,dst2, WSI_path, _ = ps_max[0],ps_max[1],ps_max[2],ps_max[3], ps_max[4]
        
        """  
        1. 最もmatching pointsが多かったWSIの画像から、スマホの画像に相当する部分を切り出す。
        """  

        img =  cv2.resize(cv2.imread(WSI_path),dsize=None, fx=0.35, fy=0.35)#cv2.imread(GIVH_AT2[11])
        
        x= int(os.path.splitext(os.path.basename(WSI_path))[0].split("_")[-2]) #######全体のWSIにおいてのx座標
        y= int(os.path.splitext(os.path.basename(WSI_path))[0].split("_")[-1]) #######全体のWSIにおいてのy座標
        
        height = img.shape[0]
        width = img.shape[1]

        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.int32(dst2).reshape(1,-1,2)#### 他のWSIを切り取るために保存が必要。#np.array([[[10,150],[150,100],[300,150],[350,100],[310,20],[35,10]]])

        #points.shape
        cv2.fillPoly(mask, points, (255))

        res = cv2.bitwise_and(img,img,mask = mask)

        rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
        cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
        #cropped =  cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        
        cropped_resized = cv2.resize(cropped, dsize=None, fx=1/0.35, fy=1/0.35)
        
        left_upper_x = str(int(rect[0]*(1/0.35)) + x)
        left_upper_y  = str( int(rect[1]*(1/0.35)) + y)
        right_lower_x =  str(int(( rect[0] + rect[2])*(1/0.35)) + x)
        right_lower_y =   str(int((rect[1] + rect[3])*(1/0.35)) + y)

        """
        2. スマートフォン写真(iphone6)をWSI(AT)の方に合わせる
        """
        #########1回目レジストレーション#########
        #iphone６をまず合わせる方にした
        float_img_1 = cv2.imread(q[ps_max], cv2.COLOR_BGR2GRAY)##cv.imread('img/float.jpg', cv.IMREAD_GRAYSCALE)
        ref_img_1 = cv2.cvtColor(cropped_resized, cv2.IMREAD_GRAYSCALE)#cv.imread(iphone6_givh[2], cv.IMREAD_GRAYSCALE)

        akaze = cv2.AKAZE_create()
        float_kp_1, float_des_1 = akaze.detectAndCompute(float_img_1, None)
        ref_kp_1, ref_des_1 = akaze.detectAndCompute(ref_img_1, None)
        

        bf_1 = cv2.BFMatcher()
        matches_1 = bf_1.knnMatch(float_des_1, ref_des_1, k=2)
            
        good_matches_1 = []
        
        for mm, nn in matches_1:
            if mm.distance < 0.75 * nn.distance:
                good_matches_1.append([mm])

        # 適切なキーポイントを選択
        ref_matched_kpts_1 = np.float32(
                            [float_kp_1[m_1[0].queryIdx].pt for m_1 in good_matches_1]).reshape(-1, 1, 2)
        sensed_matched_kpts_1 = np.float32(
                            [ref_kp_1[m_1[0].trainIdx].pt for m_1 in good_matches_1]).reshape(-1, 1, 2)

        # ホモグラフィを計算
        H_1, status_1 = cv2.findHomography(
                            ref_matched_kpts_1, sensed_matched_kpts_1, cv2.RANSAC, 5.0)

        color_img = cv2.imread(q, cv2.COLOR_BGR2RGB)#cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        # 画像を変換
        warped_image_0 = cv2.warpPerspective(
                            color_img[:,:,0], H_1, (float_img_1.shape[1], float_img_1.shape[0]))
        warped_image_1 = cv2.warpPerspective(
                            color_img[:,:,1], H_1, (float_img_1.shape[1], float_img_1.shape[0]))
        warped_image_2 = cv2.warpPerspective(
                            color_img[:,:,2], H_1, (float_img_1.shape[1], float_img_1.shape[0]))

        converted_img = np.stack([warped_image_0, warped_image_1, warped_image_2], -1)
        
        #########2回目レジストレーション#########
        float_img_2 = cv2.cvtColor(converted_img, cv2.COLOR_BGR2GRAY)##cv.imread('img/float.jpg', cv.IMREAD_GRAYSCALE)
        ref_img_2 = cv2.cvtColor(cropped_resized, cv2.IMREAD_GRAYSCALE)#cv.imread(iphone6_givh[num], cv.IMREAD_GRAYSCALE)#cv.imread(iphone6_givh[1], cv.IMREAD_GRAYSCALE)

        akaze = cv2.AKAZE_create()
        float_kp_2, float_des_2 = akaze.detectAndCompute(float_img_2, None)
        ref_kp_2, ref_des_2 = akaze.detectAndCompute(ref_img_2, None)

        bf_2 = cv2.BFMatcher()
        matches_2 = bf_2.knnMatch(float_des_2, ref_des_2, k=2)

        good_matches_2 = []
        for mm, nn in matches_2:
            if mm.distance < 0.75 * nn.distance:
                good_matches_2.append([mm])

        # 適切なキーポイントを選択
        ref_matched_kpts_2 = np.float32(
            [float_kp_2[m_2[0].queryIdx].pt for m_2 in good_matches_2]).reshape(-1, 1, 2)
        sensed_matched_kpts_2 = np.float32(
            [ref_kp_2[m_2[0].trainIdx].pt for m_2 in good_matches_2]).reshape(-1, 1, 2)
        
        
        # ホモグラフィを計算
        H_2, status_2 = cv2.findHomography(
                ref_matched_kpts_2, sensed_matched_kpts_2, cv2.RANSAC, 5.0)

        warped_image_0_1 = cv2.warpPerspective(
                converted_img[:,:,0], H_2, (float_img_2.shape[1], float_img_2.shape[0]))
        warped_image_1_1 = cv2.warpPerspective(
                converted_img[:,:,1], H_2, (float_img_2.shape[1], float_img_2.shape[0]))
        warped_image_2_1 = cv2.warpPerspective(
                converted_img[:,:,2], H_2, (float_img_2.shape[1], float_img_2.shape[0]))

        converted_img_1 = np.stack([warped_image_0_1, warped_image_1_1, warped_image_2_1], -1)

        image, coordinate = crop_center(Image.fromarray(cropped_resized),crop_size,crop_size)
        cropped_convert = Image.fromarray(converted_img_1).crop(coordinate)

        os.makedirs(result_dir, exist_ok=True)
        cropped_convert.save(result_dir+"/"+left_upper_x+"_"+left_upper_y+"_"+right_lower_x+"_"+right_lower_y+".png",quality=100)

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