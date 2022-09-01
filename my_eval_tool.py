import cv2, sys
import numpy as np
from draw_ellipse import *
from ellipses import *
from bwperim_2 import *
np.set_printoptions(threshold=sys.maxsize)

class Eval_tool:
    def __init__(self):
        pass

    def get_ellipse_info(self,img_mask):
        ellipse_info = fit_ellipse_compact(img_mask)
        if ellipse_info:
            return ellipse_info
        return None

    def get_ellipse_centers(self,img_mask):
        ellipse_info = self.get_ellipse_info(img_mask)
        if ellipse_info:
            center, ellipse_w, ellipse_h, ellipse_radian = ellipse_info
            return center[0],center[1]
        return None

    def get_roundness(self, img_mask):
        island = isolate_islands(img_mask)
        perim = bwperim(island)
        ellipse_info = gen_ellipse_contour_perim_compact(perim)
        if ellipse_info:
            center, w, h, radian = ellipse_info
            if w and h:
                if w > h:
                    return h/w
                else:
                    return w/h
        return 0

    def get_calib_ellipse_info1(self,img_mask):
        island = isolate_islands(img_mask)
        perim = bwperim(island)

        retval, labelled, stats, centroids = cv2.connectedComponentsWithStats(island.astype(np.uint8))
        if retval > 1:
            x,y,w,h,area = stats[1]
            h_max = int(h*0.3)
            labelled[:y+h_max,:] = 0
            perim = np.where(labelled==1,perim,0)
            
            ellipse_info = gen_ellipse_contour_perim_compact(perim)
            if ellipse_info:
                return ellipse_info

        return None

    def get_calib_ellipse_centers1(self,img_mask):
        ellipse_info = self.get_calib_ellipse_info1(img_mask)
        if ellipse_info:
            center, w, h, radian = ellipse_info
            return center[0], center[1]

        return None
        
    def get_calib_roundness1(self,img_mask):
        ellipse_info = self.get_calib_ellipse_info1(img_mask)
        if ellipse_info:
            center, w, h, radian = ellipse_info
            if w and h:
                    if w > h:
                        return h/w
                    else:
                        return w/h
        return 0
        
    

    def get_calib_ellipse_info2(self,img_mask):
        island = isolate_islands(img_mask)
        perim = bwperim(island)

        retval, labelled, stats, centroids = cv2.connectedComponentsWithStats(island.astype(np.uint8))
        if retval > 1:
            x,y,w,h,area = stats[1]
            h_max = int(h*0.5)
            labelled[:y+h_max,x+1:x+w-1] = 0
            perim = np.where(labelled==1,perim,0)
            
            ellipse_info = gen_ellipse_contour_perim_compact(perim)
            if ellipse_info:
                return ellipse_info
                    
        return None

    def get_calib_ellipse_centers2(self,img_mask):
        ellipse_info = self.get_calib_ellipse_info2(img_mask)
        if ellipse_info:
            center, w, h, radian = ellipse_info
            return center[0], center[1]

        return None
        
    def get_calib_roundness2(self,img_mask):
        ellipse_info = self.get_calib_ellipse_info2(img_mask)
        if ellipse_info:
            center, w, h, radian = ellipse_info
            if w and h:
                    if w > h:
                        return h/w
                    else:
                        return w/h
        return 0
        
        

    def get_binary_dice_score(self, img1, img2):
        non1 = img1[np.nonzero(img1)]
        non2 = img2[np.nonzero(img2)]

        if non1.size != 0 :
            if np.mean(non1) == 255:
                img1 = img1/255.
                non1 = img1[np.nonzero(img1)]
        if non2.size != 0 :
            if np.mean(non2) == 255:
                img2 = img2/255.
                non2 = img2[np.nonzero(img2)]
        
        if non1.size != 0 :
            if np.mean(non1) != 1:
                # np.set_printoptions(threshold=sys.maxsize)
                # print(img1)
                print("mean of img1's nonzero :",np.mean(non1))
                raise ValueError("binary imgs not found. if the image has been resized, it's highly likely to non binary. use get_dice_score")
        if non2.size != 0 :
            if np.mean(non2) != 1:
                # np.set_printoptions(threshold=sys.maxsize)
                # print(img2)
                print("mean of img2's nonzero :",np.mean(non2))
                raise ValueError("binary imgs not found. if the image has been resized, it's highly likely to non binary. use get_dice_score")

        # img1_f = K.flatten(img1)
        img1_f = img1.reshape(1,-1)
        # img2_f = K.flatten(img2)
        img2_f = img2.reshape(1,-1)
        intersection = np.sum(img1_f * img2_f)
        # return (2. * intersection + np.epsilon()) / (K.sum(img1_f) + K.sum(img2_f) + K.epsilon())
        return (2. * intersection + np.finfo(float).eps) / (np.sum(img1_f) + np.sum(img2_f) + np.finfo(float).eps)


    def get_dice_score(self, img1, img2):

        # img1_f = K.flatten(img1)
        img1_f = img1.reshape(1,-1)
        # img2_f = K.flatten(img2)
        img2_f = img2.reshape(1,-1)
        intersection = np.sum(img1_f * img2_f)
        # return (2. * intersection + np.epsilon()) / (K.sum(img1_f) + K.sum(img2_f) + K.epsilon())
        return (2. * intersection + np.finfo(float).eps) / (np.sum(img1_f) + np.sum(img2_f) + np.finfo(float).eps)

    def draw_with_ellipse_info(self,img_src,ellipse_info, color=(0,0,255)):
        img_result = img_src.copy()
        center, ellipse_w, ellipse_h, ellipse_radian = ellipse_info
        ellipse_center_x = center[0]
        ellipse_center_y = center[1]

        img_result = cv2.ellipse(img_result, (int(ellipse_center_x), int(ellipse_center_y)), (int(ellipse_w), int(ellipse_h)), int(np.rad2deg(ellipse_radian)),0,360, color, 1)    
        img_result = cv2.circle(img_result, (int(ellipse_center_x), int(ellipse_center_y)), 1,color,-1)

        return img_result

    def draw_roi_ellipse(self,img_src,img_mask,color=(0,255,0)):
        if np.ndim(img_src) != 3:
            raise ValueError("img's dimension is incorrect")

        img_result = img_src.copy()
        ellipse_info = fit_ellipse_compact(img_mask)
        if ellipse_info:
            center, ellipse_w, ellipse_h, ellipse_radian = ellipse_info
            ellipse_center_x = center[0]
            ellipse_center_y = center[1]
            img_result = cv2.ellipse(img_result, (int(ellipse_center_x), int(ellipse_center_y)), (int(ellipse_w), int(ellipse_h)), int(np.rad2deg(ellipse_radian)),0,360, color, 1)    
            img_result = cv2.circle(img_result, (int(ellipse_center_x), int(ellipse_center_y)), 1,color,-1)
        
        return img_result

    def draw_pred_ellipse(self,img_src,img_mask, color=(0,0,255)):
        img_result = self.draw_roi_ellipse(img_src,img_mask,color)
        return img_result

    def draw_pred_calib_ellipse1(self, img_src, img_mask, color=(0,0,255)):
        if np.ndim(img_src) != 3:
            raise ValueError("img's dimension is incorrect")
        img_result = img_src.copy()
        if not fit_ellipse_compact(img_mask):
            return img_result

        ellipse_info = self.get_calib_ellipse_info1(img_mask)
        if ellipse_info:
            center, w, h, radian = ellipse_info
            img_result = cv2.ellipse(img_result, (int(center[0]), int(center[1])), (int(w), int(h)), int(np.rad2deg(radian)),0,360, color, 1)    
            img_result = cv2.circle(img_result, (int(center[0]), int(center[1])), 1,color,-1)
                    
        return img_result

    def draw_pred_calib_ellipse2(self, img_src, img_mask, color=(0,0,255)):
        if np.ndim(img_src) != 3:
            raise ValueError("img's dimension is incorrect")
        img_result = img_src.copy()
        if not fit_ellipse_compact(img_mask):
            return img_result

        ellipse_info = self.get_calib_ellipse_info2(img_mask)
        if ellipse_info:
            center, w, h, radian = ellipse_info
            img_result = cv2.ellipse(img_result, (int(center[0]), int(center[1])), (int(w), int(h)), int(np.rad2deg(radian)),0,360, color, 1)    
            img_result = cv2.circle(img_result, (int(center[0]), int(center[1])), 1,color,-1)

        return img_result

    def draw_roi(self,img_src,img_mask, color=(0,255,0)):
        if np.ndim(img_src) != 3:
            raise ValueError("img's dimension is incorrect")
        img_result = img_src.copy()
        island = isolate_islands(img_mask)
        perim = bwperim(island)
        
        img_result[perim != 0] = color

        return img_result

    def draw_pred_roi(self, img_src, img_mask, color=(0,0,255)):
        img_result = self.draw_roi(img_src,img_mask,color)
        return img_result


    def draw_results(self,img_original_3c, pred, mask,index=-1,located='NAN',fx=2,fy=2):
        if np.ndim(img_original_3c) != 3:
            raise ValueError("palate image should have dimention 3")
        if img_original_3c.shape[:2] != pred.shape or img_original_3c.shape[:2] != mask.shape :
            raise ValueError("palate image and pred, mask should have same (height,width)")

        original_3c = img_original_3c
        result_img1 = self.draw_roi_ellipse(original_3c,mask)
        result_img1 = self.draw_pred_ellipse(result_img1,pred)

        result_img2 = self.draw_roi(original_3c,mask)
        result_img2 = self.draw_pred_roi(result_img2,pred)
        # result_img2 = cv2.putText(result_img2,f'dice_score:{dice_score_}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        dice_score_2 = self.get_dice_score(mask,pred)
        dice_score_2 = round(dice_score_2,5)
        result_img3 = self.draw_pred_calib_ellipse1(original_3c,pred)
        result_img4 = self.draw_pred_calib_ellipse2(original_3c,pred)

        result_img1_resized = cv2.resize(result_img1,(0,0),fx=fx,fy=fy,interpolation=cv2.INTER_LINEAR)
        result_img2_resized = cv2.resize(result_img2,(0,0),fx=fx,fy=fy,interpolation=cv2.INTER_LINEAR)
        # result_img2_resized = cv2.putText(result_img2_resized,f'dice_score:{dice_score_}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        result_img3_resized = cv2.resize(result_img3,(0,0),fx=fx,fy=fy,interpolation=cv2.INTER_LINEAR)
        result_img4_resized = cv2.resize(result_img4,(0,0),fx=fx,fy=fy,interpolation=cv2.INTER_LINEAR)

        # result = np.concatenate((result_img1_resized,result_img2_resized,result_img3_resized),axis=1)
        # print("index : ",index)
        # print("mask center : ", eval_tool.get_ellipse_centers(masks_lt_pupil[index]), ", sum :", np.sum(masks_lt_pupil[index]))
        # print("pred center : ", eval_tool.get_ellipse_centers(imgs_lt_saved[index]), ", sum :", np.sum(imgs_lt_saved[index]))
        
        x,y,w,h = cv2.selectROI(f"index : {index}, dice_score : {dice_score_2}",result_img2_resized,False)
        if w and h :
            result_img1_resized_cropped = result_img1_resized[y:y+h, x:x+w]
            result_img1_resized_cropped = cv2.putText(result_img1_resized_cropped,f'{located}, index:{index}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            
            result_img1_resized_cropped = cv2.putText(result_img1_resized_cropped,f'r:{round(self.get_roundness(mask),4)}',(int(w* 0/10) + 10,int(h * 4/5)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0))
            result_img1_resized_cropped = cv2.putText(result_img1_resized_cropped,f'r:{round(self.get_roundness(pred),4)}',(int(w* 6/10),int(h * 4/5)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255))

            result_img2_resized_cropped = result_img2_resized[y:y+h, x:x+w]
            result_img2_resized_cropped = cv2.putText(result_img2_resized_cropped,f'dice_score:{dice_score_2}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            result_img3_resized_cropped = result_img3_resized[y:y+h, x:x+w]
            result_img4_resized_cropped = result_img4_resized[y:y+h, x:x+w]
            npimg_result_whole_cropped = np.concatenate([result_img1_resized_cropped, result_img2_resized_cropped, result_img3_resized_cropped, result_img4_resized_cropped], axis=1)
            winname = f'index:{index}, dice_score:{dice_score_2}'
            cv2.imshow(winname, npimg_result_whole_cropped)
            cv2.moveWindow(winname, 20, 300)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    
    def draw_results2(self,img_original_3c, pred1, pred2, mask,index=-1,located='NAN',fx=2,fy=2):
        if np.ndim(img_original_3c) != 3:
            raise ValueError("palate image should have dimention 3")
        if img_original_3c.shape[:2] != pred1.shape or img_original_3c.shape[:2] != pred2.shape or img_original_3c.shape[:2] != mask.shape :
            raise ValueError("palate image and pred1, pred2, mask should have same (height,width)")


        original_3c = img_original_3c
        result_img1 = self.draw_roi_ellipse(original_3c,mask)
        result_img1 = self.draw_pred_ellipse(result_img1,pred1)

        result_img2 = self.draw_roi(original_3c,mask)
        result_img2 = self.draw_pred_roi(result_img2,pred1,(0,0,255))
        result_img2 = self.draw_pred_roi(result_img2,pred2,(122,122,0))
        # result_img2 = cv2.putText(result_img2,f'dice_score:{dice_score_}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        dice_score_2 = self.get_dice_score(mask,pred1)
        dice_score_2 = round(dice_score_2,5)
        result_img3 = self.draw_pred_calib_ellipse1(original_3c,pred1)
        result_img4 = self.draw_pred_calib_ellipse2(original_3c,pred1)

        result_img1_resized = cv2.resize(result_img1,(0,0),fx=fx,fy=fy,interpolation=cv2.INTER_LINEAR)
        result_img2_resized = cv2.resize(result_img2,(0,0),fx=fx,fy=fy,interpolation=cv2.INTER_LINEAR)
        # result_img2_resized = cv2.putText(result_img2_resized,f'dice_score:{dice_score_}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        result_img3_resized = cv2.resize(result_img3,(0,0),fx=fx,fy=fy,interpolation=cv2.INTER_LINEAR)
        result_img4_resized = cv2.resize(result_img4,(0,0),fx=fx,fy=fy,interpolation=cv2.INTER_LINEAR)

        # result = np.concatenate((result_img1_resized,result_img2_resized,result_img3_resized),axis=1)
        # print("index : ",index)
        # print("mask center : ", eval_tool.get_ellipse_centers(masks_lt_pupil[index]), ", sum :", np.sum(masks_lt_pupil[index]))
        # print("pred center : ", eval_tool.get_ellipse_centers(imgs_lt_saved[index]), ", sum :", np.sum(imgs_lt_saved[index]))
        
        x,y,w,h = cv2.selectROI(f"index : {index}, dice_score : {dice_score_2}",result_img2_resized,False)
        if w and h :
            result_img1_resized_cropped = result_img1_resized[y:y+h, x:x+w]
            result_img1_resized_cropped = cv2.putText(result_img1_resized_cropped,f'{located}, index:{index}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            
            result_img1_resized_cropped = cv2.putText(result_img1_resized_cropped,f'r:{round(self.get_roundness(mask),4)}',(int(w* 0/10) + 10,int(h * 4/5)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0))
            result_img1_resized_cropped = cv2.putText(result_img1_resized_cropped,f'r:{round(self.get_roundness(pred1),4)}',(int(w* 6/10),int(h * 4/5)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255))

            result_img2_resized_cropped = result_img2_resized[y:y+h, x:x+w]
            result_img2_resized_cropped = cv2.putText(result_img2_resized_cropped,f'dice_score:{dice_score_2}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            result_img3_resized_cropped = result_img3_resized[y:y+h, x:x+w]
            result_img4_resized_cropped = result_img4_resized[y:y+h, x:x+w]
            npimg_result_whole_cropped = np.concatenate([result_img1_resized_cropped, result_img2_resized_cropped, result_img3_resized_cropped, result_img4_resized_cropped], axis=1)
            winname = f'index:{index}, dice_score:{dice_score_2}'
            cv2.imshow(winname, npimg_result_whole_cropped)
            cv2.moveWindow(winname, 20, 300)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    

    def get_img_with_mask(self, img, mask, modify_rate=1):
        add_value = 20

        if not isinstance(mask,numpy.ndarray):
            return 0
        if not isinstance(img,numpy.ndarray):
            return 0
        if np.ndim(img) != 3:
            return 0
        if img.shape[0] != mask.shape[0] or img.shape[1] != mask.shape[1]:
            return 0

        if modify_rate < 0:
            modify_rate = 1

        if np.count_nonzero(mask) == 0:
            return img
        
        img = img.copy()
        mask = mask.copy()
        palate_resized = np.zeros((img.shape[0] * modify_rate, img.shape[1] * modify_rate, 3), dtype=np.uint8)

        indices = np.where(mask != 0)
        for i in range(len(indices[0])):
            y = indices[0][i]
            x = indices[1][i]

            r = img[y, x, 2] 
            r += add_value
            img[y, x, 2] = r

        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                palate_resized[y*modify_rate: y*modify_rate+modify_rate, x*modify_rate: x*modify_rate + modify_rate] = img[y,x]

        return palate_resized


