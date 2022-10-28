
import os, cv2, copy, time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from my_eval_tool import Eval_tool

from scipy import signal
from typing import Union, Tuple, List, Set, Dict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
CURRENT_DEVICE = "/device:CPU:0"

class FeatureExtractor:

    def __init__(self):
        self.eval_tool = Eval_tool()
        self.MODEL_HEIGHT = 240
        self.MODEL_WIDTH = 320
    
    def dice_score(img1, img2):
        img1_f = img1.reshape(1,-1)
        img2_f = img2.reshape(1,-1)
        intersection = np.sum(img1_f * img2_f)
        return (2. * intersection + np.finfo(float).eps) / (np.sum(img1_f) + np.sum(img2_f) + np.finfo(float).eps)
    # def dice_score(y_true, y_pred):
        #     y_true_f = K.flatten(y_true)
        #     y_pred_f = K.flatten(y_pred)
        #     intersection = K.sum(y_true_f * y_pred_f)
        #     return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

    def get_dict_from_preds(self, preds) -> dict:
        center_indices = []
        roundnesses = []
        widths = []
        heights = []
        radians = []

        for pred in preds:
            infos = self.eval_tool.get_calib_ellipse_info2(pred)
            if infos:
                center, w, h, radian = infos
                if w > h:
                    roundness = h/w
                else:
                    roundness = w/h
                center_indices.append(list(center))
                widths.append(w)
                heights.append(h)
                radians.append(radian)
                roundnesses.append(roundness)
            else :
                center_indices.append([None, None])
                widths.append(None)
                heights.append(None)
                radians.append(None)
                roundnesses.append(0)
        
        return {'centers': center_indices, 'roundnesses': roundnesses, 'widths': widths, 'heights': heights, 'radians': radians}


    def extract_ellipse_infos_dict_on_video(self, video_file_path: str, MODEL_NAME: str) -> dict:
        ## prepare to resize video frame images
        MODEL_HEIGHT = self.MODEL_HEIGHT
        MODEL_WIDTH = self.MODEL_WIDTH
        
        ## load video
        if os.path.isfile(video_file_path):
            cap = cv2.VideoCapture(video_file_path)
        else:
            print(video_file_path)
            raise ValueError("File not found")

        retval, zero_index_frame = cap.read()
        
        video_frame_height = zero_index_frame.shape[0]
        video_frame_width = zero_index_frame.shape[1]
        video_frame_channel = zero_index_frame.shape[2]
        print('video_frame_shape: ', zero_index_frame.shape)
        frames = [zero_index_frame]        
        
        start_time = time.time()
        ## reading whole of frames of the video
        while True:
            retval, frame = cap.read()
            if not retval:
                print("reading whole frames finished")
                break

            frames.append(frame)

        ## reading frames done
        end_time = time.time()
        print("read video done. time: ", end_time - start_time)
        print("length of frames: ", len(frames))
        start_time = end_time
        cap.release()

        ## resizing frames and convert to gray
        if video_frame_height != MODEL_HEIGHT or video_frame_width != (MODEL_WIDTH *2):
            frames = [cv2.cvtColor(cv2.resize(frame, (MODEL_WIDTH *2, MODEL_HEIGHT), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_RGB2GRAY) for frame in frames]
        end_time = time.time()
        print("resize, convert gray done. time: ", end_time - start_time)
        start_time = end_time

        ## nomalize and seperate frames
        frames = np.stack(frames)
        print('frames shape:', frames.shape)
        frames = frames / 255.
        right_frames = frames[:,:,:MODEL_WIDTH]
        right_frames = right_frames[...,np.newaxis]
        left_frames = frames[:,:,MODEL_WIDTH:]
        left_frames = left_frames[...,np.newaxis]
        print('left_frames shape: ', left_frames.shape)
        print('right_frames shape: ', right_frames.shape)
        end_time = time.time()
        print("normalize, seperate done. time: ", end_time - start_time)
        start_time = end_time

        ## model predict
        with tf.device(CURRENT_DEVICE):
            model = load_model(MODEL_NAME, custom_objects={'dice_score': self.dice_score})
            index = 0

            left_preds = model.predict(left_frames)
            end_time = time.time()
            print("left_frames prediction done. time: ", end_time - start_time)
            start_time = end_time

            right_preds = model.predict(right_frames)
            end_time = time.time()
            print("right_frames prediction done. time: ", end_time - start_time)
            start_time = end_time

            left_preds = np.squeeze(left_preds)
            left_preds = (left_preds > 0.5).astype(np.uint8)
            right_preds = np.squeeze(right_preds)
            right_preds = (right_preds > 0.5).astype(np.uint8)

        left_dict = self.get_dict_from_preds(left_preds)
        right_dict = self.get_dict_from_preds(right_preds)

        return {'left_centers': left_dict["centers"], 'left_roundnesses': left_dict["roundnesses"], 'left_widths': left_dict["widths"], 'left_heights' : left_dict["heights"], 'left_radians' : left_dict["radians"] \
                , 'right_centers': right_dict["centers"], 'right_roundnesses': right_dict["roundnesses"], 'right_widths': right_dict["widths"], 'right_heights': right_dict["heights"], 'right_radians': right_dict["radians"]}

            
    # ## no longer use, will be deprecated
    # def extract_indices(self,video_file_path, MODEL_NAME) -> list:
    #     MODEL_HEIGHT = self.MODEL_HEIGHT
    #     MODEL_WIDTH = self.MODEL_WIDTH
        
    #     if os.path.isfile(video_file_path):
    #         cap = cv2.VideoCapture(video_file_path)
    #     else:
    #         print(video_file_path)
    #         raise ValueError("File not found")
        

    #     l_left_center_axis = []
    #     l_left_roundness = []
    #     l_left_result_imgs = []

    #     l_right_center_axis = []
    #     l_right_roundness = []
    #     l_right_result_imgs = []

    #     with tf.device(CURRENT_DEVICE):
    #         model = load_model(MODEL_NAME, custom_objects={'dice_score': self.dice_score})
            
    #         index = 0
    #         prev_left_x = 0
    #         prev_left_y = 0
    #         prev_right_x = 0
    #         prev_right_y = 0

    #         start_time = time.time()
    #         while True:
    #             retval, frame = cap.read()
    #             if not retval:
    #                 print("model prediction has been finished")
    #                 break
    #             # print(retval,type(frame))
    #             # print(frame.shape)
    #             height = frame.shape[0]
    #             width = frame.shape[1]
    #             channel = frame.shape[2]
                
    #             # cv2.imshow('frame',frame)
    #             # cv2.waitKey(0)
    #             # cv2.destroyAllWindows()

    #             ####### Lt at right, Rt at left on frame ####### 
    #             img_right = frame[:,:int(width/2),:]
    #             img_left = frame[:,int(width/2):,:]

    #             img_left_resized = cv2.resize(img_left,(MODEL_WIDTH, MODEL_HEIGHT),interpolation=cv2.INTER_LANCZOS4)
    #             img_right_resized = cv2.resize(img_right,(MODEL_WIDTH, MODEL_HEIGHT),interpolation=cv2.INTER_LANCZOS4)

    #             img_left_resized_gray = cv2.cvtColor(img_left_resized,cv2.COLOR_BGR2GRAY)
    #             img_right_resized_gray = cv2.cvtColor(img_right_resized,cv2.COLOR_BGR2GRAY)

    #             img_left_resized_gray_norm = img_left_resized_gray / 255.
    #             img_right_resized_gray_norm = img_right_resized_gray / 255.
    #             # print(img_left_resized_gray_norm.shape)

    #             img_left_resized_gray_norm = img_left_resized_gray_norm[:,:,np.newaxis]
    #             img_right_resized_gray_norm = img_right_resized_gray_norm[:,:,np.newaxis]
    #             # print(img_left_resized_gray_norm.shape)
    #             pred_left = model.predict(img_left_resized_gray_norm[np.newaxis,:,:,:])
    #             pred_left = pred_left.squeeze()
    #             pred_left = (pred_left > 0.5).astype(np.uint8)

    #             pred_right = model.predict(img_right_resized_gray_norm[np.newaxis,:,:,:])
    #             pred_right = pred_right.squeeze()
    #             pred_right = (pred_right > 0.5).astype(np.uint8)

    #             if self.eval_tool.get_calib_ellipse_centers2(pred_left):
    #                 left_x, left_y = self.eval_tool.get_calib_ellipse_centers2(pred_left)
    #                 roundness_left = self.eval_tool.get_calib_roundness2(pred_left)
    #                 left_x = round(left_x,3)
    #                 left_y = round(left_y,3)
    #                 roundness_left = round(roundness_left,3)
    #                 prev_left_x = left_x
    #                 prev_left_y = left_y
    #                 img_left_result = self.eval_tool.draw_pred_calib_ellipse2(img_left_resized,pred_left)
    #                 img_left_result = cv2.putText(img_left_result,f'Lt, x:{left_x},y:{left_y}',(10,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255))
    #             else:
    #                 roundness_left = 0
    #                 # left_x = prev_left_x
    #                 left_x = None
    #                 # left_y = prev_left_y
    #                 left_y = None
    #                 img_left_result = img_left_resized
    #                 # img_left_result = cv2.putText(img_left_result,f'Lt, x:{left_x},y:{left_y}',(10,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0))

    #             l_left_center_axis.append([left_x,left_y])
    #             l_left_roundness.append(roundness_left)
    #             l_left_result_imgs.append(img_left_result)

    #             if self.eval_tool.get_calib_ellipse_centers2(pred_right):
    #                 right_x, right_y = self.eval_tool.get_calib_ellipse_centers2(pred_right)
    #                 roundness_right = self.eval_tool.get_calib_roundness2(pred_right)
    #                 right_x = round(right_x,3)
    #                 right_y = round(right_y,3)
    #                 roundness_right = round(roundness_right,3)
    #                 prev_right_x = right_x
    #                 prev_right_y = right_y
    #                 img_right_result = self.eval_tool.draw_pred_calib_ellipse2(img_right_resized,pred_right)
    #                 img_right_result = cv2.putText(img_right_result,f'Rt, x:{right_x},y:{right_y}',(10,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255))
    #             else:
    #                 roundness_right = 0
    #                 # right_x = prev_right_x
    #                 right_x = None
    #                 # right_y = prev_right_y
    #                 right_y = None
    #                 img_right_result = img_right_resized
    #                 # img_right_result = cv2.putText(img_right_result,f'Rt, x:{right_x},y:{right_y}',(10,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0))
                
    #             l_right_center_axis.append([right_x, right_y])
    #             l_right_roundness.append(roundness_right)
    #             l_right_result_imgs.append(img_right_result)
                
    #             if index % 100 == 0:
    #                 current_time= time.time()
    #                 print('index: ', index, 'time: ', current_time-start_time)
    #             index += 1
            
    #     cap.release()
    #     print("model_inference finished, it tooks ", time.time()-start_time, "sec")
    #     return [l_left_center_axis,l_left_roundness,l_left_result_imgs, l_right_center_axis, l_right_roundness, l_right_result_imgs]


    def fill_na(self, list_:list, kinds:str='inner') -> list:
        ## the input list_ should have None value.
        if not isinstance(list_, list):
            raise Exception("the input should be a instance of list")

        list_ = copy.deepcopy(list_)
        
        start_index = 0
        end_index = 0
        FLAG_INNER_NONE = False
        none_count = 0
        none_start_index=0
        none_end_index=0
        current_index = 0

        if len(list_) < 2:
            return list_

        ## detect not none value index of each side
        for i in range(len(list_)):
            if list_[i] != None:
                start_index = i
                break
        for i in range(len(list_)-1,-1,-1):
            if list_[i] != None:
                end_index = i
                break
        # print(len(list_),start_index,end_index)

        if kinds == 'inner' or kinds == 'all':
            ## filling inner none values  (linear interpolation)
            for i in range(start_index, end_index+1):
                if list_[i] == None and FLAG_INNER_NONE == False:
                    FLAG_INNER_NONE = True
                    none_start_index = i
                    none_end_index = i
                    none_count = 1
                elif list_[i] == None and FLAG_INNER_NONE ==True:
                    none_count += 1
                    none_end_index = i
                elif list_[i] != None and FLAG_INNER_NONE == True:
                    FLAG_INNER_NONE = False
                    
                    ## start fill the None values
                    values = np.linspace(list_[none_start_index-1],list_[none_end_index+1],none_count+2)
                    list_[none_start_index:none_end_index+1] = values[1:-1]
        
        if kinds == 'tip' or kinds == 'all':
            ## filling None values of each tips (not none edge padding)
            # for i in range(start_index-1,-1,-1):
            #     list_[i] = list_[start_index]
            # for i in range(end_index+1,len(list_)):
            #     list_[i] = list_[end_index]
            list_[:start_index] = [list_[start_index]] * start_index
            list_[end_index+1:] = [list_[end_index]] * (len(list_)-(end_index+1))
        
        return list_


    def get_gradient_outlier_indices(self, data:np.ndarray, edge_indices:np.ndarray, info_dict:dict, iqr_multiplier_x10:float=1.5, partial:str='all', flag_edge:str=True) ->'list[np.ndarray, list[float,float]]':
        ## the parameter named partial can be either 'all' or 'positive'.
        ## 'positive' will make the grads to abs(grads)
        ## flag_edge means that it will be considered with only withing the edge_indices otherwise considered with all gradient of the data  ( grads = data[1:]-data[:-1] ).
        
        if not isinstance(data, np.ndarray):
            raise Exception("from get_outlier: the input must be a instance of np.ndarray")
        iqr_multiplier_x10 = iqr_multiplier_x10 / 10.

        grads = info_dict['grads']
        
        if flag_edge == False:
            grads = data[1:]-data[:-1]
            edge_indices = np.array(list(range(len(grads))))
            
        if partial != 'all':
            grads = abs(grads)
        
        q1, q3 = np.percentile(grads,[25,75])
        iqr = q3-q1
        upper_bound = q3 + (iqr * iqr_multiplier_x10)
        lower_bound = q1 - (iqr * iqr_multiplier_x10)

        bool_bound_condition = (grads > upper_bound) & (grads < lower_bound)
        
        return [edge_indices[bool_bound_condition], lower_bound, upper_bound]


    
    def mask_with_difference_outlier(self, list_:list, iqr_multiplier:float=1.5) -> 'list[list, np.ndarray]':
        ## the input list will contain several None values. so need to be consider None value.
        ## the input list_ should be filled with not none value on their tips. before call this func you must call the fill_na function with the kind of 'tip' 
        ## fill_na(list_, 'tip')
        
        if not isinstance(list_, list):
            raise Exception("the input should be a instance of list")
        list_ = copy.deepcopy(list_)
        start_index = 0
        end_index = len(list_) -1
        ## detect not none value index of each side
        for i in range(len(list_)):
            if list_[i] != None:
                start_index = i
                break
        for i in range(len(list_)-1,-1,-1):
            if list_[i] != None:
                end_index = i
                break
        if start_index != 0 or end_index != len(list_)-1:
            # raise Exception("from erase_outliser: input has none value on their tip. you must call fill_na(list_,'tip') before calling this function")
            list_ = self.fill_na(list_,'tip')

        if len(list_) < 3:
            return list_

        most_close_not_none_left_values = [0] * len(list_)
        most_close_not_none_left_values[0] = list_[0]

        # most_close_not_none_right_values = [0] * len(list_)
        # most_close_not_none_right_values[-1] = list_[-1]

        
        temp_left_value = list_[0]
        for i in range(1,len(list_)):
            if list_[i] == None:
                most_close_not_none_left_values[i] = temp_left_value
            else:
                most_close_not_none_left_values[i] = temp_left_value
                temp_left_value = list_[i]
        ## ex) list_                       = [5,None,None,10,None,25,17]
        ## most_close_not_none_left_values = [5,   5,   5, 5,  10,10,25]

        # temp_right_value = list_[-1]
        # for i in range(len(list_)-1,-1,-1):
        #     if list_[i] == None:
        #         most_close_not_none_right_values[i] = temp_right_value
        #     else:
        #         most_close_not_none_right_values[i] = temp_right_value
        #         temp_right_value = list_[i]

        diffs = [0] * len(list_)
        for i in range(1,len(list_)-1):
            if list_[i] == None:
                diffs[i] = 0
                continue
            else:  
                left_diff = list_[i] - most_close_not_none_left_values[i]
                # right_diff = list_[i] - most_close_not_none_right_values[i]
                # diffs[i] = max(abs(left_diff), abs(right_diff))
                diffs[i] = left_diff
                
        q1, q3 = np.percentile(diffs,[25, 75])
        iqr = q3 - q1

        lower_bound = q1 - (iqr * iqr_multiplier)
        upper_bound = q3 + (iqr * iqr_multiplier)

        outlier_indices = np.where((diffs < lower_bound) | (diffs >upper_bound))[0]
        for i in outlier_indices:
            list_[i] = None

        return [list_, outlier_indices]


    def mask_with_value_outlier(self, list_:list, iqr_multiplier:int=1.5) -> 'list[list,np.ndarray]':
        ## the input list will contain several None values. so need to be consider None value.
        ## input list contains None values and consists of positive values.
        ## the output list_ also contain None value when it becomes the outlier.
        
        ## it can cause some wrong result.
        ## this method detect outliers with on their values
        
        if not isinstance(list_, list):
            raise Exception("the input should be a instance of list")
        list_ = copy.deepcopy(list_)
        
        start_index = 0
        end_index = len(list_) -1
        ## detect not none value index of each side
        for i in range(len(list_)):
            if list_[i] != None:
                start_index = i
                break
        for i in range(len(list_)-1,-1,-1):
            if list_[i] != None:
                end_index = i
                break
        if start_index != 0 or end_index != len(list_)-1:
            # raise Exception("from erase_outliser: input has none value on their tip. you must call fill_na(list_,'tip') before calling this function")
            list_ = self.fill_na(list_,'tip')

        if len(list_) < 3:
            return list_

        not_none_list = []
        for i in list_:
            if i != None:
                not_none_list.append(i)
        
        not_none_npa = np.array(not_none_list)
        q1, q3 = np.percentile(not_none_npa,[25, 75])
        iqr = q3 - q1

        lower_bound = q1 - (iqr * iqr_multiplier)
        upper_bound = q3 + (iqr * iqr_multiplier)

        outlier_indices = []
        for idx, item in enumerate(list_):
            if item == None:
                continue
            if item < lower_bound or item > upper_bound:
                list_[idx] = None
                outlier_indices.append(idx)

        outlier_indices = np.array(outlier_indices)

        return [list_, outlier_indices]


    def mask_with_roundness(self, list_:list, roundnesses:list, r:float=0.6) -> 'list[list, np.ndarray]':
        ## the input lists will contain several None values. so need to be consider None value.
        ## the output list_ also contain None value when it matched with roundness_threshold.
        if not isinstance(list_,list) or not isinstance(roundnesses, list):
            raise Exception("the input should be a instance of list")
        if r < 0:
            r = 0
        elif r > 1:
            r = 1

        list_ = copy.deepcopy(list_)
        
        ## the roundness value is expected 0 when it comes not to be detected.
        ## it makes works more easily using by numpy 
        np_roundnesses = np.array(roundnesses)
        masked_indices = np.where(np_roundnesses <= r)[0]
        for i in masked_indices:
            list_[i] = None

        return list_ , masked_indices

    def spread_none_with_n_step(self,list_, n=2) -> list:
        if not isinstance(list_,list):
            raise Exception("the input should be a instance of list")
        if n < 1:
            return list_
        
        list_ = copy.deepcopy(list_)
        nones = []
        for i in range(0,len(list_)):
            if list_[i] == None:
                nones.append(i)

        for i in nones:
            left_index = max(0, i-n)
            right_index = min(len(list_)-1, i+n)
            n_inners = right_index - left_index + 1
            list_[max(0,i-n):min(len(list_)-1,i+n)+1] = [None] * n_inners 

        return list_

    def merge_none(self, list_1:list, list_2:list) -> 'list[list, np.ndarray]':
        ## the input lists will contain several None values. so need to be consider None value.
        ## the output list_ also contain None value.
        if not isinstance(list_1, list) or not isinstance(list_2, list):
            raise Exception("the input should be a instance of list")
        if len(list_1) != len(list_2):
            raise Exception("two inputs must have same length")

        list_1 = copy.deepcopy(list_1)
        list_2 = copy.deepcopy(list_2)
        
        for i, v in enumerate(list_2):
            if v == None:
                list_1[i] = None
        
        none_indices = []
        for idx in range(len(list_1)):
            if list_1[idx] == None:
                none_indices.append(idx) 

        none_indices = np.array(none_indices)
        return [list_1, none_indices]

    def filter_low_pass(self, data) -> np.ndarray:
        b1 = signal.firwin(21, cutoff=1, fs=300, pass_zero='lowpass')
        # b = signal.firwin(101, cutoff=5, fs=30, pass_zero='highpass')
        filtered1 = signal.lfilter(b1, [1.0], data)

        # b2, a2 = signal.butter(11, 0.2)
        # filtered2 = signal.filtfilt(b2, a2, data)
        # print(type(filtered2)) # numpy.ndarray
        
        return filtered1

    def get_curve_indices(self, data : Union[list, np.ndarray]) -> np.ndarray:
        if not isinstance(data, list) and not isinstance(data, np.ndarray):
            raise Exception("from get_curve_indices: input(data) should be list or np.array")
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            
        diffs = data[1:] - data[:-1]
        prev_diffs = np.insert(diffs,0,0)
        next_diffs = np.append(prev_diffs[1:], 0)
        diff_mul_with_next_diff = prev_diffs * next_diffs
        
        indices_curve = np.where(diff_mul_with_next_diff < 0)[0]
        
        return indices_curve

    # ## require fix
    # def get_bent_indices(self, data:Union[list,np.ndarray]) -> np.ndarray:
    
    #     if not isinstance(data, list) and not isinstance(data, np.ndarray):
    #         raise Exception("from get_bent_indices: input(data) should be list or np.array")
    #     if not isinstance(data, np.ndarray):
    #         data = np.array(data)
        
    #     temp_prev_index = data[0]
    #     connected_index_head = None
    #     FLAG_CONNECT_ON = False
    #     result = []
    #     for index in data[1:]:
    #         if temp_prev_index + 1 == index:
    #             if FLAG_CONNECT_ON == False:
    #                 FLAG_CONNECT_ON = True
    #                 connected_index_head = temp_prev_index
    #         else:
    #             if FLAG_CONNECT_ON == True:
    #                 connected_index_tail = temp_prev_index
    #                 result.append([connected_index_head, connected_index_tail])
    #                 FLAG_CONNECT_ON = False
    #             else:
    #                 result.append([temp_prev_index])

    #         temp_prev_index = index

    #     return result


    def get_grouped_sequence(self, data : Union[list, np.ndarray]) -> 'list[list[int]]':
        # make sequence grouped. ex) data = [1,2,3,6,7,9,12,15,16,17], result = [[1,3],[6,7],[9],[12],[15,17]]
        if not isinstance(data, list) and not isinstance(data, np.ndarray):
            raise Exception("from get_inner_range: input(data) should be list or np.array")
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if len(data) == 1:
            return data
        
        last_index = len(data)-1
        results = []
        temp_list = []
        FLAG_CONNECT_ON = False
        for index in range(last_index):
            current_value = data[index]
            next_value = data[index+1]
            if next_value == current_value+1:
                if FLAG_CONNECT_ON == False:
                    FLAG_CONNECT_ON = True
                    temp_list = [current_value, next_value]
                else:
                    temp_list = [temp_list[0], next_value]
            else:
                if FLAG_CONNECT_ON == True:
                    FLAG_CONNECT_ON = False
                    results.append(temp_list)
                else:
                    results.append([current_value])
            if index == last_index-1:
                results.append(temp_list)
        
        return results


    # def connect_curves(self, data:np.ndarray, curve_indices:np.ndarray, start_curve_number:int, end_curve_number:int) -> 'list[np.ndarray, List[int]]':
    #     if start_curve_number >= end_curve_number:
    #         raise Exception(f"from connect_curves: end_index({end_curve_number}) is over than start_index({start_curve_number})")
    #     ## start curve index and end curve index have several inner curve indices which have short distance to be deleted.
    #     ## this method condiders that if inner curve is over outer curves limit or not
    #     result = data.copy()

    #     start_value = data[curve_indices[start_curve_number]]
    #     end_value = data[curve_indices[end_curve_number]]

    #     if start_value >= end_value:
    #         top = start_value
    #         bottom = end_value
    #     else:
    #         top = end_value
    #         bottom = start_value

    #     remain_curves_indices = [curve_indices[start_curve_number]]
    #     remain_index_list = [start_curve_number]
    #     if end_curve_number > start_curve_number + 1:
    #         # check inner curves value
    #         if max(data[curve_indices[start_curve_number+1 : end_curve_number]]) >= top:
    #             temp_argmax = np.argmax(data[curve_indices[start_curve_number+1 : end_curve_number]])
    #             max_curve_index = start_curve_number+1 + temp_argmax
    #             remain_curves_indices.append(curve_indices[max_curve_index])
    #             remain_index_list.append(max_curve_index)
                        
    #         if min(data[curve_indices[start_curve_number+1 : end_curve_number]]) <= bottom:
    #             temp_argmin = np.argmin(data[curve_indices[start_curve_number+1 : end_curve_number]])
    #             min_curve_index = start_curve_number+1 + temp_argmin
    #             remain_curves_indices.append(curve_indices[min_curve_index])
    #             remain_index_list.append(min_curve_index)
                    
    #     remain_curves_indices.append(curve_indices[end_curve_number])
    #     remain_curves_indices = sorted(remain_curves_indices)
    #     remain_index_list = sorted(remain_index_list)

    #     for i in range(len(remain_curves_indices)-1):
    #         temp = np.linspace(data[remain_curves_indices[i]],data[remain_curves_indices[i+1]],remain_curves_indices[i+1]-remain_curves_indices[i]+1)
    #         result[remain_curves_indices[i]:remain_curves_indices[i+1]+1] = temp
        
    #     return [result, remain_index_list]


    def erase_inside_single_curve(self,data:np.ndarray, curve_indices:np.ndarray, curve_infos:'list[list[int, int, float, float]]') -> 'list[list[int],list[int]]':
        ## this function will modify the data by inplace not copy.
        
        remain_curve_indices = []
        erased_curve_indices = []
        min_curve_number = 0
        max_curve_number = len(curve_indices) - 1
        
        ## the difference (end_curve_number - start_curve_number) must be 1
        for start_curve_number, end_curve_number, left_outer_distance, right_outer_distance in curve_infos:
            # if start_curve_number >= end_curve_number:
                # raise Exception(f"from connect_curves: end_index({end_curve_number}) is over than start_index({start_curve_number})")
            ## start curve index and end curve index have several inner curve indices which have short distance to be deleted.
            ## this method condiders that if inner curve is over outer curves limit or not

            if left_outer_distance >= right_outer_distance:
                right_outer_curve_number = min(end_curve_number + 1, max_curve_number)
                
                start_index = curve_indices[start_curve_number] 
                start_value = data[start_index]
                end_index = curve_indices[right_outer_curve_number]
                end_value = data[end_index]
                
                data[start_index: end_index+1] = np.linspace(start_value, end_value, end_index-start_index+1)
                remain_curve_indices.append(curve_indices[start_curve_number])
                erased_curve_indices.append(curve_indices[end_curve_number])
            else: 
                left_outer_curve_number = max(start_curve_number - 1, min_curve_number)

                start_index = curve_indices[left_outer_curve_number]
                start_value = data[start_index]
                end_index = curve_indices[end_curve_number]
                end_value = data[end_index]

                data[start_index: end_index+1] = np.linspace(start_value, end_value, end_index-start_index+1)
                remain_curve_indices.append(curve_indices[end_curve_number])
                erased_curve_indices.append(curve_indices[start_curve_number])
            
            
        return [erased_curve_indices, remain_curve_indices]


    def erase_curve_sequnce_directly(self,data:np.ndarray, curve_indices:np.ndarray, curve_infos:'list[list[int, int, float, float]]') -> 'list[list[int],list[int]]':
        ## this function will modify the data by inplace not copy.
        
        remain_curve_indices = []
        erased_curve_indices = []
        min_curve_number = 0
        max_curve_number = len(curve_indices) - 1
        
        ## the difference (end_curve_number - start_curve_number) must be over 2
        for start_curve_number, end_curve_number, left_outer_distance, right_outer_distance in curve_infos:
            # if start_curve_number >= end_curve_number:
                # raise Exception(f"from connect_curves: end_index({end_curve_number}) is over than start_index({start_curve_number})")
            ## start curve index and end curve index have several inner curve indices which have short distance to be deleted.
            ## this method condiders that if inner curve is over outer curves limit or not

            start_index = curve_indices[start_curve_number]
            end_index = curve_indices[end_curve_number]
            start_value = data[start_index]
            end_value = data[end_index]
            
            data[start_index: end_index+1] = np.linspace(start_value, end_value, end_index-start_index+1)
            
            remain_curve_indices += [start_index, end_index]
            erased_curve_indices += curve_indices[start_curve_number+1:end_curve_number].tolist()
        
        return [erased_curve_indices, remain_curve_indices]



    def erase_curve_sequence_with_non_directly(self, data:np.ndarray, curve_indices:np.ndarray, curve_infos:'list[list[int, int, float, float]]', flag_inner_outer:str='inner') -> 'list[list[int],list[int]]':
        ## this function will modify the data by inplace not copy.
        total_remain_curve_indices = []
        total_erased_curve_indices = []
        
        min_curve_number = 0
        max_curve_number = len(curve_indices) - 1
        
        ## the difference (end_curve_number - start_curve_number) must be over 2
        for start_curve_number, end_curve_number, left_outer_distance, right_outer_distance in curve_infos:
            # if start_curve_number >= end_curve_number:
                # raise Exception(f"from connect_curves: end_index({end_curve_number}) is over than start_index({start_curve_number})")
            
            ## start curve index and end curve index have several inner curve indices which have short distance to be deleted.
            ## this method condiders that if inner curve is over outer curves limit or not
            
            if flag_inner_outer == 'outer':
                start_curve_number = max(start_curve_number-1, min_curve_number)
                end_curve_number = min(end_curve_number+1, max_curve_number)
                
            start_index = curve_indices[start_curve_number]
            end_index = curve_indices[end_curve_number]
            
            start_value = data[start_index]
            end_value = data[end_index]
            
            # left_inner_index = curve_indices[start_curve_number+1]
            # right_inner_index = curve_indices[end_curve_number-1]
            
            if start_value >= end_value:
                top = start_value
                bottom = end_value
            else:
                top = end_value
                bottom = start_value
                
            remain_curve_indices = [curve_indices[start_curve_number]]     ## fill first curve to prepare calibration
            inner_curves_indices = curve_indices[start_curve_number+1:end_curve_number]  
            total_erased_curve_indices += inner_curves_indices.tolist()
            
            values = data[inner_curves_indices]
            if max(values) >= top:                             ## to find most upper curve over the top value in the inner_curves
                temp_argmax = np.argmax(values)
                curve_max_index = curve_indices[start_curve_number+1 + temp_argmax]
                remain_curve_indices.append(curve_max_index)
                total_remain_curve_indices.append(curve_max_index)
                # print('total:' ,total_erased_curve_indices, 'max_index:',curve_max_index, 'max_index_dtype, type:',curve_max_index.dtype, type(curve_max_index))
                total_erased_curve_indices.remove(curve_max_index)
                # print('total_removed:' ,total_erased_curve_indices, 'max_index:',curve_max_index, 'max_index_dtype, type:',curve_max_index.dtype, type(curve_max_index))
                        
            if min(values) <= bottom:                          ## to find most below curve over the bottom value in the inner_curves
                temp_argmin = np.argmin(values)
                min_curve_index = start_curve_number+1 + temp_argmin
                remain_curve_indices.append(curve_indices[min_curve_index])
                total_remain_curve_indices.append(curve_indices[min_curve_index])
                total_erased_curve_indices.remove(curve_indices[min_curve_index])
      
            remain_curve_indices.append(curve_indices[end_curve_number])    ## fill last curve to prepare calibration
            remain_curve_indices = sorted(remain_curve_indices)
            
            for i in range(len(remain_curve_indices)-1):                    ## calibration with remain_curves
                start_idx = remain_curve_indices[i]
                end_idx = remain_curve_indices[i+1]
                temp = np.linspace(data[start_idx],data[end_idx], end_idx-start_idx+1)
                data[start_idx:end_idx+1] = temp
            
        return [total_erased_curve_indices, total_remain_curve_indices]


    ## below function not successful. needed more implementation
    # def connect_curves_with_new_point(self, data:np.ndarray, curve_indices:np.ndarray, start_end_list:'list[int, int]')->'list[np.ndarray, list[int]]':
    #     ## start curve index and end curve index have several inner curve indices which have short distance to be deleted.
    #     ## this method will make new curve point as well as save outer curve's gradients
    #     # result = data.copy()
    #     result=data

    #     new_point_index = []
    #     for start_curve_number, end_curve_number in start_end_list:
    #         if start_curve_number >= end_curve_number:
    #             raise Exception(f"from connect_curves_with_new_point: end_index({end_curve_number}) is over than start_index({start_curve_number})")
            
    #         x1 = curve_indices[start_curve_number]
    #         y1 = data[x1]
    #         x2 = curve_indices[start_curve_number+1]
    #         y2 = data[x2]
    #         x3 = curve_indices[end_curve_number-1]
    #         y3 = data[x3]
    #         x4 = curve_indices[end_curve_number]
    #         y4 = data[x4]

    #         intersect_point = self.get_intersect_point([x1,y1],[x2,y2],[x3,y3],[x4,y4])
    #         if intersect_point == None:
    #             continue
            
    #         new_x, new_y = intersect_point
    #         new_x = int(new_x)
    #         new_point_index.append(new_x)

    #         temp1 = np.linspace(y1,new_y, new_x-x1+1)
    #         temp2 = np.linspace(new_y,y4, x4-new_x+1)
    #         result[x1:new_x+1] = temp1
    #         result[new_x:x4+1] = temp2
        
    #     return [result, new_point_index]



    def filter_modify_base_curve(self, data:Union[np.ndarray,list], curve_indices:np.ndarray=None, value_distance_thres:float=2.0, frames_thres:int=5) -> 'list[np.ndarray, np.ndarray, np.ndarray]':
        # thres is the threshold distance between two of connected curve
        # frame_distance_thres is the threshold for the horizontal wave to make flat

        if not isinstance(data, list) and not isinstance(data, np.ndarray):
            raise Exception("from get_inner_range: input(data) should be list or np.array")
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        result_data = data.copy()
        # it is expected that curve_indices should be sorted
        if not isinstance(curve_indices, np.ndarray):
            curve_indices = self.get_curve_indices(data)

        values = data[curve_indices]
        diffs = values[1:] - values[:-1]
        frame_diffs = curve_indices[1:] - curve_indices[:-1]
        
        norm1_distances = abs(diffs) + frame_diffs
        raise NotImplementedError()
        consider_needed = np.where(abs(diffs[:-1]) < value_distance_thres)[0]   ## diffs[:-1] ==> because of the below process (consider_needed_grouped_2[idx] = [first_value, last_value+1]) last_value+1 means last_index+1 so it has over limit the array's index
        # make sequence grouped. ex) [1,2,3,6,7,9,12,15,16,17] ==> [[1,3],[6,7],[9],[12],[15,17]]
        consider_needed_grouped = self.get_grouped_sequence(consider_needed)
        
        consider_needed_grouped_2 = consider_needed_grouped[:]
        for idx, item in enumerate(consider_needed_grouped):
            last_value = item[-1]
            first_value = item[0]
            if len(item) == 1:
                consider_needed_grouped_2[idx] = [last_value, last_value+1]
            else:
                consider_needed_grouped_2[idx] = [first_value, last_value+1]
        
        ## erase_needed_infos == [[start_index_of_curve_indices, end_index_of_curve_indices, left_outer_distance, right_outer_distance],...]
        consider_needed_infos = []
        for consider_needed_start, consider_needed_end in consider_needed_grouped_2:
            if consider_needed_start == 0:
                consider_needed_infos.append([consider_needed_start, consider_needed_end, 0, norm1_distances[consider_needed_end]])
            else:
                consider_needed_infos.append([consider_needed_start, consider_needed_end, norm1_distances[consider_needed_start-1], norm1_distances[consider_needed_end]])
        
   
        erased_curve_index_list = []
        # remain_curve_index_list = []
        ## required_consider_edge_curve is needed when have second filtering with curve2linear3
        required_consider_edge_curve_list = []
        for consider_infos in consider_needed_infos:
            consider_start, consider_end, left_outer_distance, right_outer_distance = consider_infos
            ## not good result when it comes to outside of the tip, just skip
            if consider_start <= 0:
                continue
            if consider_end >= len(curve_indices)-1:
                continue

            ## if single curve, call the self.erase_inside_single_curve().
            if (consider_end - consider_start) == 1:
                temp_erased_curve_list, temp_remain_curve_list = self.erase_inside_single_curve(result_data, curve_indices, [[consider_start, consider_end, left_outer_distance, right_outer_distance]])
                
                erased_curve_index_list += temp_erased_curve_list
                # remain_curve_index_list += temp_remain_curve_list
                
            else:
                ## if not single curve. it requires to consider frame_distance through the curve sequence. 
                ## check if the frame_distance is over thres
                start_index = curve_indices[consider_start]
                end_index = curve_indices[consider_end]
                frame_diff = (end_index - start_index)

                
                if frame_diff < frames_thres:
                    ## the frame gap is over the thres. so each edge of curve sequnce will be remain
                    
                    temp_erased_curve_list, temp_remain_curve_list = self.erase_curve_sequence_with_non_directly(result_data, curve_indices, [[consider_start, consider_end, left_outer_distance, right_outer_distance]],flag_inner_outer='inner')
                    erased_curve_index_list += temp_erased_curve_list
                    # remain_curve_index_list += temp_remain_curve_list
                    
                else: ## continueous curves are less than thres
                    # required_consider_edge_curve_list += [start_index, end_index]
                    temp_erased_curve_list, temp_remain_curve_list = self.erase_curve_sequence_with_non_directly(result_data, curve_indices, [[consider_start, consider_end, left_outer_distance, right_outer_distance]],flag_inner_outer='outer')
                    erased_curve_index_list += temp_erased_curve_list
                    # remain_curve_index_list += temp_remain_curve_list


        erased_curve_indices = np.array(erased_curve_index_list)
        # required_consider_edge_curve_list = np.array(required_consider_edge_curve_list)
        # return [result, curve_indices[erased_curve_index_list], curve_indices[required_consider_edge_curve_list]]
        
        remain_curve_indices = [i for i in curve_indices if i not in erased_curve_index_list]
        return [result_data, erased_curve_indices, np.array(remain_curve_indices)]
        

    # ## require fix
    # def filter_modify_base_curve2(self, data:Union[list,np.ndarray], thres:int=2, frames_thres:int=15) -> 'list[np.ndarray, np.ndarray, np.ndarray]':
    #     # input(data) should be derived from low_pass_filter
    #     # thres is the threshold distance between two of connected curve
    #     # frame_distance_thres is the threshold for the horizontal wave to make flat

    #     if not isinstance(data, list) and not isinstance(data, np.ndarray):
    #         raise Exception("from get_inner_range: input(data) should be list or np.array")
    #     if not isinstance(data, np.ndarray):
    #         data = np.array(data)

    #     result = data.copy()
    #     # it is expected that curve_indices should be sorted
    #     curve_indices = self.get_curve_indices(data)

        
    #     values = data[curve_indices]
    #     distances = abs(values[1:] - values[:-1])
    #     erase_distance_indices = np.where(distances < thres)[0]

    #     erase_distance_grouped_indices = self.get_grouped_sequence(erase_distance_indices)
    #     for i in erase_distance_grouped_indices:
    #         last_value = i[-1]
    #         first_value = i[0]
    #         if len(i) == 1:
    #             i.remove(last_value)
    #             i.append(first_value-1)
    #             i.append(last_value+2)
    #         else:
    #             i.remove(first_value)
    #             i.append(first_value-1)
    #             i.remove(last_value)
    #             i.append(last_value+2)


    #     erased_curve_indices = []

    #     connect_new_point_list = []
    #     connect_direct_list = []
    #     erase_curve_sequence_with_non_directly_list = []
    #     for grouped_index in erase_distance_grouped_indices:
    #         erase_distance_index_start, erase_distance_index_end = grouped_index
    #         if erase_distance_index_start < 0:
    #             erase_distance_index_start = 0
    #         if erase_distance_index_end > len(curve_indices)-1:
    #             erase_distance_index_end = len(curve_indices)-1
            
            
    #         ## check if the frame_distance is over thres
    #         frames = (curve_indices[erase_distance_index_end-1] - curve_indices[erase_distance_index_start+1])
    #         if frames > frames_thres:
    #             erase_curve_sequence_with_non_directly_list.append([erase_distance_index_start+1,erase_distance_index_end-1])
    #             for i in range(erase_distance_index_start+2,erase_distance_index_end-1):
    #                 erased_curve_indices.append(i)
    #         else:
    #             top = data[curve_indices[erase_distance_index_start]]
    #             bottom = data[curve_indices[erase_distance_index_end]]

    #             FLAG_ALL_INNER_RANGE = True
    #             for i in range(erase_distance_index_start+1,erase_distance_index_end):
    #                 if top < data[curve_indices[i]] or bottom > data[curve_indices[i]]:
    #                     connect_new_point_list.append([erase_distance_index_start, erase_distance_index_end])
    #                     FLAG_ALL_INNER_RANGE = False
    #                     break
    #             if FLAG_ALL_INNER_RANGE == False:
    #                 for i in range(erase_distance_index_start+1,erase_distance_index_end):
    #                     erased_curve_indices.append(i)
    #             else:
    #                 connect_direct_list.append([erase_distance_index_start, erase_distance_index_end])
    #                 for i in range(erase_distance_index_start+1,erase_distance_index_end):
    #                     erased_curve_indices.append(i)

    #         ## modify the curves
    #         self.erase_curve_sequence_with_non_directly(result, curve_indices, erase_curve_sequence_with_non_directly_list)
    #         self.connect_curves_with_new_point(result, curve_indices, connect_new_point_list)
    #         self.connect_curves_direct(result, curve_indices, connect_direct_list)

    #     erased_curve_indices = np.array(erased_curve_indices)
        
    #     return [result, erased_curve_indices]
    

    # ## require fix
    # def filter_curve2linear3(self, data:Union[np.ndarray,list], required_consider_curve_indices:np.ndarray, value_distance_thres:float=1.0, frame_distance_thres:int=5) -> 'list[np.ndarray, np.ndarray]':
    #     ## this filter do easy mechanism to erase curve within value_distance_thres
    #     ## this function should come after using filter_modify_base_curve since this function will use consider edge curve indices
    #     if not isinstance(data, list) and not isinstance(data, np.ndarray):
    #         raise Exception("from get_inner_range: input(data) should be list or np.array")
    #     if not isinstance(data, np.ndarray):
    #         data = np.array(data)

    #     new_datas = data.copy()
    #     edge_indices = required_consider_curve_indices.copy()
    #     erased_curves_indices = []
        
    #     flag_no_erased_curve = True
    #     while True:
    #         if flag_no_erased_curve == False:
    #             break
            
    #         values = data[edge_indices]
    #         value_distances = abs(values[1:]-values[:-1])
    #         value_distances = np.insert(value_distances, 0, 0)
    #         frame_distances = abs(edge_indices[1:]-edge_indices[:-1])
    #         frame_distances = np.insert(frame_distances, 0, 0)
    #         diffs = value_distances / (frame_distances + 1e-7)
    #         flag_no_erased_curve = False    
            
    #         for index in range(2,len(value_distances)-1):
    #             if value_distances[index] < value_distance_thres:
    #                 if frame_distances[index] < frame_distance_thres:
    #                     flag_no_erased_curve = True
                        
    #                     prev_diff = diffs[index-1]
    #                     curr_diff = diffs[index]
    #                     next_diff = diffs[index+1]

    #                     if next_diff < prev_diff:
    #                         start_index = index
    #                     else:
    #                         start_index = index -1

    #                     erased_curves_indices.append(edge_indices[start_index])
    #                     start_curve = edge_indices[start_index-1]
    #                     end_curve = edge_indices[start_index+1]
    #                     temp = np.linspace(new_datas[start_curve], new_datas[end_curve], end_curve-start_curve+1)
    #                     new_datas[start_curve:end_curve+1] = temp
    #                     edge_indices = np.append(edge_indices[:start_index],edge_indices[start_index+1:])
    #                     break
        
    #     remain_required_consider_curves_indices = [i for i in required_consider_curve_indices if i not in erased_curves_indices]

    #     return [new_datas, np.array(erased_curves_indices), np.array(remain_required_consider_curves_indices)]


    def get_intersect_point(self,a: 'list[float,float]',b:'list[float,float]',c:'list[float,float]',d:'list[float,float]') -> 'list[float, float]':
        ## https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
        ## Gabriel Eng : Using formula from: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

        ## two points(a,b) consist of line1
        ## two points(c,d) consist of line2
        ## find the intersect point of two lines
        
        t = ((a[0] - c[0]) * (c[1] - d[1]) - (a[1] - c[1]) * (c[0] - d[0])) / (((a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0])) + 1e-6)
        u = ((a[0] - c[0]) * (a[1] - b[1]) - (a[1] - c[1]) * (a[0] - b[0])) / (((a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0])) + 1e-6)

        # check if line actually intersect
        if (0 <= t and t <= 1 and 0 <= u and u <= 1):
            return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
        else: 
            return None


    def get_inner_values(self, data : Union[list, np.ndarray], start : int=0, end=None) -> np.ndarray:
        ## ex) data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        ##    start = 24
        ##      end = 78
        ##   return = [30,40,50,60,70]
        
        if not isinstance(data, list) and not isinstance(data, np.ndarray):
            raise Exception("from get_inner_range: input(data) should be list or np.array")
        if len(data)==0:
            return np.array([])
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if end == None:
            end = len(data)
        if end < start:
            raise Exception("from get_inner_range: the arg(end) is lower than the arg(start)")
        return data[(data >= start) & (data <= end)]


    def get_gradients_infos(self, data:np.ndarray, edge_indices:np.ndarray) -> 'dict[np.ndarray]':
        ## return infos about gradient on curve point and gradient ratio between a curve and its next one  

        y_diffs = data[edge_indices[1:]] - data[edge_indices[:-1]]

        # diffs2 = data[edge_indices[2:]] - data[edge_indices[1:-1]]
        x_diffs = edge_indices[1:]-edge_indices[:-1]

        # distances1 = np.sqrt(np.square(y_diffs) + np.square(x_diffs))
        distances1 = np.linalg.norm([y_diffs,x_diffs],axis=0)
        distances1 = np.append(distances1,0)
        distances2 = distances1[1:]
        distances2 = np.append(distances2, 0)

        grads = (y_diffs) / (edge_indices[1:] - edge_indices[:-1]).astype(np.float32)
        grads = np.append(grads,0)
        y_diffs = np.append(y_diffs,0)
        x_diffs = np.append(x_diffs,0)
        abs_grads = np.abs(grads)
        grad_ratios = abs_grads / (np.append(np.square(grads[1:]),0) + 1e-6)

        return {'distances1':distances1, 'distances2':distances2, 'x_diffs':x_diffs, 'y_diffs':y_diffs, 'grads':grads, 'abs_grads':abs_grads, 'grad_ratios':grad_ratios}


    def get_manual_nystagmus_indices(self, data:np.ndarray, edge_indices:np.ndarray, info_dict:dict = None)->np.ndarray:
        ## data must have no None value, only numeric.
        
        # y_diff_min_limit1 = 0.4
        # y_diff_min_limit2 = 0.05
        grad_ratio_min_limit = 0.7
        grad_ratio_max_limit = np.inf
        distance_min_limit = 0.4
        next_distance_min_limit = 0.05

        if not isinstance(info_dict,dict):
            info_dict = self.get_gradients_infos(data,edge_indices)

        distances = info_dict['distances1']
        next_distance = info_dict['distances2']
        y_diffs = info_dict['y_diffs']
        grads = info_dict['grads']
        abs_grads = info_dict['abs_grads']
        grad_ratios = info_dict['grad_ratios']

        bool_distance_condition = (distances >= distance_min_limit)
        bool_next_distance_condition = (next_distance >= next_distance_min_limit)
        
        ## grad_ratio limit
        bool_grad_ratio_condition3 = (grad_ratios >= grad_ratio_min_limit) & (grad_ratios <= grad_ratio_max_limit)

        ## prev gradient must bigger than next one.
        bool_gradient_compare_condition = abs_grads[:-1] > abs_grads[1:]
        bool_gradient_compare_condition = np.append(bool_gradient_compare_condition, False)

        bool_manual_nystagmus_condition = bool_distance_condition & bool_next_distance_condition & bool_grad_ratio_condition3 & bool_gradient_compare_condition

        return edge_indices[bool_manual_nystagmus_condition]

    
    def get_most_left_close_curve_indices(self, data:np.ndarray, edge_indices:np.ndarray, candidate_indices:np.ndarray) -> np.ndarray:
        ## to erase duplicate nearby index.
        ## return left closest curve indices.
        ##  ex) edge_indices      =       [0,5,10,15,20,25,30]
        ##      candidate_indices =       [4,5,11,14,16,23,27,29,30]
        ##      result            =       [0,5,10,10,15,20,25,29,30]        
        
        result = []
        for idx in candidate_indices:
            left_indices = edge_indices[edge_indices<=idx]
            if len(left_indices) == 0:
                continue
            closest_max = np.max(left_indices)
            result.append(closest_max)
        
        result = np.unique(result)
        # result.sort()  ## np.unique will sort the array.
        return result


    def get_statistical_nystagmus_indices(self, data:np.ndarray, edge_indices:np.ndarray, info_dict:dict, additional_indices:np.ndarray=np.array([]))->np.ndarray:
        ## make additional indieces with outliers of the gradient of the data
        
        outlier_indices1, lower_bound1, upper_bound1 = self.get_gradient_outlier_indices(data, edge_indices, info_dict, iqr_multiplier_x10=15, partial='all', flag_edge=False)
        outlier_indices2, lower_bound2, upper_bound2 = self.get_gradient_outlier_indices(data, edge_indices, info_dict, iqr_multiplier_x10=15, partial='positive', flag_edge=False)
        outlier_indices3, lower_bound3, upper_bound3 = self.get_gradient_outlier_indices(data, edge_indices, info_dict, iqr_multiplier_x10=15, partial='all', flag_edge=True)
        outlier_indices4, lower_bound4, upper_bound4 = self.get_gradient_outlier_indices(data, edge_indices, info_dict, iqr_multiplier_x10=15, partial='positive', flag_edge=True)
        total_outlier_indices = np.concatenate([outlier_indices1, outlier_indices2, outlier_indices3, outlier_indices4, additional_indices])
        total_outlier_indices = np.unique(total_outlier_indices)
        closest_curve_indices = self.get_most_left_close_curve_indices(data, edge_indices, total_outlier_indices)
        
        return closest_curve_indices





if __name__ == '__main__':
    fe = FeatureExtractor()
    a = fe.get_grouped_sequence([0,5,7,8,9,10,15,17,18,22,24,25,26])
    print(a)
    
    # np.linalg.norm()