import weakref
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5.QtGui import QImage
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, QMutex
from PyQt5 import QtCore, QtGui
from my_feature_extractor import FeatureExtractor

## kwoncy_clone (tensorflow 2.6 installed with conda)
# from keras.models import Model, load_model
# from tensorflow.keras.models import Model, load_model
## kwoncy-only-pip (tensorflow 2.10 installed with pip)
from keras.models import Model, load_model
import cv2, os, time, sys, re, pickle
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from draw_ellipse import *
from ellipses import *
from bwperim_2 import *
from my_eval_tool import Eval_tool
from ellipses import LSqEllipse #The code is pulled frm https://github.com/bdhammel/least-squares-ellipse-fitting

import roi_tool_subview1

np.set_printoptions(threshold=sys.maxsize)

# MODEL_NAME = "new_pupil3_320x240_15376_E15_B5_R3333_S9889.h5"
MODEL_NAME = "new2_pupil3_320x240_6912_E30_B5_R4444_S9709.h5"

MODEL_HEIGHT = 240
MODEL_WIDTH = 320



def dice_score(img1, img2):

    # img1_f = K.flatten(img1)
    img1_f = img1.reshape(1,-1)
    # img2_f = K.flatten(img2)
    img2_f = img2.reshape(1,-1)
    intersection = np.sum(img1_f * img2_f)
    # return (2. * intersection + np.epsilon()) / (K.sum(img1_f) + K.sum(img2_f) + K.epsilon())
    return (2. * intersection + np.finfo(float).eps) / (np.sum(img1_f) + np.sum(img2_f) + np.finfo(float).eps)


class Worker(QObject):
    def __init__(self,start_index,max_index,d_flags):
        super().__init__()
        self.start_index = start_index
        self.max_index = max_index
        self.d_flags = d_flags
        self.running = False

    progress = pyqtSignal(int)
    finished = pyqtSignal()
    stopper = pyqtSignal(bool)
    waitted = pyqtSignal()

    def run(self):

        if not (self.start_index > self.max_index):
            while self.running:
                if self.start_index < self.max_index:
                    self.start_index += 1
                    self.progress.emit(self.start_index)
                    print(self.start_index)
                    time.sleep(0.2)
                else:
                    break
            self.finished.emit()   
        else:
            self.finished.emit()


    def set_running(self, state):
        self.running = state
    
class MainCtl():
    def __init__(self,view):
        self.view = view
        self.MODEL_HEIGHT = 240
        self.MODEL_WIDTH = 320
        self.MODEL_NAME = MODEL_NAME
        self.model1 = load_model(self.MODEL_NAME, custom_objects={'dice_score': dice_score})
        self.model2 = 0
        self.myfe = FeatureExtractor()
        self.eval_tool = Eval_tool()
        self.l_frames = []
        self.l_frames_gray = []
        # maskimgs for open folder 
        self.l_mask_imgs = []
        self.n_frames = 0
        self.worker = None

        self.current_location = 'Lt'
        self.current_index = None
        self.l_save_indices = None
        self.current_npimg = None
        self.current_maskimg = None
        
        self.resize_palate = None
        self.resize_maskimg = None

        self.ellipse_info = None

        self.max_frame_index = 0

        self.CURRENT_FILE_FULL_PATH = None
        self.FILE_PATH = None
        self.FILE_NAME = None
        self.FILE_EXT = None
        self.FILE_NAME_WITH_EXT = None
        # self.SAVE_BASE_PATH = "C:/kwoncy/eye/temp_xml/pair/new_test_sets"
        self.SAVE_BASE_PATH = "."
        self.SAVE_PATH = None
        self.LOAD_PATH = None

        ### for drawing the ellipse
        self.X = -1
        self.Y = -1
        self.W = -1
        self.H = -1
        self.IMG = None
        self.MASK = None

        self.flag_video_on = None
        self.flag_open_folder = None
        #### for play btn
        self.thread = QThread()
        self.flags = {"play frmaes" : None}
        self.stopper = Worker(0,0,self.flags)
        
        ### for drawing
        self.flag_lmouse_down = None
        self.flag_rmouse_down = None
        self.flag_kernel_size = 1

        ### subgui for extract index
        self.subview = None
        self.subviews = []
        self.l_resized_frames_lt = []
        self.l_resized_frames_rt = []
        ## inferred_info contains 'left_centers', 'left_roundnesses', 'left_widths', 'left_heights', 'left_radians'
        ## 'right_centers', 'right_roundnesses', 'right_widths', 'right_heights', 'right_radians'
        self.d_inferred_info = None
        self.connect_view()

    def connect_view(self):
        self.info1_btn_open_video = self.view.info1.btn_open_video
        self.info1_btn_open_folder = self.view.info1.btn_open_folder
        self.info1_btn_set_lt = self.view.info1.btn_set_lt
        self.info1_btn_set_rt = self.view.info1.btn_set_rt
        self.info1_btn_set_lt_rt = self.view.info1.btn_set_lt_rt
        self.info1_lbl_current_location = self.view.info1.lbl_current_location
        self.info1_btn_extract_index = self.view.info1.btn_extract_index

        self.info2_lbl_main_canvas1 = self.view.info2.lbl_main_canvas1
        self.info2_lbl_current_index = self.view.info2.lbl_current_index
        self.info2_ledit_goto_index = self.view.info2.ledit_goto_index
        self.info2_btn_goto_index = self.view.info2.btn_goto_index
        self.info2_btn_prev_index = self.view.info2.btn_prev_index
        self.info2_btn_next_index = self.view.info2.btn_next_index
        self.info2_btn_play_frames = self.view.info2.btn_play_frames
        self.info2_btn_stop_frames = self.view.info2.btn_stop_frames
        self.info2_btn_set_orig = self.view.info2.btn_set_orig
        self.info2_btn_set_rois = self.view.info2.btn_set_rois
        self.info2_btn_set_ellipse = self.view.info2.btn_set_ellipse
        self.info2_btn_set_calib_ellipse1 = self.view.info2.btn_set_calib_ellipse1
        self.info2_btn_set_calib_ellipse2 = self.view.info2.btn_set_calib_ellipse2
        self.info2_btn_get_mask1 = self.view.info2.btn_get_mask1
        self.info2_btn_get_mask2 = self.view.info2.btn_get_mask2
        self.info2_btn_resize_2x = self.view.info2.btn_resize_2x
        self.info2_btn_resize_4x = self.view.info2.btn_resize_4x
        self.info2_lbl_ellipse_info = self.view.info2.lbl_ellipse_info
        self.info2_ledit_ellipse_info_center_x = self.view.info2.ledit_ellipse_info_center_x
        self.info2_ledit_ellipse_info_center_y = self.view.info2.ledit_ellipse_info_center_y
        self.info2_ledit_ellipse_info_width = self.view.info2.ledit_ellipse_info_width
        self.info2_ledit_ellipse_info_height = self.view.info2.ledit_ellipse_info_height
        self.info2_ledit_ellipse_info_radian = self.view.info2.ledit_ellipse_info_radian
        self.info2_btn_load_current_ellipse_info = self.view.info2.btn_load_current_ellipse_info
        self.info2_btn_ellipse_info_change = self.view.info2.btn_ellipse_info_change
        self.info2_btn_ellipse_info_save = self.view.info2.btn_ellipse_info_save
        self.info2_vlist_ellipse_info_list = self.view.info2.vlist_ellipse_info_list
        self.info2_btn_erase_ellipse_info_list = self.view.info2.btn_erase_ellipse_info_list
        self.info2_btn_clear_ellipse_info_list = self.view.info2.btn_clear_ellipse_info_list
        self.info2_btn_draw_ellipse_with_points_4x = self.view.info2.btn_draw_ellipse_with_points_4x
        self.info2_btn_draw_ellipse_with_points_8x = self.view.info2.btn_draw_ellipse_with_points_8x


        self.info3_lbl_main_canvas2 = self.view.info3.lbl_main_canvas2
        self.info3_btn_edit_mask_empty = self.view.info3.btn_edit_mask_empty
        self.info3_btn_edit_mask_left = self.view.info3.btn_edit_mask_left
        self.info3_btn_edit_mask_right = self.view.info3.btn_edit_mask_right
        self.info3_btn_edit_mask_up = self.view.info3.btn_edit_mask_up
        self.info3_btn_edit_mask_down = self.view.info3.btn_edit_mask_down
        self.info3_btn_edit_mask_island = self.view.info3.btn_edit_mask_island
        self.info3_btn_edit_mask_fill_contour = self.view.info3.btn_edit_mask_fill_contour
        self.info3_btn_edit_mask_1x = self.view.info3.btn_edit_mask_1x
        self.info3_btn_edit_mask_2x = self.view.info3.btn_edit_mask_2x
        self.info3_btn_edit_mask_4x = self.view.info3.btn_edit_mask_4x
        self.info3_btn_edit_mask_8x = self.view.info3.btn_edit_mask_8x
        self.info3_btn_edit_mask_16x = self.view.info3.btn_edit_mask_16x
        self.info3_btn_load_current_mask = self.view.info3.btn_load_current_mask
        self.info3_btn_load_mask_file = self.view.info3.btn_load_mask_file
        self.info3_btn_save_temp_mask = self.view.info3.btn_save_temp_mask
        self.info3_btn_save_file_mask = self.view.info3.btn_save_file_mask
        self.info3_vlist_maskimgs = self.view.info3.vlist_maskimgs
        self.info3_btn_erase_vlist_maskimgs = self.view.info3.btn_erase_vlist_maskimgs
        self.info3_btn_clear_vlist_maskimgs = self.view.info3.btn_clear_vlist_maskimgs


        self.connect_signals()


    def connect_signals(self):
        self.info1_btn_open_video.clicked.connect(self.cb_info1_btn_open_video)
        self.info1_btn_open_folder.clicked.connect(self.cb_info1_btn_open_folder)
        self.info1_btn_set_lt.clicked.connect(self.cb_info1_btn_set_lt)
        self.info1_btn_set_rt.clicked.connect(self.cb_info1_btn_set_rt)
        self.info1_btn_extract_index.clicked.connect(self.cb_info1_btn_extract_index)

        self.info2_btn_goto_index.clicked.connect(self.cb_info2_btn_goto_index)
        self.info2_btn_prev_index.clicked.connect(self.cb_info2_btn_prev_index)
        self.info2_btn_next_index.clicked.connect(self.cb_info2_btn_next_index)
        self.info2_btn_play_frames.clicked.connect(self.cb_info2_btn_play_frames)
        self.info2_btn_stop_frames.clicked.connect(self.cb_info2_btn_stop_frames)
        self.info2_btn_set_orig.clicked.connect(self.cb_info2_btn_set_orig)
        self.info2_btn_set_rois.clicked.connect(self.cb_info2_btn_set_rois)
        self.info2_btn_set_ellipse.clicked.connect(self.cb_info2_btn_set_ellipse)
        self.info2_btn_get_mask1.clicked.connect(self.cb_info2_btn_get_mask1)
        self.info2_btn_set_calib_ellipse1.clicked.connect(self.cb_info2_btn_set_calib_ellipse1)
        self.info2_btn_set_calib_ellipse2.clicked.connect(self.cb_info2_btn_set_calib_ellipse2)
        self.info2_btn_resize_2x.clicked.connect(self.cb_info2_btn_resized_2x)
        self.info2_btn_resize_4x.clicked.connect(self.cb_info2_btn_resized_4x)
        self.info2_btn_load_current_ellipse_info.clicked.connect(self.cb_info2_btn_load_current_ellipse_info)
        self.info2_btn_ellipse_info_change.clicked.connect(self.cb_info2_btn_ellipse_info_change)
        self.info2_btn_ellipse_info_save.clicked.connect(self.cb_info2_btn_ellipse_info_save)
        # self.info2_vlist_ellipse_info_list.currentItemChanged.connect(self.cb_info2_vlist_ellipse_info_list)
        self.info2_vlist_ellipse_info_list.itemDoubleClicked.connect(self.cb_info2_vlist_ellipse_info_list)
        self.info2_btn_erase_ellipse_info_list.clicked.connect(self.cb_info2_btn_erase_ellipse_info_list)
        self.info2_btn_clear_ellipse_info_list.clicked.connect(self.cb_info2_btn_clear_ellipse_info_list)
        self.info2_btn_draw_ellipse_with_points_4x.clicked.connect(self.cb_info2_btn_draw_ellipse_with_points_4x)
        self.info2_btn_draw_ellipse_with_points_8x.clicked.connect(self.cb_info2_btn_draw_ellipse_with_points_8x)


        self.info3_btn_edit_mask_empty.clicked.connect(self.cb_info3_btn_edit_mask_empty)
        self.info3_btn_edit_mask_left.clicked.connect(self.cb_info3_btn_edit_mask_left)
        self.info3_btn_edit_mask_right.clicked.connect(self.cb_info3_btn_edit_mask_right)
        self.info3_btn_edit_mask_up.clicked.connect(self.cb_info3_btn_edit_mask_up)
        self.info3_btn_edit_mask_down.clicked.connect(self.cb_info3_btn_edit_mask_down)
        self.info3_btn_edit_mask_island.clicked.connect(self.cb_info3_btn_edit_mask_island)
        self.info3_btn_edit_mask_fill_contour.clicked.connect(self.cb_info3_btn_edit_mask_fill_contour)

        self.info3_btn_edit_mask_1x.clicked.connect(self.cb_info3_btn_edit_mask_1x)
        self.info3_btn_edit_mask_2x.clicked.connect(self.cb_info3_btn_edit_mask_2x)
        self.info3_btn_edit_mask_4x.clicked.connect(self.cb_info3_btn_edit_mask_4x)
        self.info3_btn_edit_mask_8x.clicked.connect(self.cb_info3_btn_edit_mask_8x)
        self.info3_btn_edit_mask_16x.clicked.connect(self.cb_info3_btn_edit_mask_16x)
        self.info3_btn_load_current_mask.clicked.connect(self.cb_info3_btn_load_current_mask)
        self.info3_btn_load_mask_file.clicked.connect(self.cb_info3_btn_load_mask_file)
        self.info3_btn_save_temp_mask.clicked.connect(self.cb_info3_btn_save_temp_mask)
        self.info3_btn_save_file_mask.clicked.connect(self.cb_info3_btn_save_file_mask)
        self.info3_vlist_maskimgs.itemDoubleClicked.connect(self.cb_info3_vlist_maskimgs)
        self.info3_btn_erase_vlist_maskimgs.clicked.connect(self.cb_info3_btn_erase_vlist_maskimgs)
        self.info3_btn_clear_vlist_maskimgs.clicked.connect(self.cb_info3_btn_clear_vlist_maskimgs)
    
    # # @QtCore.pyqtSlot('QObject*')
    # def on_child_destroyed(self, obj):
    #     self.subviews = [item for item in self.subviews if not sip.isdeleted(item)]
    #     # del obj
    #     self.subviews = []
    #     # self.subview.close()
    #     # del self.subview
    #     # print("obj:", obj)
    #     print("after deleted len(self.subviews):", len(self.subviews))

    def get_current_base_npimg(self):
        if not self.l_frames:
            return None
        if self.flag_video_on and not self.flag_open_folder:
            if self.current_location == 'Lt' or self.current_location == 'lt':
                img_color = self.l_frames[self.current_index][:,self.MODEL_WIDTH:]
            elif self.current_location == 'Rt' or self.current_location == 'rt':
                img_color = self.l_frames[self.current_index][:,:self.MODEL_WIDTH]
            return img_color
        elif self.flag_video_on and self.flag_open_folder:
            img_color = self.l_frames[self.current_index]
            return img_color

    def get_pred1_npimg(self):
        if not self.l_frames_gray:
            return
        if self.flag_video_on and not self.flag_open_folder:
            if self.current_location == 'Lt' or self.current_location == 'lt':
                img_gray = self.l_frames_gray[self.current_index][:,self.MODEL_WIDTH:]
                # img_color = self.l_frames[self.current_index][:,self.MODEL_WIDTH:]
            if self.current_location == 'Rt' or self.current_location == 'rt':
                img_gray = self.l_frames_gray[self.current_index][:,:self.MODEL_WIDTH]
                # img_color = self.l_frames[self.current_index][:,:self.MODEL_WIDTH]
        elif self.flag_video_on and self.flag_open_folder:
            img_gray = self.l_frames_gray[self.current_index]

        img_gray = img_gray / 255.
        img_gray = np.expand_dims(img_gray, axis=-1)
        pred1 = self.model1.predict(img_gray[np.newaxis,...])
        pred1 = pred1.squeeze()
        pred1 = (pred1 > 0.5).astype(np.uint8)
        island = isolate_islands(pred1)

        return island

    def get_ellipse_mask(self,ellipse_info):
        if not ellipse_info:
            return None
        center, w, h, radian = ellipse_info
        mask = np.zeros((self.MODEL_HEIGHT,self.MODEL_WIDTH), dtype=np.uint8)
        mask = cv2.ellipse(mask, (int(center[0]), int(center[1])), (int(w), int(h)), int(np.rad2deg(radian)),0,360, (255,255,255), -1)    

        return mask

    # this is for when validation of orig-mask with using open folder  
    def set_relpace_list_of_mask_item(self, mask):
        if not isinstance(mask,numpy.ndarray):
            return
        if self.flag_video_on and self.flag_open_folder:
            self.l_mask_imgs[self.current_index] = mask.copy()


    def set_current_maskimg(self):
        if self.flag_video_on and self.flag_open_folder:
            self.current_maskimg = self.l_mask_imgs[self.current_index]
        return True

    def set_current_base_npimg(self):
        self.current_npimg = self.get_current_base_npimg()
        return True

    def set_current_index(self,n):
        if not self.flag_video_on:
            return False
        if n > self.max_frame_index:
            self.current_index = self.max_frame_index
            return True
        self.current_index = n
        return True

    def set_current_index_lbl(self):
        if not self.flag_video_on:
            return False
        if self.current_index > self.max_frame_index:
            self.current_index = self.max_frame_index
        if self.flag_open_folder:
            self.info2_lbl_current_index.setText(f'{self.current_index} / {self.max_frame_index} : {self.l_save_indices[self.current_index]}')
        else:
            self.info2_lbl_current_index.setText(f'{self.current_index} / {self.max_frame_index}')
        return True
    
    def set_current_location(self):
        self.info1_lbl_current_location.setText(self.current_location)
        return True
    
    def set_lbl_img(self, lbl, npimg):
        lbl.clear()
        npimg_copy = npimg.copy()
        if np.ndim(npimg) == 3:
            npimg_cvt = cv2.cvtColor(npimg_copy, cv2.COLOR_BGR2RGB)
        elif np.ndim(npimg) == 2:
            npimg_cvt = cv2.cvtColor(npimg_copy, cv2.COLOR_GRAY2RGB)
        else:
            print('np.ndim is not 2 or 3')
            return

        qimg = QtGui.QImage(npimg_cvt.data, npimg_cvt.shape[1], npimg_cvt.shape[0], npimg_cvt.shape[1]*3, QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        lbl.setPixmap(pixmap)
        return True

    def set_info3_main_canvas2(self):
        if not isinstance(self.current_maskimg,numpy.ndarray):
            QMessageBox.information(None,'QMessageBox', 'from set_info3_main_canvas : None self.current_maskimg')
            return
        self.set_lbl_img(self.info3_lbl_main_canvas2, self.current_maskimg)

    def set_info2_main_canvas1(self):
        if not isinstance(self.current_npimg,numpy.ndarray):
            return
        self.set_lbl_img(self.info2_lbl_main_canvas1, self.current_npimg)
        return True

    def set_ellipse_info(self):
        if self.ellipse_info == None:
            self.info2_lbl_ellipse_info.setText('no ellipse_info')
            return
        centers, w, h, radian = self.ellipse_info
        self.info2_lbl_ellipse_info.setText(f'x:{round(centers[0],3)}, y:{round(centers[1],3)}, w:{round(w,3)}, h:{round(h,3)}, rad:{round(radian,3)}')


    def cb_info1_btn_open_video(self):
        
        fname = QFileDialog.getOpenFileName(None,'Open file', './')
        if fname[0]:
            # print(fname)
            cap = cv2.VideoCapture(fname[0])
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frameWidth != 320 or frameHeight != 120:
                print(f'framWidth({frameWidth}) != 320, frameHeight({frameHeight}) != 120')
                return
            ######## check if video file selected
            retval, frame = cap.read()
            if not retval:
                return
            else :
                self.CURRENT_FILE_FULL_PATH = fname[0]
                self.FILE_PATH, self.FILE_NAME_WITH_EXT = os.path.split(self.CURRENT_FILE_FULL_PATH)
                self.FILE_NAME, self.FILE_EXT = os.path.splitext(self.FILE_NAME_WITH_EXT)
                self.view.setWindowTitle('roi tool / ' + self.FILE_NAME_WITH_EXT)

                # frame = cv2.resize(frame,(self.MODEL_WIDTH *2, self.MODEL_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
                frame = cv2.resize(frame,(self.MODEL_WIDTH *2, self.MODEL_HEIGHT), interpolation=cv2.INTER_CUBIC)
                frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                self.l_frames = [frame]
                self.l_frames_gray = [frame_gray]
                ########### extract all of frames
                while True:
                    retval, frame = cap.read()
                    if not retval:
                        break
                    else:
                        frame = cv2.resize(frame,(self.MODEL_WIDTH *2, self.MODEL_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
                        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                        self.l_frames.append(frame)
                        self.l_frames_gray.append(frame_gray)

                self.flag_video_on = True
                self.flag_open_folder = False
                self.current_index = 0
                self.current_location = self.info1_lbl_current_location.text()
                self.info1_btn_set_lt.setDisabled(False)
                self.info1_btn_set_rt.setDisabled(False)
                self.max_frame_index = len(self.l_frames) - 1
                # self.l_save_indices for index extractor
                self.l_save_indices = [i for i in range(self.max_frame_index+1)]
                # set save path for video loading
                self.SAVE_PATH = f'{self.SAVE_BASE_PATH}/{self.FILE_NAME}'
                self.set_current_index_lbl()
                self.set_current_base_npimg()
                self.current_maskimg = np.zeros((self.current_npimg.shape[0],self.current_npimg.shape[1]), dtype=np.uint8)
                self.set_current_maskimg()
                self.set_info3_main_canvas2()
                

                ## their are some problems with release subview. guessing Qthread blocking __del__ 
                # if self.subview1 != None:
                #     print('del')
                #     del self.subview1
                # self.subview1 = None

                # if len(self.subviews) != 0:
                #     for i in range(len(self.subviews)):
                #         print('new_video_loded. prev_subview_delete')
                #         del self.subviews[i]
                #     self.subviews = []
                
                self.l_resized_frames_lt = []
                self.l_resized_frames_rt = []

                self.FILE_PATH, self.FILE_NAME, self.MODEL_NAME

                # pkl_file = os.path.join(self.FILE_PATH,f'{self.FILE_NAME}_{self.MODEL_NAME}_indices.pkl')
                pkl_file = f'{self.FILE_PATH}/{self.FILE_NAME}_{self.MODEL_NAME}_indices.pkl'
                if os.path.exists(pkl_file):
                    with open(pkl_file, 'rb') as f:
                        self.d_inferred_info = pickle.load(f)
                else:
                    self.d_inferred_info = None
                    print('cannot found pkl')


                try:
                    self.set_info2_main_canvas1()
                except Exception as e:
                    QMessageBox.information(None, 'QMessageBox',f'{e}')


    def cb_info1_btn_open_folder(self):
        file = QFileDialog.getExistingDirectory(None,"Select Directory")
        if file:
            files = os.listdir(file)
            re_orig = re.compile('([0-9]+)_orig.png')
            re_mask = re.compile('([0-9]+)_mask.png')
            l_orig = []
            l_mask = []
            for i in files:
                m_orig = re_orig.match(i)
                m_mask = re_mask.match(i)
                if m_orig:
                    l_orig.append(int(m_orig.group(1)))
                if m_mask:
                    l_mask.append(int(m_mask.group(1)))
            if len(l_orig) != len(l_mask):
                QMessageBox.information(None,"QMessageBox","from cb_info1_btn_open_folder: the number of orig and mask not same")
                return
            if len(l_orig) == 0:
                QMessageBox.information(None,"QMessageBox","from cb_info1_btn_open_folder: could not found orig and mask")
                return

            # set save_path for folder loading 
            self.SAVE_PATH = f'{file}/..'
            l_orig = sorted(l_orig)
            self.flag_video_on = True
            self.flag_open_folder = True
            self.l_frames = []
            self.l_frames_gray = []
            self.l_mask_imgs = []
            for i in l_orig:
                self.l_frames.append(cv2.imread(f'{file}/{i}_orig.png'))
                self.l_frames_gray.append(cv2.imread(f'{file}/{i}_orig.png',cv2.IMREAD_GRAYSCALE))
                self.l_mask_imgs.append(cv2.imread(f'{file}/{i}_mask.png',cv2.IMREAD_GRAYSCALE))
            self.max_frame_index = len(self.l_frames)-1
            self.current_index = 0
            self.l_save_indices = sorted(l_orig)
            self.view.setWindowTitle('roi tool / ' + file)
            self.info1_lbl_current_location.setText(file.split("/")[-1])
            # self.info1_lbl_current_location.setText(file.split("/")[-1].lower())
            self.current_location = self.info1_lbl_current_location.text()
            self.info1_btn_set_lt.setDisabled(True)
            self.info1_btn_set_rt.setDisabled(True)
            self.set_current_index_lbl()
            self.set_current_base_npimg()
            self.set_current_maskimg()
            self.set_info2_main_canvas1()
            self.set_info3_main_canvas2()

        
    def cb_info1_btn_set_lt(self):
        self.current_location = 'Lt'
        self.set_current_location()
        self.set_current_index_lbl()
        self.set_current_base_npimg()
        self.set_current_maskimg()
        self.set_info2_main_canvas1()

    def cb_info1_btn_set_rt(self):
        self.current_location = 'Rt'
        self.set_current_location()
        self.set_current_index_lbl()
        self.set_current_base_npimg()
        self.set_current_maskimg()
        self.set_info2_main_canvas1()

    def cb_info1_btn_extract_index(self):
        if not self.flag_video_on:
            QMessageBox.information(None, "QMessageBox", f"from cb_info1_btn_extract_index: video not loaded")
            return
        if not self.l_resized_frames_lt:
           if self.l_frames[0].shape != (self.MODEL_HEIGHT * 4, self.MODEL_WIDTH * 2, 1) and self.l_frames[0].shape != (self.MODEL_HEIGHT *4, self.MODEL_WIDTH *2):
                self.l_resized_frames_lt = []
                for i in self.l_frames:
                    result = cv2.cvtColor(cv2.resize(i,(self.MODEL_WIDTH*4,self.MODEL_HEIGHT*2),interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
                    self.l_resized_frames_lt.append(result[:,self.MODEL_WIDTH *2:])
                    self.l_resized_frames_rt.append(result[:,:self.MODEL_WIDTH*2])

        if isinstance(self.d_inferred_info,dict):
            d_lt_inferred_info = {'centers': self.d_inferred_info['left_centers'], 'roundnesses': self.d_inferred_info['left_roundnesses'], 'widths': self.d_inferred_info['left_widths'], \
                                    'heights':self.d_inferred_info['left_heights'], 'radians': self.d_inferred_info['left_radians']}
            d_rt_inferred_info = {'centers': self.d_inferred_info['right_centers'], 'roundnesses': self.d_inferred_info['right_roundnesses'], 'widths': self.d_inferred_info['right_widths'], \
                                    'heights':self.d_inferred_info['right_heights'], 'radians': self.d_inferred_info['right_radians']}
        
        else:
            reply = QMessageBox.question(self.view, "center sequences required", "do you want to make center sequences file for extracting index?", QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            self.d_inferred_info = self.myfe.extract_ellipse_infos_dict_on_video(self.CURRENT_FILE_FULL_PATH, self.MODEL_NAME, from_loaded_video_images=self.l_frames_gray)
            
            save_file = f'{self.FILE_PATH}/{self.FILE_NAME}_{self.MODEL_NAME}_indices.pkl'
            with open(save_file,'wb') as f:
                pickle.dump(self.d_inferred_info,f)
            
                        
            d_lt_inferred_info = {'centers': self.d_inferred_info['left_centers'], 'roundnesses': self.d_inferred_info['left_roundnesses'], 'widths': self.d_inferred_info['left_widths'], \
                                    'heights':self.d_inferred_info['left_heights'], 'radians': self.d_inferred_info['left_radians']}
            d_rt_inferred_info = {'centers': self.d_inferred_info['right_centers'], 'roundnesses': self.d_inferred_info['right_roundnesses'], 'widths': self.d_inferred_info['right_widths'], \
                                    'heights':self.d_inferred_info['right_heights'], 'radians': self.d_inferred_info['right_radians']}
        

        # # self.subview1 = roi_tool_subview1.SubGui1(self.l_resized_frames_lt,self.l_resized_frames_rt, d_lt_inferred_info, d_rt_inferred_info, self)
        # # self.subview1.setWindowTitle(f"index extractor / {self.FILE_NAME_WITH_EXT}")
        # print("show subview_ len(self.subviews):", len(self.subviews))
        # if len(self.subviews) == 0:
        #     self.subview = roi_tool_subview1.SubGui1(self.l_resized_frames_lt,self.l_resized_frames_rt, d_lt_inferred_info, d_rt_inferred_info, self)
        #     self.subview.setWindowTitle(f"index extractor / {self.FILE_NAME_WITH_EXT}")
        #     # https://stackoverflow.com/questions/55870336/how-to-empty-the-ram-after-closing-the-second-window
        #     # self.subviews.append(weakref.ref(subview, self.subviews.remove))
        #     self.subviews.append(self.subview)
        # # QMessageBox.information(None, "QMessageBox", f"from cb_info1_btn_extract_index: done")
        # # self.subview1.show()
        # # self.subviews[-1].show()
        if not isinstance(self.subview,roi_tool_subview1.SubGui1):
            print("not instance")
            self.subview = roi_tool_subview1.SubGui1(data_lt = self.l_resized_frames_lt, data_rt = self.l_resized_frames_rt, d_inferred_info_lt=d_lt_inferred_info, d_inferred_info_rt=d_rt_inferred_info, parent =self)
            self.subview.setWindowTitle(f"index extractor / {self.FILE_NAME_WITH_EXT}")
        elif self.CURRENT_FILE_FULL_PATH != self.subview.CURRENT_FILE_FULL_PATH:
            print("extractor_set_new_data")
            self.subview.set_new_data(data_lt = self.l_resized_frames_lt, data_rt = self.l_resized_frames_rt, d_inferred_info_lt=d_lt_inferred_info, d_inferred_info_rt=d_rt_inferred_info, parent =self)
            self.subview.setWindowTitle(f"index extractor / {self.FILE_NAME_WITH_EXT}")
        self.subview.show()
        
    def cb_info2_btn_goto_index(self):
        if not self.flag_video_on:
            return False
        index = self.info2_ledit_goto_index.text()
        if not index.isdigit():
            return
        index = int(index)
        if index < 0 :
            index = 0
        if index > self.max_frame_index:
            index = self.max_frame_index
        self.current_index = index
        self.set_current_index_lbl()
        self.set_current_base_npimg()
        self.set_current_maskimg()
        self.set_info2_main_canvas1()

    def cb_info2_btn_prev_index(self):
        if not self.flag_video_on:
            return False
        index = self.current_index
        index -= 1
        if index < 0 :
            index = 0
        self.current_index = index
        self.set_current_index_lbl()
        self.set_current_base_npimg()
        self.set_current_maskimg()
        self.set_info2_main_canvas1()
        self.set_info3_main_canvas2()
        return True

    def cb_info2_btn_next_index(self):
        if not self.flag_video_on:
            return False
        index = self.current_index
        index += 1
        if index > self.max_frame_index :
            index = self.max_frame_index
        self.current_index = index
        if self.set_current_index_lbl() and self.set_current_base_npimg() and self.set_current_maskimg() and self.set_info2_main_canvas1() and self.set_info3_main_canvas2():
            return True
        return False

    def help_play_frames(self,int_):
        self.current_index = int_
        # print(int_)
        if self.set_current_index_lbl() and self.set_current_base_npimg() and self.set_current_maskimg() and self.set_info2_main_canvas1() and self.set_info3_main_canvas2():
            return True
        return False

    def cb_info2_btn_play_frames(self):
        if not self.flag_video_on:
            return False

        self.flags["play frames"] = True
        self.worker = Worker(self.current_index,self.max_frame_index,self.flags)
        self.worker.running = True
        # self.thread = QThread()
        # thread = QThread()
        self.worker.moveToThread(self.thread)
        # worker.moveToThread(thread)
        self.thread.started.connect(self.worker.run)
        # thread.started.connect(worker.run)
        self.worker.waitted.connect(self.thread.wait)
        self.worker.finished.connect(self.thread.quit)
        # worker.finished.connect(thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        # worker.finished.connect(worker.deleteLater)
        # self.worker.finished.connect(self.thread.deleteLater)
        # worker.finished.connect(thread.deleteLater)
        self.worker.progress.connect(self.help_play_frames)
        self.stopper.stopper.connect(self.worker.set_running)
        # thread.start()
        self.thread.start()
        self.info2_btn_play_frames.setEnabled(False)
        self.worker.finished.connect(lambda : self.info2_btn_play_frames.setEnabled(True))

    def cb_info2_btn_stop_frames(self):
        if not self.flag_video_on:
            return
        # self.stopper.stopper.emit(False)
        # self.stopper.set_running(False)
        # pass
        # self.thread.finished()
        if not self.worker:
            return
        self.worker.running = False
        # self.thread.wait()
        # self.flags["play frames"] = False
        # self.thread.quit()
        # self.thread.terminate()
        
        # self.info2_btn_play_frames.setEnabled(True)

    def cb_info2_btn_set_orig(self):
        if not self.flag_video_on:
            return False
        self.set_current_base_npimg()
        self.set_info2_main_canvas1()

    def cb_info2_btn_set_rois(self):
        if not self.flag_video_on:
            return False
        pred1 = self.get_pred1_npimg()
        palate = self.get_current_base_npimg()
        self.ellipse_info = None
        self.set_ellipse_info()
        self.current_maskimg = pred1
        img_result = self.eval_tool.draw_pred_roi(palate, pred1, color=(122,121,255))
        self.current_npimg = img_result
        self.set_info2_main_canvas1()
        self.set_info3_main_canvas2()

    def cb_info2_btn_set_ellipse(self):
        if not self.flag_video_on:
            return False
        pred1 = self.get_pred1_npimg()
        palate = self.get_current_base_npimg()
        self.ellipse_info = self.eval_tool.get_ellipse_info(pred1)
        if not self.ellipse_info:
            return
        self.set_ellipse_info()
        self.current_maskimg = self.get_ellipse_mask(self.ellipse_info)
        img_result = self.eval_tool.draw_pred_ellipse(palate,pred1,color=(122,121,255))
        self.current_npimg = img_result
        self.set_info2_main_canvas1()
        self.set_info3_main_canvas2()

    def cb_info2_btn_set_calib_ellipse1(self):
        if not self.flag_video_on:
            return False
        pred1 = self.get_pred1_npimg()
        palate = self.get_current_base_npimg()
        self.ellipse_info = self.eval_tool.get_calib_ellipse_info1(pred1)
        if not self.ellipse_info:
            return
        self.set_ellipse_info()
        self.current_maskimg = self.get_ellipse_mask(self.ellipse_info)
        img_result = self.eval_tool.draw_pred_calib_ellipse1(palate,pred1,color=(122,121,255))
        self.current_npimg = img_result
        self.set_info2_main_canvas1()
        self.set_info3_main_canvas2()

    def cb_info2_btn_set_calib_ellipse2(self):
        if not self.flag_video_on:
            return False
        pred1 = self.get_pred1_npimg()
        palate = self.get_current_base_npimg()
        self.ellipse_info = self.eval_tool.get_calib_ellipse_info2(pred1)
        if not self.ellipse_info:
            return
        self.set_ellipse_info()
        self.current_maskimg = self.get_ellipse_mask(self.ellipse_info)
        img_result = self.eval_tool.draw_pred_calib_ellipse2(palate,pred1,color=(122,121,255))
        self.current_npimg = img_result
        self.set_info2_main_canvas1()
        self.set_info3_main_canvas2()

    def cb_info2_btn_get_mask1(self):
        self.set_info3_main_canvas2()
        
    def cb_info2_btn_resized_2x(self):
        if not self.flag_video_on:
            return False
        npimg = self.current_npimg
        npimg = cv2.resize(npimg,(npimg.shape[1]*2,npimg.shape[0]*2),interpolation=cv2.INTER_LANCZOS4)
        cv2.imshow('2x', npimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def cb_info2_btn_resized_4x(self):
        if not self.flag_video_on:
            return False
        npimg = self.current_npimg
        npimg = cv2.resize(npimg,(npimg.shape[1]*4,npimg.shape[0]*4),interpolation=cv2.INTER_LANCZOS4)
        cv2.imshow('4x', npimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def cb_info2_btn_load_current_ellipse_info(self):
        if self.ellipse_info == None:
            return
        centers, w, h, radian = self.ellipse_info
        self.info2_ledit_ellipse_info_center_x.setText(f'{round(centers[0],3)}')
        self.info2_ledit_ellipse_info_center_y.setText(f'{round(centers[1],3)}')
        self.info2_ledit_ellipse_info_width.setText(f'{round(w,3)}')
        self.info2_ledit_ellipse_info_height.setText(f'{round(h,3)}')
        self.info2_ledit_ellipse_info_radian.setText(f'{round(radian,3)}')


    def cb_info2_btn_ellipse_info_change(self):
        center_x = self.info2_ledit_ellipse_info_center_x.text()
        center_y = self.info2_ledit_ellipse_info_center_y.text()
        width = self.info2_ledit_ellipse_info_width.text()
        height = self.info2_ledit_ellipse_info_height.text()
        radian = self.info2_ledit_ellipse_info_radian.text()
        
        try:
            center_x = float(center_x)
            center_y = float(center_y)
            width = float(width)
            height = float(height)
            radian = float(radian)
        except Exception as e:
            print(f'Exception from cb_info2_btn_ellipse_info_change: {e}')
            return

        self.ellipse_info = ([center_x,center_y],width,height,radian)
        self.set_ellipse_info()
        self.current_maskimg = self.get_ellipse_mask(self.ellipse_info)

        palate = self.get_current_base_npimg()
        img_result = self.eval_tool.draw_with_ellipse_info(palate,self.ellipse_info,(122,122,255))
        self.current_npimg = img_result
        self.set_info2_main_canvas1()
        self.set_info3_main_canvas2()

    def cb_info2_btn_ellipse_info_save(self):
        if self.ellipse_info == None:
            return
        
        centers,width,height,radian = self.ellipse_info
        n = self.info2_vlist_ellipse_info_list.count()
        item = QListWidgetItem(f'{n}, {centers[0]}, {centers[1]}, {width}, {height}, {radian}')
        self.info2_vlist_ellipse_info_list.addItem(item)
        # d_ = {1:'123', 2: 0}
        # item.setData(Qt.ItemDataRole.UserRole,d_)
        # print(item.data(Qt.ItemDataRole.UserRole))
        # print(item.data(Qt.ItemDataRole.DisplayRole))


    def cb_info2_vlist_ellipse_info_list(self,index1):
        l_ = index1.text().split(',')
        if len(l_) != 6:
            return
        index, center_x, center_y, width, height, radian = l_
        try:
            center_x = float(center_x)
            center_y = float(center_y)
            width = float(width)
            height = float(height)
            radian = float(radian)
        except Exception as e:
            print(f'Exception from cb_info2_vlist_ellipse_info_list: {e}')
            return

        self.ellipse_info = ([center_x,center_y],width,height,radian)
        self.set_ellipse_info()

    def cb_info2_btn_erase_ellipse_info_list(self):
        current_item_row = self.info2_vlist_ellipse_info_list.currentRow()
        current_item = self.info2_vlist_ellipse_info_list.takeItem(current_item_row)
        self.info2_vlist_ellipse_info_list.removeItemWidget(current_item)

        n = self.info2_vlist_ellipse_info_list.count()
        new_list = []
        for i in range(n):
            item = self.info2_vlist_ellipse_info_list.item(i)
            l_ = item.text().split(',')
            l_.pop(0)
            l_.insert(0,str(i))
            str_ = ','.join(l_)
            new_list.append(str_)

        self.info2_vlist_ellipse_info_list.clear()
        for i in range(n):
            self.info2_vlist_ellipse_info_list.addItem(new_list[i])


    def cb_info2_btn_clear_ellipse_info_list(self):
        self.info2_vlist_ellipse_info_list.clear()


    def cb_draw_roi_rect(self,event,x,y,flags,param):
        img_copy = param.copy()
        if event == cv2.EVENT_LBUTTONUP:
            self.flag_lmouse_down = False
            self.H = int(y)-self.Y
            self.W = int(x)-self.X

        if event == cv2.EVENT_RBUTTONUP:
            self.flag_rmouse_down = False
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.flag_lmouse_down = True
            self.X = int(x)
            self.Y = int(y)
        
        if event == cv2.EVENT_MOUSEMOVE:
            if self.flag_lmouse_down == True:
                if self.X >= 0 or self.Y >= 0:
                    img_copy = cv2.rectangle(img_copy,(self.X, self.Y),(x,y),(0,255,0),1)
                    self.IMG = img_copy
        
        if event == cv2.EVENT_RBUTTONDBLCLK:
            self.IMG = img_copy

       
    def cb_info2_btn_draw_ellipse_with_points_4x(self):
        if not self.flag_video_on:
            return False
        modify_rate = 4
        self.IMG = self.get_current_base_npimg()
        drag_flag = 0

        # get roi area
        cv2.namedWindow('get_roi_to_4x')
        # cv2.setMouseCallback('get_roi_to_4x',self.cb_draw_roi_rect, self.get_current_base_npimg())
        cv2.setMouseCallback('get_roi_to_4x',self.cb_draw_roi_rect, self.IMG)
        while True:
            cv2.imshow('get_roi_to_4x',self.IMG)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break


        # after getting roi araa, draw ellipse by select pixels
        img = self.get_current_base_npimg()
        img = img[min(self.Y,self.Y+self.H):max(self.Y,self.Y+self.H),min(self.X,self.X+self.W):max(self.X,self.X+self.W)]
        img_resized = self.help_make_resized_img(img,modify_rate)
        
        self.MASK = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.namedWindow('draw_ellipse_4x')
        cv2.setMouseCallback('draw_ellipse_4x',self.help_cb_draw_ellipse_with_points_4x,img_resized)

        while True:
            if np.count_nonzero(self.MASK) > 5:
                input_points = np.where(self.MASK != 0)
                if np.unique(input_points[0]).shape[0] < 6 or np.unique(input_points[1]).shape[0] < 6:
                    pass
                else:
                    try:
                        vertices = np.array([input_points[0], input_points[1]]).T
                        # Contour
                        fitted = LSqEllipse()
                        fitted.fit([vertices[:,1], vertices[:,0]])
                        center, w,h, radian = fitted.parameters()
                        if center:
                            drag_flag = 1
                        # Because of the np indexing of y-axis, orientation needs to be minus
                        img_ellipsed = cv2.ellipse(img.copy(), (int(center[0]), int(center[1])), (int(w), int(h)), int(np.rad2deg(radian)),0,360, (0,200,0), 1)    
                        img_resized = self.help_make_resized_img(img_ellipsed,modify_rate)
                        
                    except Exception as e:
                        QMessageBox.information(None,'QMessageBox',f'from cb_info2_btn_draw_ellipse_with_points_4x : {e}')

                    img_resized = self.help_change_resized_view_color_and_mask(self.MASK,img_resized,modify_rate=modify_rate,direction='PLUS')

            cv2.imshow('draw_ellipse_4x',img_resized)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break
        
        if drag_flag:
            mask = np.zeros_like(self.MASK)
            mask = cv2.ellipse(mask,(int(center[0]), int(center[1])), (int(w), int(h)), int(np.rad2deg(radian)),0,360, (255,255,255), -1)
            self.current_maskimg = np.zeros((self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.uint8)
            self.current_maskimg[min(self.Y,self.Y+self.H): max(self.Y, self.Y+self.H), min(self.X, self.X+self.W):max(self.X, self.X+self.W)] = mask
            self.set_relpace_list_of_mask_item(self.current_maskimg)
            # self.set_current_maskimg()
            self.set_info3_main_canvas2()


    def cb_info2_btn_draw_ellipse_with_points_8x(self):
        if not self.flag_video_on:
            return False
        modify_rate = 8
        self.IMG = self.get_current_base_npimg()
        flag_draw = 0
        cv2.namedWindow('get_roi_to_8x')
        # cv2.setMouseCallback('get_roi_to_8x',self.cb_draw_roi_rect, self.get_current_base_npimg())
        cv2.setMouseCallback('get_roi_to_8x',self.cb_draw_roi_rect, self.IMG)
        while True:
            cv2.imshow('get_roi_to_8x',self.IMG)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break

        img = self.get_current_base_npimg()
        img = img[min(self.Y,self.Y+self.H):max(self.Y,self.Y+self.H),min(self.X,self.X+self.W):max(self.X,self.X+self.W)]
        img_resized = self.help_make_resized_img(img,modify_rate)
        
        self.MASK = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.namedWindow('draw_ellipse_8x')
        cv2.setMouseCallback('draw_ellipse_8x',self.help_cb_draw_ellipse_with_points_8x,img_resized)
        while True:
            if np.count_nonzero(self.MASK) > 5:
                input_points = np.where(self.MASK != 0)
                if np.unique(input_points[0]).shape[0] < 6 or np.unique(input_points[1]).shape[0] < 6:
                    pass
                else:
                    try:
                        vertices = np.array([input_points[0], input_points[1]]).T
                        # Contour
                        fitted = LSqEllipse()
                        fitted.fit([vertices[:,1], vertices[:,0]])
                        center, w,h, radian = fitted.parameters()
                        if center:
                            flag_draw = 1
                        # Because of the np indexing of y-axis, orientation needs to be minus
                        img_ellipsed = cv2.ellipse(img.copy(), (int(center[0]), int(center[1])), (int(w), int(h)), int(np.rad2deg(radian)),0,360, (0,200,0), 1)    
                        img_resized = self.help_make_resized_img(img_ellipsed,modify_rate)
                        
                    except Exception as e:
                        QMessageBox.information(None,'QMessageBox',f'from cb_info2_btn_draw_ellipse_with_points_8x : {e}')

                    img_resized = self.help_change_resized_view_color_and_mask(self.MASK,img_resized,modify_rate=modify_rate,direction='PLUS')

            cv2.imshow('draw_ellipse_8x',img_resized)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break
        
        if flag_draw:
            mask = np.zeros_like(self.MASK)
            mask = cv2.ellipse(mask,(int(center[0]), int(center[1])), (int(w), int(h)), int(np.rad2deg(radian)),0,360, (255,255,255), -1)
            self.current_maskimg = np.zeros((self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=np.uint8)
            self.current_maskimg[min(self.Y,self.Y+self.H): max(self.Y, self.Y+self.H), min(self.X, self.X+self.W):max(self.X, self.X+self.W)] = mask
            self.set_relpace_list_of_mask_item(self.current_maskimg)
            # self.set_current_maskimg()
            self.set_info3_main_canvas2()

    def cb_info3_btn_edit_mask_empty(self):
        if not isinstance(self.current_maskimg, numpy.ndarray):
            return
        new_mask = np.zeros_like(self.current_maskimg)
        self.current_maskimg = new_mask
        self.set_relpace_list_of_mask_item(new_mask)
        # self.set_current_maskimg()
        self.set_info3_main_canvas2()

    def cb_info3_btn_edit_mask_left(self):
        if not isinstance(self.current_maskimg, numpy.ndarray):
            return
        new_mask = np.zeros_like(self.current_maskimg)
        new_mask[:,:-1] = self.current_maskimg[:,1:]
        self.current_maskimg = new_mask
        self.set_relpace_list_of_mask_item(new_mask)
        # self.set_current_maskimg()
        self.set_info3_main_canvas2()

    def cb_info3_btn_edit_mask_right(self):
        if not isinstance(self.current_maskimg, numpy.ndarray):
            return
        new_mask = np.zeros_like(self.current_maskimg)
        new_mask[:,1:] = self.current_maskimg[:,:-1]
        self.current_maskimg = new_mask
        self.set_relpace_list_of_mask_item(new_mask)
        # self.set_current_maskimg()
        self.set_info3_main_canvas2()

    def cb_info3_btn_edit_mask_up(self):
        if not isinstance(self.current_maskimg, numpy.ndarray):
            return
        new_mask = np.zeros_like(self.current_maskimg)
        new_mask[:-1,:] = self.current_maskimg[1:,:]
        self.current_maskimg = new_mask
        self.set_relpace_list_of_mask_item(new_mask)
        # self.set_current_maskimg()
        self.set_info3_main_canvas2()

    def cb_info3_btn_edit_mask_down(self):
        if not isinstance(self.current_maskimg, numpy.ndarray):
            return
        new_mask = np.zeros_like(self.current_maskimg)
        new_mask[1:,:] = self.current_maskimg[:-1,:]
        self.current_maskimg = new_mask
        self.set_relpace_list_of_mask_item(new_mask)
        # self.set_current_maskimg()
        self.set_info3_main_canvas2()

    def get_island_img_with_info(self, img):
        # result = [image, max_x, max_y, max_w, max_w, max_area]
        results = [img, None, None, None, None, None]

        if not isinstance(img, numpy.ndarray):
            QMessageBox.information(None, "QMessageBox", "from get_island_img: the input img is not ndarray")
            return results
        
        # check if the input img is binary
        n = np.count_nonzero(img)
        if n == 0:
            return results
        else:
            ave_ = np.sum(img)/n
            if ave_ != 1.0 and ave_ != 255.0:
                QMessageBox.information(None, "QMessageBox", f"from get_island_img: the input img is not binary (average = {ave_}")
                return results
        
        # make island img. island means remaining only biggest connected area
        img_island = np.zeros_like(img)

        retval, labelled, stats, centroids = cv2.connectedComponentsWithStats(img)

        max_area = 0
        max_label = 0
        max_x = 0
        max_y = 0
        max_w = 0
        max_h = 0
        for index, stat in enumerate(stats[1:]):
            x,y,w,h,area = stat
            if area > max_area:
                max_area = area
                max_label = index + 1
                max_x = x
                max_y = y
                max_w = w
                max_h = h

        if max_label == 0:
            return results
        else:
            img_island[labelled == max_label] = 255
            results = [img_island, max_x, max_y, max_w, max_h, max_area]
            return results
    
    def cb_info3_btn_edit_mask_island(self):
        if not isinstance(self.current_maskimg, numpy.ndarray):
            return
        results = self.get_island_img_with_info(self.current_maskimg)
        
        self.current_maskimg = results[0]
        self.set_relpace_list_of_mask_item(results[0])
        # self.set_current_maskimg()
        self.set_info3_main_canvas2()


    def cb_info3_btn_edit_mask_fill_contour(self):
        if not isinstance(self.current_maskimg, numpy.ndarray):
            QMessageBox.information(None,'QMessageBox',f"from cb_info3_btn_edit_mask_fill_contour : there's no self.current_maskimg")
            return
        palate = np.zeros((self.current_maskimg.shape[0], self.current_maskimg.shape[1], 3), dtype=np.uint8)
        contours, hierachy = cv2.findContours(self.current_maskimg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        
        #### not work well because of contours[0]. it should be [contours[0]]
        # cv2.drawContours(palate, contours[0],-1,color = (255,255,255), thickness=cv2.FILLED)
        
        #### work well
        # cv2.drawContours(palate, [contours[0]], 0, color = (255,255,255), thickness=cv2.FILLED)
        # cv2.drawContours(palate, [contours[0]], -1, color = (255,255,255), thickness=cv2.FILLED)
        cv2.drawContours(palate, contours, -1, color = (255,255,255), thickness=cv2.FILLED)
        new_mask = np.zeros_like(self.current_maskimg)
        # print(palate==(255))
        # print((palate==(255)).shape)
        # print(np.sum((palate==(255))))
        new_mask[(palate == (255))[:,:,0]] = 255
        self.current_maskimg = new_mask
        self.set_relpace_list_of_mask_item(new_mask)
        # self.set_current_maskimg()
        self.set_info3_main_canvas2()        
        # cv2.imshow('draw_contour', palate)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        

    def help_make_resized_img(self, img_palate, modify_rate, mask=None):
        img_palate = img_palate.copy()
        add_value = 60

        #### change red colors when the mask exist. otherwise just make resized img
        if isinstance(mask, numpy.ndarray):
            if np.sum(mask) != 0:
                indices = np.where(mask == 255)
                red_axis = np.array([2] * len(indices[0]))
                l_colors_red = img_palate[(*indices,red_axis)]
                # l_colors_red = img_palate[indices][:,2]
                l_new_reds = []
                for red in l_colors_red:
                    # red = min(255,red+add_value)
                    red = red+add_value
                    l_new_reds.append(red)
                ### not working in this way
                ### img_palate[indices][:,2] = l_new_reds
                img_palate[(*indices,red_axis)] = l_new_reds

            # l_colors = img_palate[indices]
            # l_new_colors = []
            # for color in l_colors:
            #     b,g,r = color
            #     r = min(255,r+add_value)
            #     l_new_colors.append(np.array([b,g,r]))

            # img_palate[indices] = l_new_colors

        img_modified = np.zeros((img_palate.shape[0]*modify_rate, img_palate.shape[1]*modify_rate, img_palate.shape[2]), dtype=np.uint8)
        for y in range(img_palate.shape[0]):
            for x in range(img_palate.shape[1]):
                img_modified[y*modify_rate:y*modify_rate+modify_rate, x*modify_rate:x*modify_rate+modify_rate] = img_palate[y, x]

        return img_modified

    def help_change_resized_view_color_and_mask(self, img_mask, img_palate, modify_rate, direction='PLUS', point=None, kernel_size=1):
        if direction == 'PLUS':
            # r += add_value
            # r = min(r+add_value, 255)
            add_value = 60
        if direction == 'MINUS':
            # r = max(r-add_value,0)
            # r -= add_value
            add_value = -60

        kernel_size = kernel_size
        
        if not isinstance(img_mask,numpy.ndarray):
            QMessageBox.information(None, 'QMessageBox',f'from help_change_resized_view_color_and_mask : modify_rate({modify_rate}), img_mask is not numpy.ndarray')
            return
        if not isinstance(img_palate,numpy.ndarray):
            QMessageBox.information(None, 'QMessageBox',f'from help_change_resized_view_color_and_mask : modify_rate({modify_rate}), img_palate is not numpy.ndarray')
            return
        if np.ndim(img_palate) != 3:
            QMessageBox.information(None, 'QMessageBox',f'from help_change_resized_view_color_and_mask : modify_rate({modify_rate}), img_palate ndim is not 3')
            return
        if modify_rate < 0 or modify_rate != (img_palate.shape[0]//img_mask.shape[0]) or modify_rate != (img_palate.shape[1]//img_mask.shape[1]):
            QMessageBox.information(None, 'QMessageBox',f'from help_change_resized_view_color_and_mask : modify_rate({modify_rate}) wrong with (img_palate.shape[0]//img_mask.shape[0]) {img_palate.shape[0]//img_mask.shape[0]}')
            return

        ## change whole area color with img_mask != 0 and with area * modify_rate 
        if point == None:
            if np.count_nonzero(img_mask) == 0:
                return img_palate
            indices = np.where(img_mask != 0)
            for i in range(len(indices[0])):
                y = indices[0][i]
                x = indices[1][i]
                for resize_y in range(y*modify_rate,y*modify_rate+modify_rate):
                    for resize_x in range(x*modify_rate,x*modify_rate+modify_rate):
                        r = img_palate[resize_y, resize_x, 2]
                        r += 60
                        img_palate[resize_y, resize_x, 2] = r
            return img_palate 

        ## change color on paticular point
        else:
            try:
                x,y = point
            except Exception as e:
                QMessageBox.information(None, 'QMessageBox',f'from help_change_resized_view_color_and_mask : modify_rate({modify_rate}) wrong with point ({point})')
            
            l_points = []
            if direction == 'PLUS':
                # if img_mask[y, x] == 255:
                    # return img_palate 
                # else:
                for h in range(max(0,y-kernel_size//2),min(img_mask.shape[0],y+kernel_size//2+1)):
                    for w in range(max(0,x-kernel_size//2),min(img_mask.shape[1],x+kernel_size//2+1)):
                        if img_mask[h, w] != 255:
                            img_mask[h, w] = 255
                            l_points.append([h,w])

            elif direction == 'MINUS':
                for h in range(max(0,y-kernel_size//2),min(img_mask.shape[0],y+kernel_size//2+1)):
                    for w in range(max(0,x-kernel_size//2),min(img_mask.shape[1],x+kernel_size//2+1)):
                        if img_mask[h, w] != 0:
                            img_mask[h, w] = 0
                            l_points.append([h,w])
            
            for temp_h, temp_w in l_points:
                y = temp_h
                x = temp_w
                for resize_y in range(y*modify_rate,y*modify_rate+modify_rate):
                        for resize_x in range(x*modify_rate,x*modify_rate+modify_rate):
                            r = img_palate[resize_y, resize_x, 2]
                            r += add_value
                            img_palate[resize_y, resize_x, 2] = r
            return img_palate          


    def help_cb_draw_ellipse_with_points_4x(self, event, x, y, flags, param):
        modify_rate = 4
        root_x = int(x//modify_rate)
        root_y = int(y//modify_rate)
        if event == cv2.EVENT_LBUTTONUP:
            self.flag_lmouse_down = False
        if event == cv2.EVENT_RBUTTONUP:
            self.flag_rmouse_down = False

        if event == cv2.EVENT_LBUTTONDOWN:
            self.flag_lmouse_down = True
            param = self.help_change_resized_view_color_and_mask(self.MASK,param,modify_rate=modify_rate,direction='PLUS',point=(root_x,root_y))

        if event == cv2.EVENT_RBUTTONDOWN:
            self.flag_rmouse_down = True
            param = self.help_change_resized_view_color_and_mask(self.MASK,param,modify_rate=modify_rate,direction='MINUS',point=(root_x,root_y))

        if event == cv2.EVENT_MOUSEMOVE:
            if self.flag_lmouse_down:
                param = self.help_change_resized_view_color_and_mask(self.MASK,param,modify_rate=modify_rate,direction='PLUS',point=(root_x,root_y))

            if self.flag_rmouse_down:
                param = self.help_change_resized_view_color_and_mask(self.MASK,param,modify_rate=modify_rate,direction='MINUS',point=(root_x,root_y))


    def help_cb_draw_ellipse_with_points_8x(self, event, x, y, flags, param):
        modify_rate = 8
        root_x = int(x//modify_rate)
        root_y = int(y//modify_rate)
        if event == cv2.EVENT_LBUTTONUP:
            self.flag_lmouse_down = False
        if event == cv2.EVENT_RBUTTONUP:
            self.flag_rmouse_down = False

        if event == cv2.EVENT_LBUTTONDOWN:
            self.flag_lmouse_down = True
            param = self.help_change_resized_view_color_and_mask(self.MASK,param,modify_rate=modify_rate,direction='PLUS',point=(root_x,root_y))

        if event == cv2.EVENT_RBUTTONDOWN:
            self.flag_rmouse_down = True
            param = self.help_change_resized_view_color_and_mask(self.MASK,param,modify_rate=modify_rate,direction='MINUS',point=(root_x,root_y))

        if event == cv2.EVENT_MOUSEMOVE:
            if self.flag_lmouse_down:
                param = self.help_change_resized_view_color_and_mask(self.MASK,param,modify_rate=modify_rate,direction='PLUS',point=(root_x,root_y))

            if self.flag_rmouse_down:
                param = self.help_change_resized_view_color_and_mask(self.MASK,param,modify_rate=modify_rate,direction='MINUS',point=(root_x,root_y))

    def cb_modify_roi_1x(self, event, x, y, flags, param):
        modify_rate = 1
        root_x = int(x//modify_rate)
        root_y = int(y//modify_rate)        
        
        if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
            self.flag_kernel_size += 2
            if self.flag_kernel_size > 7:
                self.flag_kernel_size = 7
        elif event == cv2.EVENT_RBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
            self.flag_kernel_size -= 2
            if self.flag_kernel_size < 1:
                self.flag_kernel_size = 1

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.flag_lmouse_down = True
            param = self.help_change_resized_view_color_and_mask(self.current_maskimg, param, modify_rate, direction='PLUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)

        elif event == cv2.EVENT_LBUTTONUP:
            self.flag_lmouse_down = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.flag_rmouse_down = True
            param = self.help_change_resized_view_color_and_mask(self.current_maskimg, param, modify_rate, direction='MINUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)

        elif event == cv2.EVENT_RBUTTONUP:
            self.flag_rmouse_down = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.flag_lmouse_down == True:
                param = self.help_change_resized_view_color_and_mask(self.current_maskimg, param, modify_rate, direction='PLUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)
            
            if self.flag_rmouse_down == True:
                param = self.help_change_resized_view_color_and_mask(self.current_maskimg, param, modify_rate, direction='MINUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)


    def cb_info3_btn_edit_mask_1x(self):
        modify_rate = 1
        self.cb_info2_btn_get_mask1()
        base_npimg = self.get_current_base_npimg()
        if not isinstance(base_npimg, numpy.ndarray):
            QMessageBox.information(None, 'QMessageBox','from cb_info3_btn_edit_mask_1x : None current_base_npimg')
            return
        if not isinstance(self.current_maskimg, numpy.ndarray):
            QMessageBox.information(None, 'QMessageBox','from cb_info3_btn_edit_mask_1x : None current_maskimg')
            return
        # if np.sum(self.current_maskimg) == 0 :
        #     QMessageBox.information(None, 'QMessageBox','from cb_info3_btn_edit_mask_1x : empty current_maskimg')
        #     return

        base_npimg = base_npimg.copy()
        
        view_npimg = self.help_make_resized_img(base_npimg, modify_rate,mask=self.current_maskimg)
    
        cv2.namedWindow(f'{modify_rate}x')
        cv2.setMouseCallback(f'{modify_rate}x', self.cb_modify_roi_1x, view_npimg)
        while True:

            view_with_npimg = view_npimg.copy()
            view_with_npimg = cv2.putText(view_with_npimg,f"{self.flag_kernel_size}",(5,15),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0))
            cv2.imshow(f'{modify_rate}x',view_with_npimg)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break

        self.set_info3_main_canvas2()


    def cb_modify_roi_2x(self, event, x, y, flags, param):
        modify_rate = 2
        root_x = int(x//modify_rate)
        root_y = int(y//modify_rate)

        if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
            self.flag_kernel_size += 2
            if self.flag_kernel_size > 7:
                self.flag_kernel_size = 7
        elif event == cv2.EVENT_RBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
            self.flag_kernel_size -= 2
            if self.flag_kernel_size < 1:
                self.flag_kernel_size = 1

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.flag_lmouse_down = True
            param = self.help_change_resized_view_color_and_mask(self.current_maskimg, param, modify_rate, direction='PLUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)

        elif event == cv2.EVENT_LBUTTONUP:
            self.flag_lmouse_down = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.flag_rmouse_down = True
            param = self.help_change_resized_view_color_and_mask(self.current_maskimg, param, modify_rate, direction='MINUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)

        elif event == cv2.EVENT_RBUTTONUP:
            self.flag_rmouse_down = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.flag_lmouse_down == True:
                param = self.help_change_resized_view_color_and_mask(self.current_maskimg, param, modify_rate, direction='PLUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)
            
            if self.flag_rmouse_down == True:
                param = self.help_change_resized_view_color_and_mask(self.current_maskimg, param, modify_rate, direction='MINUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)


    def cb_info3_btn_edit_mask_2x(self):

        modify_rate = 2
        self.cb_info2_btn_get_mask1()
        base_npimg = self.get_current_base_npimg()
        if not isinstance(base_npimg, numpy.ndarray):
            QMessageBox.information(None, 'QMessageBox','from cb_info3_btn_edit_mask_2x : None current_base_npimg')
            return
        if not isinstance(self.current_maskimg, numpy.ndarray):
            QMessageBox.information(None, 'QMessageBox','from cb_info3_btn_edit_mask_2x : None current_maskimg')
            return
        # if np.sum(self.current_maskimg) == 0 :
        #     QMessageBox.information(None, 'QMessageBox','from cb_info3_btn_edit_mask_2x : empty current_maskimg')
        #     return

        base_npimg = base_npimg.copy()
        view_npimg = self.help_make_resized_img(base_npimg, modify_rate, mask=self.current_maskimg)
    
        cv2.namedWindow(f'{modify_rate}x')
        cv2.setMouseCallback(f'{modify_rate}x', self.cb_modify_roi_2x, view_npimg)
        while True:

            view_with_npimg = view_npimg.copy()
            view_with_npimg = cv2.putText(view_with_npimg,f"{self.flag_kernel_size}",(5,15),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0))
            cv2.imshow(f'{modify_rate}x',view_with_npimg)
            # if cv2.waitKey(0) & 0xFF == 27:
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break
        self.set_info3_main_canvas2()
            
    
    def cb_modify_roi_4x(self, event, x, y, flags, param):
        modify_rate = 4
        root_x = int(x//modify_rate)
        root_y = int(y//modify_rate)

        if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
            self.flag_kernel_size += 2
            if self.flag_kernel_size > 7:
                self.flag_kernel_size = 7
        elif event == cv2.EVENT_RBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
            self.flag_kernel_size -= 2
            if self.flag_kernel_size < 1:
                self.flag_kernel_size = 1

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.flag_lmouse_down = True
            param = self.help_change_resized_view_color_and_mask(self.current_maskimg, param, modify_rate, direction='PLUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)

        elif event == cv2.EVENT_LBUTTONUP:
            self.flag_lmouse_down = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.flag_rmouse_down = True
            param = self.help_change_resized_view_color_and_mask(self.current_maskimg, param, modify_rate, direction='MINUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)

        elif event == cv2.EVENT_RBUTTONUP:
            self.flag_rmouse_down = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.flag_lmouse_down == True:
                param = self.help_change_resized_view_color_and_mask(self.current_maskimg, param, modify_rate, direction='PLUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)

            if self.flag_rmouse_down == True:
                param = self.help_change_resized_view_color_and_mask(self.current_maskimg, param, modify_rate, direction='MINUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)



    def cb_info3_btn_edit_mask_4x(self):

        modify_rate = 4
        self.cb_info2_btn_get_mask1()
        base_npimg = self.get_current_base_npimg()
        if not isinstance(base_npimg, numpy.ndarray):
            QMessageBox.information(None, 'QMessageBox','from cb_info3_btn_edit_mask_4x : None current_base_npimg')
            return
        if not isinstance(self.current_maskimg, numpy.ndarray):
            QMessageBox.information(None, 'QMessageBox','from cb_info3_btn_edit_mask_4x : None current_maskimg')
            return
        

        base_npimg = base_npimg.copy()
        view_npimg = self.help_make_resized_img(base_npimg, modify_rate, mask=self.current_maskimg)

        # if np.sum(self.current_maskimg) != 0:
        #     indices = np.where(self.current_maskimg == 255)
        #     l_color = base_npimg[indices]
        #     new_color = []
        #     for color in l_color:
        #         b,g,r = color
        #         r = min(255,r+add_value)
        #         new_color.append(np.array([b,g,r]))
        #     base_npimg[indices] = new_color

        # view_npimg = np.zeros((base_npimg.shape[0] * modify_rate, base_npimg.shape[1] * modify_rate, base_npimg.shape[2]), dtype=np.uint8)
        # for h in range(base_npimg.shape[0]):
        #     for w in range(base_npimg.shape[1]):
        #         b_g_r = base_npimg[h][w]

        #         for x2_h in range(h * modify_rate,h * modify_rate + modify_rate):
        #             for x2_w in range(w * modify_rate,w * modify_rate + modify_rate):
        #                 view_npimg[x2_h][x2_w] = b_g_r
        
    
        cv2.namedWindow(f'{modify_rate}x')
        cv2.setMouseCallback(f'{modify_rate}x', self.cb_modify_roi_4x, view_npimg)
        while True:
            view_with_npimg = view_npimg.copy()
            view_with_npimg = cv2.putText(view_with_npimg,f"{self.flag_kernel_size}",(5,15),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0))
            cv2.imshow(f'{modify_rate}x',view_with_npimg)
            # if cv2.waitKey(0) & 0xFF == 27:
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break
        self.set_info3_main_canvas2()


    def cb_modify_roi_8x(self, event, x, y, flags, param):
        modify_rate = 8
        root_x = int(x//modify_rate)
        root_y = int(y//modify_rate)

        if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
            self.flag_kernel_size += 2
            if self.flag_kernel_size > 7:
                self.flag_kernel_size = 7
        elif event == cv2.EVENT_RBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
            self.flag_kernel_size -= 2
            if self.flag_kernel_size < 1:
                self.flag_kernel_size = 1

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.flag_lmouse_down = True
            param = self.help_change_resized_view_color_and_mask(self.resize_maskimg, param, modify_rate, direction='PLUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)

        elif event == cv2.EVENT_LBUTTONUP:
            self.flag_lmouse_down = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.flag_rmouse_down = True
            param = self.help_change_resized_view_color_and_mask(self.resize_maskimg, param, modify_rate, direction='MINUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)

        elif event == cv2.EVENT_RBUTTONUP:
            self.flag_rmouse_down = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.flag_lmouse_down == True:
                param = self.help_change_resized_view_color_and_mask(self.resize_maskimg, param, modify_rate, direction='PLUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)
            
            if self.flag_rmouse_down == True:
                param = self.help_change_resized_view_color_and_mask(self.resize_maskimg, param, modify_rate, direction='MINUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)


    def cb_info3_btn_edit_mask_8x(self):
        modify_rate = 8
        self.cb_info2_btn_get_mask1()
        base_npimg = self.get_current_base_npimg()
        if not isinstance(base_npimg, numpy.ndarray):
            QMessageBox.information(None, 'QMessageBox','from cb_info3_btn_edit_mask_8x : None current_base_npimg')
            return
        if not isinstance(self.current_maskimg, numpy.ndarray):
            QMessageBox.information(None, 'QMessageBox','from cb_info3_btn_edit_mask_8x : None current_maskimg')
            return
        if np.sum(self.current_maskimg) == 0 :
            QMessageBox.information(None, 'QMessageBox','from cb_info3_btn_edit_mask_8x : empty current_maskimg')
            return

        base_npimg = base_npimg.copy()

        list_ = self.get_island_img_with_info(self.current_maskimg)
        _, max_x, max_y, max_w, max_h, max_area = list_

        view_x0 = max(0, max_x - 20)
        view_y0 = max(0, max_y - 20)
        view_x1 = min(self.current_maskimg.shape[1], max_x + max_w + 20)
        view_y1 = min(self.current_maskimg.shape[0], max_y + max_h + 20)

        self.resize_maskimg = self.current_maskimg[view_y0:view_y1, view_x0:view_x1]
        self.resize_palate = base_npimg[view_y0:view_y1, view_x0:view_x1]

        view_npimg = self.help_make_resized_img(self.resize_palate, modify_rate, mask=self.resize_maskimg)
    
        cv2.namedWindow(f'{modify_rate}x')
        cv2.setMouseCallback(f'{modify_rate}x', self.cb_modify_roi_8x, view_npimg)
        while True:
            view_with_npimg = view_npimg.copy()
            view_with_npimg = cv2.putText(view_with_npimg,f"{self.flag_kernel_size}",(5,15),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0))
            cv2.imshow(f'{modify_rate}x',view_with_npimg)
            # if cv2.waitKey(0) & 0xFF == 27:
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break

        self.current_maskimg[view_y0:view_y1, view_x0:view_x1] = self.resize_maskimg
        self.set_info3_main_canvas2()


    def cb_modify_roi_16x(self, event, x, y, flags, param):
        modify_rate = 16
        root_x = int(x//modify_rate)
        root_y = int(y//modify_rate)        
        
        if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
            self.flag_kernel_size += 2
            if self.flag_kernel_size > 7:
                self.flag_kernel_size = 7
        elif event == cv2.EVENT_RBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
            self.flag_kernel_size -= 2
            if self.flag_kernel_size < 1:
                self.flag_kernel_size = 1

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.flag_lmouse_down = True

            param = self.help_change_resized_view_color_and_mask(self.resize_maskimg, param, modify_rate, direction='PLUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)

            # if self.resize_maskimg[root_y][root_x] != 255:
            #     self.resize_maskimg[root_y][root_x] = 255
            #     b_g_r = param[y][x]
            #     b_g_r += np.array([0,0,60], dtype=np.uint8)
            #     for new_y in range(root_y * modify_rate, root_y * modify_rate + modify_rate):
            #         for new_x in range(root_x * modify_rate, root_x * modify_rate + modify_rate):
            #             param[new_y][new_x] = b_g_r

        elif event == cv2.EVENT_LBUTTONUP:
            self.flag_lmouse_down = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.flag_rmouse_down = True

            
            param = self.help_change_resized_view_color_and_mask(self.resize_maskimg, param, modify_rate, direction='MINUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)
            
            # if self.resize_maskimg[root_y][root_x] != 0:
            #     self.resize_maskimg[root_y][root_x] = 0
            #     b_g_r = param[y][x]
            #     b_g_r -= np.array([0,0,60], dtype=np.uint8)
            #     for new_y in range(root_y * modify_rate, root_y * modify_rate + modify_rate):
            #         for new_x in range(root_x * modify_rate, root_x * modify_rate + modify_rate):
            #             param[new_y][new_x] = b_g_r

        elif event == cv2.EVENT_RBUTTONUP:
            self.flag_rmouse_down = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.flag_lmouse_down == True:
                            
                param = self.help_change_resized_view_color_and_mask(self.resize_maskimg, param, modify_rate, direction='PLUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)

                # if self.resize_maskimg[root_y][root_x] != 255:
                #     self.resize_maskimg[root_y][root_x] = 255
                #     b_g_r = param[y][x]
                #     b_g_r += np.array([0,0,60], dtype=np.uint8)
                #     for new_y in range(root_y * modify_rate, root_y * modify_rate + modify_rate):
                #         for new_x in range(root_x * modify_rate, root_x * modify_rate + modify_rate):
                #             param[new_y][new_x] = b_g_r
            
            if self.flag_rmouse_down == True:
                param = self.help_change_resized_view_color_and_mask(self.resize_maskimg, param, modify_rate, direction='MINUS', point=(root_x,root_y), kernel_size=self.flag_kernel_size)


                # if self.resize_maskimg[root_y][root_x] != 0:
                #     self.resize_maskimg[root_y][root_x] = 0
                #     b_g_r = param[y][x]
                #     b_g_r -= np.array([0,0,60], dtype=np.uint8)
                #     for new_y in range(root_y * modify_rate, root_y * modify_rate + modify_rate):
                #         for new_x in range(root_x * modify_rate, root_x * modify_rate + modify_rate):
                #             param[new_y][new_x] = b_g_r


    def cb_info3_btn_edit_mask_16x(self):
        modify_rate = 16
        self.cb_info2_btn_get_mask1()
        base_npimg = self.get_current_base_npimg()
        if not isinstance(base_npimg, numpy.ndarray):
            QMessageBox.information(None, 'QMessageBox','from cb_info3_btn_edit_mask_16x : None current_base_npimg')
            return
        if not isinstance(self.current_maskimg, numpy.ndarray):
            QMessageBox.information(None, 'QMessageBox','from cb_info3_btn_edit_mask_16x : None current_maskimg')
            return
        if np.sum(self.current_maskimg) == 0 :
            QMessageBox.information(None, 'QMessageBox','from cb_info3_btn_edit_mask_16x : empty current_maskimg')
            return

        base_npimg = base_npimg.copy()
        
        ################# get a crop area for resize 16x #################
        # retval, labelled, stats, centroids = cv2.connectedComponentsWithStats(self.current_maskimg)
        # max_area = 0
        # mas_label = 0
        # max_x = 0
        # max_y = 0
        # max_w = 0
        # max_h = 0

        # for index, stat in enumerate(stats[1:]):
        #     x,y,w,h,area = stat
        #     if area > max_area:
        #         max_area = area
        #         max_label = index+1
        #         max_x = x
        #         max_y = y
        #         max_w = w
        #         max_h = h
        
        # if max_area == 0:
        #     return

        list_ = self.get_island_img_with_info(self.current_maskimg)
        _, max_x, max_y, max_w, max_h, max_area = list_

        view_x0 = max(0, max_x - 10)
        view_y0 = max(0, max_y - 10)
        view_x1 = min(self.current_maskimg.shape[1], max_x + max_w + 10)
        view_y1 = min(self.current_maskimg.shape[0], max_y + max_h + 10)

        self.resize_maskimg = self.current_maskimg[view_y0:view_y1, view_x0:view_x1]
        self.resize_palate = base_npimg[view_y0:view_y1, view_x0:view_x1]

        view_npimg = self.help_make_resized_img(self.resize_palate, modify_rate,mask=self.resize_maskimg)
        # indices = np.where(self.resize_maskimg == 255)
        # l_color = self.resize_palate[indices]
        # new_color = []
        # for color in l_color:
        #     b,g,r = color
        #     r = min(255,r+60)
        #     new_color.append(np.array([b,g,r]))
        # self.resize_palate[indices] = new_color

        # view_npimg = np.zeros((self.resize_palate.shape[0] * modify_rate, self.resize_palate.shape[1] * modify_rate, self.resize_palate.shape[2]), dtype=np.uint8)
        # for h in range(self.resize_palate.shape[0]):
        #     for w in range(self.resize_palate.shape[1]):
        #         b_g_r = self.resize_palate[h][w]

        #         for x2_h in range(h * modify_rate,h * modify_rate + modify_rate):
        #             for x2_w in range(w * modify_rate,w * modify_rate + modify_rate):
        #                 view_npimg[x2_h][x2_w] = b_g_r
        
    
        cv2.namedWindow(f'{modify_rate}x')
        cv2.setMouseCallback(f'{modify_rate}x', self.cb_modify_roi_16x, view_npimg)
        while True:
            view_with_npimg = view_npimg.copy()
            view_with_npimg = cv2.putText(view_with_npimg,f"{self.flag_kernel_size}",(5,15),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0))
            cv2.imshow(f'{modify_rate}x',view_with_npimg)
            # if cv2.waitKey(0) & 0xFF == 27:
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break

        self.current_maskimg[view_y0:view_y1, view_x0:view_x1] = self.resize_maskimg
        self.set_info3_main_canvas2()


    def cb_info3_btn_load_current_mask(self):
        file = f'{self.SAVE_PATH}/{self.current_location}/{self.l_save_indices[self.current_index]}_mask.png'
        if os.path.isfile(file):
            maskimg = cv2.imread(file,0)
            if isinstance(maskimg,numpy.ndarray):
                if maskimg.shape[0] != self.MODEL_HEIGHT or maskimg.shape[1] != self.MODEL_WIDTH:
                    QMessageBox.information(None, 'QMessageBox', f"from cb_info3_btn_load_current_mask: wrong shape. you loaded - {maskimg.shape}, expected - ({MODEL_HEIGHT},{MODEL_WIDTH}) ")
                    return
                self.current_maskimg = maskimg
                self.set_info3_main_canvas2()
        else:
            QMessageBox.information(None, 'QMessageBox', f"from cb_info3_btn_load_current_mask: file not exist")
            return

    def cb_info3_btn_load_mask_file(self):
        dir_ = f'{self.SAVE_PATH}/{self.current_location}'
        if os.path.isdir(dir_):
            fname = QFileDialog.getOpenFileName(None,'Open file',dir_)
            if fname[0]:
                file_path, filename_with_ext = os.path.split(fname[0])
                file_name, file_ext = os.path.splitext(filename_with_ext)
                if file_ext != '.png':
                    QMessageBox.information(None, 'QMessageBox', "from cb_info3_btn_load_mask_file: not png file!")
                    return

                maskimg = cv2.imread(fname[0],0)
                if maskimg.shape[0] != self.MODEL_HEIGHT or maskimg.shape[1] != self.MODEL_WIDTH:
                    QMessageBox.information(None, 'QMessageBox', f"from cb_info3_btn_load_mask_file: wrong shape. you loaded - {maskimg.shape}, expected - ({MODEL_HEIGHT},{MODEL_WIDTH}) ")
                    return

                self.current_maskimg = maskimg
                self.set_info3_main_canvas2()


    def cb_info3_btn_save_temp_mask(self):
        if not isinstance(self.current_maskimg,numpy.ndarray):
            QMessageBox.information(None, 'QMessageBox', 'from cb_info3_btn_save_temp_mask : None self.current_maskimg')
        n = self.info3_vlist_maskimgs.count()
        dist = cv2.distanceTransform(self.current_maskimg,cv2.DIST_L2,3)
        # max_ = dist.max()
        # if max_ == 0:
        #     return
        # argmax = np.where(dist == max_)
        # item = QListWidgetItem(f'{n}, x:{argmax[1]}, y:{argmax[0]}, dt_max:{round(float(max_),2)}')

        indices = np.where(self.current_maskimg == 255)
        center_x = int(np.sum(indices[1])/len(indices[1]))
        center_y = int(np.sum(indices[0])/len(indices[0]))
        item = QListWidgetItem(f'{n}, x:{center_x}, y:{center_y}, dt_center:{round(float(dist[center_y,center_x]),2)}')
        item.setData(Qt.ItemDataRole.UserRole, self.current_maskimg)
        self.info3_vlist_maskimgs.addItem(item)


    def cb_info3_btn_save_file_mask(self):
        base_img = self.get_current_base_npimg()
        if not isinstance(base_img,numpy.ndarray):
            QMessageBox.information(None, "QMessageBox", "from cb_info3_btn_save_file_mask: wrong instance of base_img")
            return
        if not isinstance(self.current_maskimg,numpy.ndarray):
            QMessageBox.information(None, "QMessageBox", "from cb_info3_btn_save_file_mask: wrong instance of current_maskimg")
            return
        
        self.cb_info3_btn_edit_mask_island()
        
        if not os.path.isdir(f'{self.SAVE_PATH}/{self.current_location}'):
            QMessageBox.information(None, "QMessageBox", f"from cb_info3_btn_save_file_mask: no such directory ({self.SAVE_PATH}/{self.current_location}) \n you should check it's capital letter of the folder name")
            return

        # cv2.imwrite(f'{self.SAVE_PATH}/{self.current_location}/{self.l_save_indices[self.current_index]}_orig.png',base_img) 
        # cv2.imwrite(f'{self.SAVE_PATH}/{self.current_location}/{self.l_save_indices[self.current_index]}_mask.png',self.current_maskimg) 

        try:
            cv2.imwrite(f'{self.SAVE_PATH}/{self.current_location}/{self.l_save_indices[self.current_index]}_orig.png',base_img) 
            cv2.imwrite(f'{self.SAVE_PATH}/{self.current_location}/{self.l_save_indices[self.current_index]}_mask.png',self.current_maskimg) 
        except Exception as e:
            QMessageBox.information(None, "QMessageBox", "from cb_info3_btn_save_file_mask: {e}")

    def cb_info3_vlist_maskimgs(self,listwidget_item):
        self.current_maskimg = listwidget_item.data(Qt.ItemDataRole.UserRole)
        self.set_info3_main_canvas2()


    def cb_info3_btn_erase_vlist_maskimgs(self):
        current_item_row = self.info3_vlist_maskimgs.currentRow()
        current_item = self.info3_vlist_maskimgs.takeItem(current_item_row)
        self.info3_vlist_maskimgs.removeItemWidget(current_item)

        n = self.info3_vlist_maskimgs.count()
        new_list = []
        for i in range(n):
            item = self.info3_vlist_maskimgs.item(i)
            l_ = item.text().split(',')
            l_.pop(0)
            l_.insert(0,str(i))
            str_ = ','.join(l_)
            new_list.append(str_)

        self.info3_vlist_maskimgs.clear()
        for i in range(n):
            self.info3_vlist_maskimgs.addItem(new_list[i])


    def cb_info3_btn_clear_vlist_maskimgs(self):
        self.info3_vlist_maskimgs.clear()

