import sys, time, cv2, csv, os, re
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
import numpy as np
from my_feature_extractor import FeatureExtractor
import matplotlib.pyplot as plt 
import matplotlib.animation
from typing import Union

class MyObserve:
    def __init__(self,value):
        self.value = value
        self._connect_functions=[]
        
    def connect(self, func):
        self._connect_functions.append(func)
    
    def set(self, new_value):
        if isinstance(new_value, list) or isinstance(new_value,tuple) or isinstance(new_value,np.ndarray):
            new_value = new_value.copy()
        self.value = new_value
        for func in self._connect_functions:
            func(self.value)


class MySeqObs1(MyObserve):
    def __init__(self,values:np.ndarray,edge_indices:np.ndarray, erased_edge_indices:np.ndarray):
        super().__init__(value=values)
        
        self.fe = FeatureExtractor()
        self.edge_indices = edge_indices
        self.erased_edge_indices = erased_edge_indices
        self.edge_info_dict = {}
        self.edge_info_distances1 = np.array([])
        self.edge_info_distances2 = np.array([])
        self.edge_info_x_diffs = np.array([])
        self.edge_info_y_diffs = np.array([])
        self.edge_info_y_abs_diffs = np.array([])
        self.edge_info_grads = np.array([])
        self.edge_info_grad_ratios = np.array([])
        self.edge_info_manual_nystagmus_indices = np.array([])
        self.edge_info_statistical_nystagmus_indices = np.array([])
        self.connect(self.update_edge_infos)
    
    def set(self, new_values:np.ndarray, new_edge_indices:np.ndarray, new_erased_edge_indices:np.ndarray):
        self.value = new_values.copy()
        self.edge_indices = new_edge_indices.copy()
        self.erased_edge_indices = new_erased_edge_indices.copy()
        
        for func in self._connect_functions:
            func(data=self.value, edge_indices=self.edge_indices, erased_edge_indices=self.erased_edge_indices, edge_info_dict=self.edge_info_dict)
            
    def update_edge_infos(self, **kargs):
        value = kargs['data']
        edge_indices = kargs['edge_indices']
        erased_edge_indices = kargs['erased_edge_indices']
        
        self.edge_info_dict = self.fe.get_gradients_infos(value,edge_indices)
        self.edge_info_distances1 = self.edge_info_dict['distances1']
        self.edge_info_distances2 = self.edge_info_dict['distances2']
        self.edge_info_x_diffs = self.edge_info_dict['x_diffs']
        self.edge_info_y_diffs = self.edge_info_dict['y_diffs']      # y_differences within edge_indices
        self.edge_info_y_abs_diffs = self.edge_info_dict['y_abs_diffs']
        self.edge_info_grads = self.edge_info_dict['grads']
        self.edge_info_grad_ratios = self.edge_info_dict['grad_ratios']   # the sequence of (grad / np.square(next_grad))
        self.edge_info_manual_nystagmus_indices = self.fe.get_manual_nystagmus_indices(value,edge_indices,info_dict=self.edge_info_dict)
        self.edge_info_statistical_nystagmus_indices = self.fe.get_statistical_nystagmus_indices(value, edge_indices, info_dict=self.edge_info_dict)


class SubGui1(QWidget):
# class SubGui1(QMainWindow):
    def __init__(self, data_lt, data_rt, d_inferred_info_lt, d_inferred_info_rt, parent):
        super(SubGui1,self).__init__()
        # self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        # self.destroyed.connect(parent.on_child_destroyed)
        self.parent_ = parent
        self.BASE_PATH = self.parent_.FILE_PATH
        self.CURRENT_FILE_FULL_PATH = self.parent_.CURRENT_FILE_FULL_PATH
        self.rgb_npimgs_lt = data_lt
        self.rgb_npimgs_rt = data_rt

        self.d_inferred_info_lt = d_inferred_info_lt
        self.d_inferred_info_rt = d_inferred_info_rt
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('index extractor')
        
        self.vlayout1 = QVBoxLayout()

        # self.vlayout1.addWidget(self.info1)
        self.hlayout1 = QHBoxLayout()
        self.rt_gui_module = MyGuiModule(self.rgb_npimgs_rt, self.d_inferred_info_rt, self)
        self.rt_gui_module.set_name("Rt")
        self.lt_gui_module = MyGuiModule(self.rgb_npimgs_lt, self.d_inferred_info_lt, self)
        self.lt_gui_module.set_name("Lt")

        self.rt_gui_module.signal_space.connect(self.active_both_module)
        self.rt_gui_module.signal_space.connect(self.lt_gui_module.run_frames)
        self.lt_gui_module.signal_space.connect(self.active_both_module)
        self.lt_gui_module.signal_space.connect(self.rt_gui_module.run_frames)

        self.hlayout1.addWidget(self.rt_gui_module)
        self.hlayout1.addWidget(self.lt_gui_module)
        self.vlayout1.addLayout(self.hlayout1)
        self.vlayout1.addWidget(QLabel())
        
        self.hlayout2 = QHBoxLayout()
        for i in range(7):
            self.hlayout2.addWidget(QLabel())
        self.btn_load_file_indices = QPushButton("load_file")
        self.btn_load_file_indices.keyPressEvent = self.myevent_key_press
        self.hlayout2.addWidget(self.btn_load_file_indices)

        self.btn_load_indices = QPushButton("load")
        self.btn_load_indices.keyPressEvent = self.myevent_key_press
        self.hlayout2.addWidget(self.btn_load_indices)

        self.btn_save_indices = QPushButton("save")
        self.btn_save_indices.keyPressEvent = self.myevent_key_press
        self.hlayout2.addWidget(self.btn_save_indices)

        self.btn_load_file_indices.clicked.connect(self.cb_btn_load_file_indices)
        self.btn_load_indices.clicked.connect(self.cb_btn_load_indices)
        self.btn_save_indices.clicked.connect(self.cb_btn_save_indices)
        self.vlayout1.addLayout(self.hlayout2)
        
        self.setLayout(self.vlayout1)

    def set_new_data(self, data_lt, data_rt, d_inferred_info_lt, d_inferred_info_rt, parent):
        self.rgb_npimgs_lt = data_lt
        self.rgb_npimgs_rt = data_rt
        self.d_inferred_info_lt = d_inferred_info_lt
        self.d_inferred_info_rt = d_inferred_info_rt
        self.parent_ = parent
        self.CURRENT_FILE_FULL_PATH = self.parent_.CURRENT_FILE_FULL_PATH
        self.BASE_PATH = self.parent_.FILE_PATH
        self.rt_gui_module.set_new_data(self.rgb_npimgs_rt, self.d_inferred_info_rt, self)
        self.lt_gui_module.set_new_data(self.rgb_npimgs_lt, self.d_inferred_info_lt, self)

    def myevent_key_press(self,e):
        if e.key() == QtCore.Qt.Key_Space:
            modifiers = QApplication.keyboardModifiers()
            event_space = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Space, modifiers)
            # self.lt_gui_module.keyPressEvent(event_space)
            # only one side module should be passed because it's already connected by signal_space
            self.rt_gui_module.keyPressEvent(event_space)

    def active_both_module(self):
        if not self.rt_gui_module.FLAG_LABEL_PUSHED and not self.lt_gui_module.FLAG_LABEL_PUSHED:
            self.lt_gui_module.toggle_label()
            self.rt_gui_module.toggle_label()
            
    def cb_btn_load_file_indices(self):
        # dir_ = f'{self.parent_.SAVE_BASE_PATH}/{self.parent_.FILE_NAME}'
        dir_ = f'{self.parent_.FILE_PATH}'
        if os.path.isdir(dir_):
            fname = QFileDialog.getOpenFileName(None, 'Open File', dir_)
            if fname[0]:
                file_path, filename_with_ext = os.path.split(fname[0])
                file_name, file_ext = os.path.splitext(filename_with_ext)
                if file_ext != '.csv':
                    QMessageBox.information(None, 'QMessageBox', "from cb_btn_load_file_indices: not csv file!")
                    return

                try:
                    with open(f'{fname[0]}','r', newline='', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        self.lt_gui_module.vlist_indices.clear()
                        self.rt_gui_module.vlist_indices.clear()
                        for row in reader:
                            if row[0] == 'Rt':
                                count_ = self.rt_gui_module.vlist_indices.count()
                                self.rt_gui_module.vlist_indices.addItem(f'{count_}, start:{row[1]}, end:{row[2]}')
                                self.rt_gui_module.help_vlist_indices_record()
                            if row[0] == 'Lt':
                                count_ = self.lt_gui_module.vlist_indices.count()
                                self.lt_gui_module.vlist_indices.addItem(f'{count_}, start:{row[1]}, end:{row[2]}')
                                self.lt_gui_module.help_vlist_indices_record()

                except Exception as e:
                    QMessageBox.information(None, "QMessageBox", f" from cb_btn_load_indices : {e}")
                    return

    def cb_btn_load_indices(self):
        n_rt = self.rt_gui_module.vlist_indices.count()
        n_lt = self.lt_gui_module.vlist_indices.count()
        if n_rt != 0 or n_lt != 0:
            reply = QMessageBox.question(self,"warning","the list view already has an item. it will be replaced", QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        base_path = self.BASE_PATH
        file_location1 = f'{base_path}/{self.parent_.FILE_NAME}.csv'
        file_location2 = f'{self.parent_.FILE_NAME}.csv'
        # file_location3 = f'{base_path}/{self.parent_.FILE_NAME}/{self.parent_.FILE_NAME}.csv'
        for i in [file_location1, file_location2]:
            if os.path.exists(i):
                try:
                    with open(i,'r', newline='', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        self.lt_gui_module.vlist_indices.clear()
                        self.rt_gui_module.vlist_indices.clear()
                        for row in reader:
                            if row[0] == 'Rt':
                                count_ = self.rt_gui_module.vlist_indices.count()
                                self.rt_gui_module.vlist_indices.addItem(f'{count_}, start:{row[1]}, end:{row[2]}')
                                self.rt_gui_module.help_vlist_indices_record()
                            if row[0] == 'Lt':
                                count_ = self.lt_gui_module.vlist_indices.count()
                                self.lt_gui_module.vlist_indices.addItem(f'{count_}, start:{row[1]}, end:{row[2]}')
                                self.lt_gui_module.help_vlist_indices_record()
                    return
                except Exception as e:
                    QMessageBox.information(None, "QMessageBox", f" from cb_btn_load_indices : {e}")
                    return
        QMessageBox.information(None, "QMessageBox", f" from cb_btn_load_indices : couldn't find file location {base_path}/{self.parent_.FILE_NAME}.csv")
        return
    
    def cb_btn_save_indices(self):
        l_rt = self.rt_gui_module.get_vlist_items()
        l_lt = self.lt_gui_module.get_vlist_items()
        
        # file_save_path = f'{self.BASE_PATH}/{self.parent_.FILE_NAME}/{self.parent_.FILE_NAME}.csv'
        file_save_path = f'{self.BASE_PATH}/{self.parent_.FILE_NAME}.csv'

        if os.path.isfile(file_save_path):
            reply = QMessageBox.question(self,"warning","the file already exists. it will be replaced", QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        try:
            with open(file_save_path,'w', newline='', encoding='utf-8') as f:
                wr = csv.writer(f)
                for i in l_rt:
                    wr.writerow(["Rt",i[0],i[1]])
                for i in l_lt:
                    wr.writerow(["Lt",i[0],i[1]])
        except Exception as e:
            QMessageBox.information(None, "QMessageBox", f" from cb_btn_save_indices : {e}")
            return


    def closeEvent(self,event):
        # print("subview_OnClose")
        # print("subview_OnDestroy")
        print("subview_closeEvent")
        # self.__del__()

    def __del__(self):
        del self.rt_gui_module
        del self.lt_gui_module
        del self.BASE_PATH
        del self.parent_
        del self.rgb_npimgs_lt
        del self.rgb_npimgs_rt
        del self.d_inferred_info_lt
        del self.d_inferred_info_rt
        print("subview deleted")


        # self.destroy_sub_module_resources()

    # def destroy_sub_module_resources(self):
    #     del self.rt_gui_module
    #     del self.lt_gui_module
    #     del self.vlayout1
    #     del self.parent_
    #     del self.BASE_PATH
    #     del self.d_inferred_info_lt
    #     del self.d_inferred_info_rt


class PltShowPalette(QWidget):
    def __init__(self,parent=None):
        super(PltShowPalette, self).__init__(parent=parent)
        self.parent_ = parent
        self.setWindowTitle('PltShowPalette')
        self.init_ui()
    
    def init_ui(self):
        self.lbl_main_palette1 = QLabel()
        # if self.parent_ != None:
            # self.lbl_main_palette1.setFixedHeight(self.parent_.PLT_SHOW_HEIGHT)
            # self.lbl_main_palette1.setFixedWidth(self.parent_.PLT_SHOW_WIDTH)
        # else:
            # self.lbl_main_palette1.setFixedHeight(1024)
            # self.lbl_main_palette1.setFixedWidth(500)
        vlayout1 = QVBoxLayout()
        vlayout1.addWidget(self.lbl_main_palette1)
        self.setLayout(vlayout1)
    
    def __del__(self):
        del self.parent_

class MyQPushButton(QPushButton):
    signal_mygbox_space = QtCore.pyqtSignal()
    def __init__(self,text:str, parent=None):
        super(MyQPushButton,self).__init__(text,parent=parent)
        self.parent_ = parent
        modifiers = QApplication.keyboardModifiers()
        self.event_space = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Space, modifiers)
    
    def keyPressEvent(self, a0: QtGui.QMouseEvent) -> None:
        char = a0.text()
        if not self.parent_:
            return super(MyQPushButton,self).keyPressEvent(a0)
        # detect space key or not digit
        if a0.key() == QtCore.Qt.Key_Space:
            self.parent_.keyPressEvent(self.event_space)       
            return 

        elif not char.isdigit() and a0.key() != QtCore.Qt.Key_Backspace:
            return

        return super(MyQPushButton,self).keyPressEvent(a0)

    def __del__(self):
        del self.parent_

class MyQRadioButton(QRadioButton):
    signal_mygbox_space = QtCore.pyqtSignal()
    def __init__(self,text:str=None, parent=None):
        super(MyQRadioButton,self).__init__(text,parent=parent)
        self.parent_ = parent
        modifiers = QApplication.keyboardModifiers()
        self.event_space = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Space, modifiers)
    
    def keyPressEvent(self, a0: QtGui.QMouseEvent) -> None:
        char = a0.text()
        if not self.parent_:
            return super(MyQRadioButton,self).keyPressEvent(a0)
        # detect space key or not digit
        if a0.key() == QtCore.Qt.Key_Space:
            self.parent_.keyPressEvent(self.event_space)       
            return 

        elif not char.isdigit() and a0.key() != QtCore.Qt.Key_Backspace:
            return

        return super(MyQRadioButton,self).keyPressEvent(a0)

    def __del__(self):
        del self.parent_


class MyQLineEdit(QLineEdit):
    signal_myledit_space = QtCore.pyqtSignal()
    def __init__(self,parent=None):
        super(MyQLineEdit,self).__init__(parent=parent)
        self.parent_ = parent
        modifiers = QApplication.keyboardModifiers()
        self.event_space = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Space, modifiers)
    
    def keyPressEvent(self, a0: QtGui.QMouseEvent) -> None:
        char = a0.text()
        if not self.parent_:
            return super(MyQLineEdit,self).keyPressEvent(a0)
        # detect space key or not digit
        if a0.key() == QtCore.Qt.Key_Space:
            # self.signal_myledit_space.emit()
            self.parent_.keyPressEvent(self.event_space)
            # self.parent_.run_frames()          

            return 

        elif not char.isdigit() and a0.key() != QtCore.Qt.Key_Backspace:
            return
        # return super().keyPressEvent(a0)
        # super().keyPressEvent(a0)

        return super(MyQLineEdit,self).keyPressEvent(a0)

    def __del__(self):
        del self.parent_


class Mythread(QtCore.QThread):
    def __init__(self, start_index, end_index) -> None:
        super().__init__()
        self.start_index = start_index
        self.end_index = end_index
        self.FLAG_DIRECTION = True
        self.FLAG_RUNNING = True
        self.time_interval = 0.03
    progress = QtCore.pyqtSignal(int)
    done = QtCore.pyqtSignal()

    def run(self):
        if (self.start_index <= self.end_index):
            while self.FLAG_RUNNING:
                if (self.FLAG_DIRECTION):
                    self.start_index += 1
                    if self.start_index > self.end_index:
                        break
                    self.progress.emit(self.start_index)
                    time.sleep(self.time_interval)
                else:
                    self.start_index -= 1
                    if self.start_index < 0:
                        break
                    self.progress.emit(self.start_index)
                    time.sleep(self.time_interval)

            self.done.emit()
        else:
            self.done.emit()
    

class MyGuiModule(QWidget):
    signal_space = QtCore.pyqtSignal()
    signal_current_index = QtCore.pyqtSignal(int)
    def __init__(self,rgb_npimgs, d_inferred_info:dict,parent=None):
        super().__init__(parent=parent)

        ## system control members
        self.myfe = FeatureExtractor()
        if not isinstance(rgb_npimgs, (list,tuple)):
            self.rgb_npimgs = []
        else:
            self.rgb_npimgs = rgb_npimgs
        self.parent_=parent
        self.MODEL_HEIGHT = 240
        self.MODEL_WIDTH = 320
        self.DISPLAY_RESIZE_WIDTH = self.MODEL_WIDTH * 2 + 20
        self.DISPLAY_RESIZE_HEIGHT = 100
        self.CURRENT_FILE_FULL_PATH = self.parent_.CURRENT_FILE_FULL_PATH
        self.BASE_PATH = self.parent_.parent_.FILE_PATH
        self.d_inferred_info = d_inferred_info
        self.no_display_img = np.zeros((1,1),dtype=np.uint8)
        
        

        self.init_members()
        self.init_ui()
        self.update_index()

    def init_members(self):
        ## basic control member
        self.current_index = 0
        self.curretn_index_add_value = 1
        self.maximum_index = len(self.rgb_npimgs)-1 if len(self.rgb_npimgs) != 0 else 0
        self.thread_ = None
        self.undo_list = []
        self.current_time_interval = 0.03
        self.FLAG_PLAY_FRAMES = False
        self.FLAG_BACK_FRAMES = False
        self.FLAG_LABEL_PUSHED = False
        self.inferred_info_imgs = None
        ## set inferred_info_imgs
        # self.get_base_arrays_imgs()  ## disabled
        
        
        ## parameters will be calculated by preprocessing function below
        ## x, y means the value of the circle's center position.
        self.init_x_list = None      # x series of the initial sequence before processing which is including None value.
        self.init_y_list = None      # y series of the initial sequence before processing which is including None value.
        self.base_roundness_catched_x_indices = None    # x series by filtering with the roundness in the d_inferred_info
        self.base_outlier_catched_x_indices = None      # x series by filtering with the outlier of the self.base_x_arr  (self.base_x_arr == processing(self.init_x_list)  , which has no None value after processing.)
        self.x_none_indices:np.ndarray = None   # indices of none value of the self.init_x_list
        self.base_merged_none_x_indices:np.ndarray = None # indices merged by each None value indices of the two of self.init_x_list, self.init_y_list
        self.base_x_arr:np.ndarray = None      # x series after processing which has no None value. None values should be filled with numeric value.
        self.base_y_arr:np.ndarray = None      # non used
        self.meaned_base_x_arr:np.ndarray = None     # meaned_base_x_arr = self.base_x_arr - self.base_x_arr.mean()
        self.meaned_base_y_arr:np.ndarray = None     # non used
        self.base_edge_indices:np.ndarray = None 
        
        ## members for displaying particular points
        self.current_selected_x_arr:MySeqObs1 = MySeqObs1(None,None,None) # x series currently selected by user's input
        self.current_outlier_indices = None    # outlier indices of currently selected x arr


        ## set members
        self.base_x_arr, self.base_y_arr = self.get_base_arrays(self.d_inferred_info)
        self.meaned_base_x_arr = self.base_x_arr - self.base_x_arr.mean()
        
        self.base_edge_indices = self.myfe.get_curve_indices(self.base_x_arr)
        
        self.current_selected_x_arr.set(self.base_x_arr,self.base_edge_indices, np.array([]))
        self.current_selected_x_arr.connect(self.set_current_outlier_indices)
        self.current_selected_x_arr.connect(self.update_index)

        ## members for plt window
        self.FLAG_PLT_SHOW = False
        self.PLT_SHOW_HEIGHT = 400
        self.PLT_SHOW_WIDTH = 1024
        self.temporal_plt_background = None
        self.fig, self.ax = plt.subplots(1,1,figsize=(20,10))
        self.plt_show_start_idx = None
        self.plt_show_end_idx = None
        self.plt_show_palette = PltShowPalette()
        self.plt_img = None
        self.plt_y_max_gap = 0
        ## parameters for show and display plt
        self.npimg_base_display = None
        self.npimg_current_display = None
        self.current_display_start_idx = 0
        self.current_display_end_idx = 120
        self.current_display_scale = 120
        # self.make_plot(self.current_display_start_idx, self.current_display_end_idx, plt_display=True)
        self.make_display_img(self.current_display_start_idx, self.current_display_end_idx)
        

    def init_ui(self):
        self.vlayout1 = QVBoxLayout()
        self.lbl_name = QLabel() 
        self.lbl_name.setLineWidth(2)
        # self.lbl_name.setFrameShape(0x0002)
        self.lbl_name.setFrameShape(QFrame.Panel)
        self.lbl_name.setFixedSize(100,30)
        self.lbl_name.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.vlayout1.addWidget(self.lbl_name)
        self.vlayout1.setAlignment(self.lbl_name,QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_pic = QLabel()
        # self.lbl_pic_rt.setFixedHeight(MODEL_HEIGHT *2)
        # self.lbl_pic_rt.setFixedWidth(MODEL_WIDTH *2)
        self.vlayout1.addWidget(self.lbl_pic)

        self.frame = QFrame()
        self.frame.setLineWidth(4)
        # self.frame.setFrameShape(0x0002)
        self.frame.setFrameShape(QFrame.Panel)
        # self.frame.setFrameShadow(0x0020)
        self.frame.setFrameShadow(QFrame.Raised)
        # self.frame.setFrameShadow(QFrame.Sunken)
        self.frame.mousePressEvent = self.revent_frame_mouse
        self.frame.setLayout(self.vlayout1)

        self.vlayout_main = QVBoxLayout()
        self.vlayout_main.addWidget(self.frame)

        self.lbl_display_info = QLabel()

        self.lbl_display = QLabel()
        # self.lbl_plt.setFixedWidth(self.MODEL_WIDTH*2)
        # self.lbl_plt.setFixedHeight(100)
        self.vlayout_main.addWidget(self.lbl_display_info)
        self.vlayout_main.addWidget(self.lbl_display)

        
        fixed_width = 55
        self.gbox_filter = QGroupBox("filter curve2linear(2,3, d_thres, f_thres)")
        self.gbox_filter.setCheckable(True)
        self.gbox_filter.setChecked(False)
        self.gbox_filter.setStyleSheet("QGroupBox { border: 1px solid black;}")
        self.hlayout_gbox_filter = QHBoxLayout()

        self.radio_filter_modify_base_curve = MyQRadioButton("",parent=self)
        self.radio_filter_curve2linear3 = MyQRadioButton("",parent=self)
        self.ledit_filter_distance_thres = MyQLineEdit()
        self.ledit_filter_distance_thres.setPlaceholderText("d")
        # self.ledit_filter_distance_thres.setDisabled(True)
        self.ledit_filter_frames_thres = MyQLineEdit()
        self.ledit_filter_frames_thres.setPlaceholderText("f")
        # self.ledit_filter_frames_thres.setDisabled(True)
        self.btn_filter_change = MyQPushButton("change")
        # self.btn_filter_change.setDisabled(True)
        self.btn_filter_change.setFixedWidth(fixed_width)
        self.btn_filter_erased_prev = MyQPushButton("e_prev")
        # self.btn_filter_erased_prev.setDisabled(True)
        self.btn_filter_erased_prev.setFixedWidth(fixed_width)
        self.btn_filter_erased_next = MyQPushButton("e_next")
        # self.btn_filter_erased_next.setDisabled(True)
        self.btn_filter_erased_next.setFixedWidth(fixed_width)

        self.hlayout_gbox_filter.addWidget(self.radio_filter_modify_base_curve)
        self.hlayout_gbox_filter.addWidget(self.radio_filter_curve2linear3)
        self.hlayout_gbox_filter.addWidget(self.ledit_filter_distance_thres)
        self.hlayout_gbox_filter.addWidget(self.ledit_filter_frames_thres)
        self.hlayout_gbox_filter.addWidget(self.btn_filter_change)
        self.hlayout_gbox_filter.addWidget(self.btn_filter_erased_prev)
        self.hlayout_gbox_filter.addWidget(self.btn_filter_erased_next)
        self.gbox_filter.setLayout(self.hlayout_gbox_filter)

        self.gbox_statistics = QGroupBox("statistics-irq(entire,dist,curve)")
        self.gbox_statistics.setCheckable(True)
        self.gbox_statistics.setChecked(False)
        self.gbox_statistics.setStyleSheet("QGroupBox { border: 1px solid black;}")
        self.hlayout_gbox_statistics = QHBoxLayout()

        self.radio_statistics_entire_gradient_outlier_activate = MyQRadioButton("e",parent=self)
        self.radio_statistics_dist_outlier_activate = MyQRadioButton("d", parent=self)
        # self.radio_statistics_dist_outlier_activate.setAutoExclusive(False)
        self.radio_statistics_curve_gradient_outlier_activate = MyQRadioButton("c",parent=self)
        self.radio_statistics_disabled_activate = MyQRadioButton("disabled",parent=self)
        # self.radio_statistics_curve_gradient_outlier_activate.setAutoExclusive(False)
        self.ledit_statistics_irq_multiplier_x10 = MyQLineEdit()
        self.ledit_statistics_irq_multiplier_x10.setPlaceholderText("x10")
        # self.ledit_statistics_irq_multiplier_x10.setDisabled(True)
        self.btn_statistics_outlier_change = MyQPushButton("change")
        # self.btn_statistics_outlier_change.setDisabled(True)
        self.btn_statistics_outlier_change.setFixedWidth(fixed_width)
        self.btn_statistics_outlier_prev = MyQPushButton("o_prev")
        # self.btn_statistics_outlier_prev.setDisabled(True)
        self.btn_statistics_outlier_prev.setFixedWidth(fixed_width)
        self.btn_statistics_outlier_next = MyQPushButton("o_next")
        # self.btn_statistics_outlier_next.setDisabled(True)
        self.btn_statistics_outlier_next.setFixedWidth(fixed_width)
        
        self.hlayout_gbox_statistics.addWidget(self.radio_statistics_entire_gradient_outlier_activate)
        self.hlayout_gbox_statistics.addWidget(self.radio_statistics_dist_outlier_activate)
        self.hlayout_gbox_statistics.addWidget(self.radio_statistics_curve_gradient_outlier_activate)
        self.hlayout_gbox_statistics.addWidget(self.radio_statistics_disabled_activate)
        self.hlayout_gbox_statistics.addWidget(self.ledit_statistics_irq_multiplier_x10)
        self.hlayout_gbox_statistics.addWidget(self.btn_statistics_outlier_change)
        self.hlayout_gbox_statistics.addWidget(self.btn_statistics_outlier_prev)
        self.hlayout_gbox_statistics.addWidget(self.btn_statistics_outlier_next)
        self.radio_statistics_disabled_activate.hide()
        self.gbox_statistics.setLayout(self.hlayout_gbox_statistics)

        self.hlayout_additional_infos1 = QHBoxLayout()
        self.hlayout_additional_infos1.addWidget(self.gbox_filter)
        self.hlayout_additional_infos1.addWidget(self.gbox_statistics)

        self.vlayout_main.addLayout(self.hlayout_additional_infos1)


        self.gbox_display = QGroupBox("display_wave")
        self.gbox_display.setStyleSheet("QGroupBox { border: 1px solid black;}")
        self.hlayout_gbox_display = QHBoxLayout()

        self.ledit_start_plt = MyQLineEdit(self)
        self.ledit_start_plt.setPlaceholderText('start')
        self.ledit_end_plt = MyQLineEdit(self)
        self.ledit_end_plt.setPlaceholderText('end')
        self.ledit_display_scale = MyQLineEdit(self)
        self.ledit_display_scale.setPlaceholderText('def_gap')

        self.hlayout_gbox_display.addWidget(self.ledit_start_plt)
        self.hlayout_gbox_display.addWidget(self.ledit_end_plt)
        self.hlayout_gbox_display.addWidget(self.ledit_display_scale)
        self.gbox_display.setLayout(self.hlayout_gbox_display)

        self.gbox_plt_show = QGroupBox("plt show")
        self.gbox_plt_show.setStyleSheet("QGroupBox { border: 1px solid black;}")
        self.hlayout_gbox_plt_show = QHBoxLayout()

        self.ledit_show_plt_width = MyQLineEdit(self)
        self.ledit_show_plt_width.setPlaceholderText("width")
        self.ledit_show_plt_height = MyQLineEdit(self)
        self.ledit_show_plt_height.setPlaceholderText("height")
        self.btn_show_plt = MyQPushButton('show', self)
        self.btn_show_plt.keyPressEvent = lambda x : x.ignore()

        self.hlayout_gbox_plt_show.addWidget(self.ledit_show_plt_width)
        self.hlayout_gbox_plt_show.addWidget(self.ledit_show_plt_height)
        self.hlayout_gbox_plt_show.addWidget(self.btn_show_plt)
        self.gbox_plt_show.setLayout(self.hlayout_gbox_plt_show)

        # self.gbox


        self.hlayout_plt_ctl = QHBoxLayout()
        self.hlayout_plt_ctl.addWidget(self.gbox_display)
        self.hlayout_plt_ctl.addWidget(self.gbox_plt_show)

        self.vlayout_main.addLayout(self.hlayout_plt_ctl)


        fixed_width = 65
        self.hlayout_fast_index = QHBoxLayout()
        self.btn_prev_5_index = MyQPushButton("-5", self)
        self.btn_prev_5_index.setFixedWidth(fixed_width)
        self.btn_next_5_index = MyQPushButton("+5", self)
        self.btn_next_5_index.setFixedWidth(fixed_width)
        self.btn_prev_10_index = MyQPushButton("-10", self)
        self.btn_prev_10_index.setFixedWidth(fixed_width)
        self.btn_next_10_index = MyQPushButton("+10", self)
        self.btn_next_10_index.setFixedWidth(fixed_width)
        self.btn_prev_60_index = MyQPushButton("-60", self)
        self.btn_prev_60_index.setFixedWidth(fixed_width)
        self.btn_next_60_index = MyQPushButton("+60", self)
        self.btn_next_60_index.setFixedWidth(fixed_width)
        self.hlayout_fast_index.addWidget(self.btn_prev_5_index)
        self.hlayout_fast_index.addWidget(self.btn_next_5_index)
        self.hlayout_fast_index.addWidget(self.btn_prev_10_index)
        self.hlayout_fast_index.addWidget(self.btn_next_10_index)
        self.hlayout_fast_index.addWidget(self.btn_prev_60_index)
        self.hlayout_fast_index.addWidget(self.btn_next_60_index)


        self.gbox_video_speed = QGroupBox("video_speed")
        self.gbox_video_speed.setStyleSheet("QGroupBox { border: 1px solid black;}")
        self.hlayout_gbox_video_speed = QHBoxLayout()

        self.radio_full_video_speed = MyQRadioButton("x1",self)
        self.radio_full_video_speed.setChecked(True)
        self.radio_half_video_speed = MyQRadioButton("x1/2",self)
        self.radio_quarter_video_speed = MyQRadioButton("x1/4",self)
        self.radio_1tenth_video_speed = MyQRadioButton("x1/10",self)
        
        self.hlayout_gbox_video_speed.addWidget(self.radio_full_video_speed)
        self.hlayout_gbox_video_speed.addWidget(self.radio_half_video_speed)
        self.hlayout_gbox_video_speed.addWidget(self.radio_quarter_video_speed)
        self.hlayout_gbox_video_speed.addWidget(self.radio_1tenth_video_speed)
        self.gbox_video_speed.setLayout(self.hlayout_gbox_video_speed)
        
        self.hlayout_video_ctl1 = QHBoxLayout()
        self.hlayout_video_ctl1.addLayout(self.hlayout_fast_index)
        self.hlayout_video_ctl1.addWidget(self.gbox_video_speed)
        self.vlayout_main.addLayout(self.hlayout_video_ctl1)



        self.hlayout1 = QHBoxLayout()
        self.lbl_current_index = QLabel()
        self.lbl_current_index.setText(f'0 / {self.maximum_index}')
        self.lbl_current_index.setFixedWidth(80)
        # self.ledit_goto_index = QLineEdit()
        self.ledit_goto_index = MyQLineEdit(self)
        self.ledit_goto_index.setFixedWidth(40)
        # self.a1 = QtGui.QIntValidator()
        # self.ledit_goto_index.setValidator(self.a1)
        # self.ledit_goto_index.keyPressEvent = lambda x : x.ignore() if x.key() == QtCore.Qt.Key_Space else super(QLineEdit,self.ledit_goto_index).keyPressEvent(x)
        # self.ledit_goto_index.keyPressEvent = self.revent_key_press1
        # self.ledit_goto_index.keyPressEvent = self.keyPressEvent
        # self.ledit_goto_index.installEventFilter(self)
        self.btn_goto_index = QPushButton('move index')
        # self.btn_goto_index.installEventFilter(self)
        self.btn_goto_index.keyPressEvent = lambda x : x.ignore() if x.key() == QtCore.Qt.Key_Space else super(QPushButton,self.btn_goto_index).keyPressEvent(x)
        self.btn_prev_index = QPushButton('prev(-d)')
        self.btn_prev_index.keyPressEvent = lambda x : x.ignore()
        self.ledit_index_add_value = MyQLineEdit()
        self.ledit_index_add_value.setFixedWidth(30)
        self.ledit_index_add_value.setPlaceholderText('d')
        self.btn_next_index = QPushButton('next(+d)')
        self.btn_next_index.keyPressEvent = lambda x : x.ignore()
        self.btn_back_frames = QPushButton('back')
        self.btn_back_frames.setIcon(QApplication.style().standardIcon(getattr(QStyle, 'SP_MediaSeekBackward')))
        self.btn_play_frames = QPushButton('play')
        pixmapi = getattr(QStyle, 'SP_MediaPlay')
        self.btn_play_frames.setIcon(QApplication.style().standardIcon(pixmapi))
        self.btn_stop_frames = QPushButton('stop',self)
        self.btn_stop_frames.keyPressEvent = lambda x: x.ignore()

        self.hlayout1.addWidget(self.lbl_current_index)
        self.hlayout1.addWidget(self.ledit_goto_index)
        self.hlayout1.addWidget(self.btn_goto_index)
        self.hlayout1.addWidget(self.btn_prev_index)
        self.hlayout1.addWidget(self.ledit_index_add_value)
        self.hlayout1.addWidget(self.btn_next_index)
        self.hlayout1.addWidget(self.btn_back_frames)
        self.hlayout1.addWidget(self.btn_play_frames)
        self.hlayout1.addWidget(self.btn_stop_frames)

        self.vlayout_main.addLayout(self.hlayout1)

        self.slide_index = QSlider(QtCore.Qt.Horizontal)
        self.slide_index.setMinimum(0)
        self.slide_index.setMaximum(self.maximum_index)
        self.slide_index.setTickPosition(QSlider.TicksBelow)
        self.slide_index.setTracking(True)
        self.is_slider_user_interact = True
        self.vlayout_main.addWidget(self.slide_index)
        
        self.hlayout2 = QHBoxLayout()
        self.lbl_start_index = QLabel("start :")
        self.ledit_start_index = MyQLineEdit(self)
        self.ledit_start_index.setFixedWidth(40)
        # self.ledit_start_index.setValidator(QtGui.QIntValidator())
        self.lbl_end_index = QLabel("  end :")
        self.ledit_end_index = MyQLineEdit(self)
        self.ledit_end_index.setFixedWidth(40)
        # self.ledit_end_index.setValidator(QtGui.QIntValidator())
        self.btn_reset_indices = QPushButton("reset")
        self.btn_reset_indices.keyPressEvent = lambda x : x.ignore()
        self.btn_push_indices = QPushButton("push")
        self.btn_push_indices.keyPressEvent = lambda x: x.ignore()
        self.btn_push_current_display_indices = QPushButton("C-display")
        self.btn_push_current_display_indices.keyPressEvent = lambda x:x.ignore()

        self.hlayout2.addWidget(self.lbl_start_index)
        self.hlayout2.addWidget(self.ledit_start_index)
        self.hlayout2.addWidget(self.lbl_end_index)
        self.hlayout2.addWidget(self.ledit_end_index)
        self.hlayout2.addWidget(self.btn_reset_indices)
        self.hlayout2.addWidget(self.btn_push_indices)
        self.hlayout2.addWidget(self.btn_push_current_display_indices)
        for i in range(8):
            self.hlayout2.addWidget(QLabel())

        self.vlayout_main.addLayout(self.hlayout2)

        self.hlayout3 = QHBoxLayout()
        self.vlist_indices = QListWidget()
        self.vlist_indices.keyPressEvent = lambda x: x.ignore()
        self.hlayout3.addWidget(self.vlist_indices)
        self.vlayout3_1 = QVBoxLayout()
        self.btn_erase_vlist = QPushButton("erase")
        self.btn_erase_vlist.keyPressEvent = lambda x: x.ignore()
        self.btn_clear_vlist = QPushButton("clear")
        self.btn_clear_vlist.keyPressEvent = lambda x: x.ignore()
        self.btn_undo_vlist = QPushButton("undo")
        self.btn_undo_vlist.keyPressEvent = lambda x: x.ignore()
        self.vlayout3_1.addWidget(self.btn_erase_vlist)
        self.vlayout3_1.addWidget(self.btn_clear_vlist)
        self.vlayout3_1.addWidget(self.btn_undo_vlist)
        self.hlayout3.addLayout(self.vlayout3_1)

        self.vlayout_main.addLayout(self.hlayout3)

        self.setLayout(self.vlayout_main)
        QtGui.QColor
        self.connect_ctl()

    def connect_ctl(self):
        self.gbox_filter.toggled.connect(self.cb_gbox_filter)
        self.radio_filter_modify_base_curve.toggled.connect(self.cb_radio_filter_modify_base_curve)
        self.radio_filter_curve2linear3.toggled.connect(self.cb_radio_filter_curve2linear3)
        self.btn_filter_change.clicked.connect(self.cb_btn_filter_change)
        self.btn_filter_erased_prev.clicked.connect(self.cb_btn_filter_erased_prev)
        self.btn_filter_erased_next.clicked.connect(self.cb_btn_filter_erased_next)

        self.gbox_statistics.toggled.connect(self.cb_gbox_statistics)
        self.radio_statistics_entire_gradient_outlier_activate.toggled.connect(self.cb_radio_statistics_entire_gradient_outlier_activate)
        self.radio_statistics_dist_outlier_activate.toggled.connect(self.cb_radio_statistics_dist_outlier_activate)
        self.radio_statistics_curve_gradient_outlier_activate.toggled.connect(self.cb_radio_statistics_curve_gradient_outlier_activate)
        self.radio_statistics_disabled_activate.toggled.connect(self.cb_radio_statistics_disabled_activate)
        self.btn_statistics_outlier_change.clicked.connect(self.cb_btn_statistics_outlier_change)
        self.btn_statistics_outlier_prev.clicked.connect(self.cb_btn_statistics_outlier_prev)
        self.btn_statistics_outlier_next.clicked.connect(self.cb_statistics_outlier_next)

        self.btn_show_plt.clicked.connect(self.cb_btn_show_plt)
        
        self.btn_prev_5_index.clicked.connect(self.cb_btn_prev_5_index)
        self.btn_next_5_index.clicked.connect(self.cb_btn_next_5_index)
        self.btn_prev_10_index.clicked.connect(self.cb_btn_prev_10_index)
        self.btn_next_10_index.clicked.connect(self.cb_btn_next_10_index)
        self.btn_prev_60_index.clicked.connect(self.cb_btn_prev_60_index)
        self.btn_next_60_index.clicked.connect(self.cb_btn_next_60_index)
        self.radio_full_video_speed.toggled.connect(self.cb_radio_full_video_speed)
        self.radio_half_video_speed.toggled.connect(self.cb_radio_half_video_speed)
        self.radio_quarter_video_speed.toggled.connect(self.cb_radio_quarter_video_speed)
        self.radio_1tenth_video_speed.toggled.connect(self.cb_radio_1tenth_video_speed)

        self.btn_goto_index.clicked.connect(self.cb_btn_goto_index)
        self.btn_prev_index.clicked.connect(self.cb_btn_prev_index)
        self.btn_next_index.clicked.connect(self.cb_btn_next_index)
        self.btn_back_frames.clicked.connect(self.cb_btn_back_frames)
        self.btn_play_frames.clicked.connect(self.cb_btn_play_frames)
        self.btn_stop_frames.clicked.connect(self.cb_btn_stop_frames)
        self.slide_index.valueChanged.connect(self.cb_slide_index)
        # self.slide_index.sliderMoved.connect(self.cb_slide_index)
        

        self.btn_reset_indices.clicked.connect(self.cb_btn_reset_indices)
        self.btn_push_indices.clicked.connect(self.cb_btn_push_indices)
        self.btn_push_current_display_indices.clicked.connect(self.cb_btn_push_current_display_indices)

        self.vlist_indices.itemDoubleClicked.connect(self.cb_vlist_indices_dclicked)
        self.btn_erase_vlist.clicked.connect(self.cb_btn_erase_vlist)
        self.btn_clear_vlist.clicked.connect(self.cb_btn_clear_vlist)
        self.btn_undo_vlist.clicked.connect(self.cb_btn_undo_vlist)

    def set_new_data(self, rgb_npimgs:np.ndarray, d_inferred_info:dict, parent):
        self.rgb_npimgs = rgb_npimgs
        self.d_inferred_info = d_inferred_info
        self.parent_ = parent
        self.CURRENT_FILE_FULL_PATH = self.parent_.CURRENT_FILE_FULL_PATH
        self.BASE_PATH = self.parent_.BASE_PATH
        self.init_members()
        try:
            self.vlist_indices.clear()
        except Exception as e:
            print(f"from set_new_data: clear_list_view error {e}")

        self.update_index()

    def __del__(self):
        del self.radio_filter_modify_base_curve
        del self.radio_statistics_entire_gradient_outlier_activate
        del self.radio_statistics_dist_outlier_activate
        del self.radio_statistics_curve_gradient_outlier_activate
        del self.radio_statistics_disabled_activate
        del self.ledit_start_plt
        del self.ledit_end_plt
        del self.ledit_display_scale
        del self.ledit_show_plt_width
        del self.ledit_show_plt_height
        del self.btn_show_plt
        del self.btn_prev_5_index
        del self.btn_next_5_index
        del self.btn_prev_10_index
        del self.btn_next_10_index
        del self.btn_prev_60_index
        del self.btn_next_60_index
        del self.radio_full_video_speed
        del self.radio_half_video_speed
        del self.radio_quarter_video_speed
        del self.radio_1tenth_video_speed
        del self.ledit_goto_index
        del self.btn_stop_frames
        del self.ledit_start_index
        del self.ledit_end_index
        del self.parent_
        del self.BASE_PATH
        del self.rgb_npimgs
        del self.d_inferred_info
        print("subview_module deleted")


    def cb_gbox_filter(self):
        if self.gbox_filter.isChecked():
             self.cb_btn_filter_change()
        else:
            self.current_selected_x_arr.set(self.base_x_arr, self.base_edge_indices, np.array([]))
            
            # self.update_index()
        ## self.current_upper_outlier_indices depends on the x_list( processed or filtered )
        self.cb_btn_statistics_outlier_change()

    def cb_radio_filter_modify_base_curve(self):
        if self.radio_filter_modify_base_curve.isChecked():
            self.cb_btn_filter_change()
        
    def cb_radio_filter_curve2linear3(self):
        if self.radio_filter_curve2linear3.isChecked():
            self.cb_btn_filter_change()
        

    def cb_btn_filter_change(self): 
        flag_filter_modify_base_curve = self.radio_filter_modify_base_curve.isChecked()
        flag_filter_curve2linear3 = self.radio_filter_curve2linear3.isChecked()

        d_thres = self.get_ledit_value(self.ledit_filter_distance_thres)
        f_thres = self.get_ledit_value(self.ledit_filter_frames_thres)

        if d_thres == 0 : return
        if d_thres == '': d_thres=2
        if f_thres == '': f_thres=5

        if flag_filter_modify_base_curve:
            filtered1_x_arr, erased_edge_indices, remain_edge_indices  = self.myfe.filter_modify_base_curve(self.base_x_arr,self.base_edge_indices,d_thres,f_thres)
            self.current_selected_x_arr.set(filtered1_x_arr, remain_edge_indices, erased_edge_indices)


    def cb_btn_filter_erased_prev(self):
        erased_edge_indices = self.current_selected_x_arr.erased_edge_indices
        if not isinstance(erased_edge_indices, np.ndarray):
            return
        candidates = erased_edge_indices[(erased_edge_indices-self.current_index)<0]
        if len(candidates) == 0:
            return
        prev_idx = candidates[-1]
        self.current_index = prev_idx
        self.update_index()

    def cb_btn_filter_erased_next(self):
        erased_edge_indices = self.current_selected_x_arr.erased_edge_indices
        if not isinstance(erased_edge_indices, np.ndarray):
            return
        candidates = erased_edge_indices[erased_edge_indices-self.current_index>0]
        if len(candidates) == 0:
            return
        next_idx = candidates[0]
        self.current_index = next_idx
        self.update_index()


    # def cb_help_radio_statistics_activate(self):
    #     bool_radio1 = self.radio_statistics_dist_outlier_activate.isChecked()
    #     bool_radio2 = self.radio_statistics_curve_gradient_outlier_activate.isChecked()
    #     if not bool_radio1 and not bool_radio2:
    #         self.ledit_statistics_irq_multiplier_x10.setDisabled(True)
    #         self.btn_statistics_outlier_change.setDisabled(True)
    #         self.btn_statistics_outlier_prev.setDisabled(True)
    #         self.btn_statistics_outlier_next.setDisabled(True)
    #     else:
    #         self.ledit_statistics_irq_multiplier_x10.setDisabled(False)
    #         self.btn_statistics_outlier_change.setDisabled(False)
    #         self.btn_statistics_outlier_prev.setDisabled(False)
    #         self.btn_statistics_outlier_next.setDisabled(False)
    #         self.cb_btn_statistics_outlier_change()

    def cb_gbox_statistics(self):
        if not self.gbox_statistics.isChecked():
            if not self.radio_statistics_disabled_activate.isChecked():
                self.radio_statistics_disabled_activate.toggle()
            
    def cb_radio_statistics_entire_gradient_outlier_activate(self):
        if self.radio_statistics_entire_gradient_outlier_activate.isChecked():
            self.cb_btn_statistics_outlier_change()

    def cb_radio_statistics_dist_outlier_activate(self):
        if self.radio_statistics_dist_outlier_activate.isChecked():
            self.cb_btn_statistics_outlier_change()

    def cb_radio_statistics_curve_gradient_outlier_activate(self):
        if self.radio_statistics_curve_gradient_outlier_activate.isChecked():
            self.cb_btn_statistics_outlier_change()

    def cb_radio_statistics_disabled_activate(self):
        if self.radio_statistics_disabled_activate.isChecked():
            self.cb_btn_statistics_outlier_change()
        
        
    def set_current_outlier_indices(self, **kargs):
        x_arr = kargs['data']
        edge_indices = kargs['edge_indices']
        edge_info_dict = kargs['edge_info_dict']
        bool_radio1 = self.radio_statistics_entire_gradient_outlier_activate.isChecked()
        bool_radio2 = self.radio_statistics_dist_outlier_activate.isChecked()
        bool_radio3 = self.radio_statistics_curve_gradient_outlier_activate.isChecked()
        bool_radio4 = self.radio_statistics_disabled_activate.isChecked()

        iqr = self.get_ledit_value(self.ledit_statistics_irq_multiplier_x10)
        if iqr == '':
            iqr = 15

        lower_bound=None
        upper_bound=None

        # if bool_radio1 and bool_radio2:
            # self.current_upper_outlier_indices, lower_bound, upper_bound = self.myfe.get_gradient_outlier_indices(x_arr,iqr_multiplier_x10=iqr,partial='upper',flag_edge=True)
        if bool_radio1:
            self.current_outlier_indices, lower_bound, upper_bound = self.myfe.get_gradient_outlier_indices(x_arr, edge_indices, edge_info_dict, iqr_multiplier_x10=iqr, partial='all', flag_edge=False)
        elif bool_radio2:
            self.current_outlier_indices, lower_bound, upper_bound = self.myfe.get_gradient_outlier_indices(x_arr, edge_indices, edge_info_dict, iqr_multiplier_x10=iqr, partial='positive', flag_edge=False)
        elif bool_radio3:
            self.current_outlier_indices, lower_bound, upper_bound = self.myfe.get_gradient_outlier_indices(x_arr, edge_indices, edge_info_dict, iqr_multiplier_x10=iqr, partial='positive', flag_edge=True)
        elif bool_radio4:
            self.current_outlier_indices = None
            lower_bound = None
            upper_bound = None

        if lower_bound is not None and upper_bound is not None:
            self.btn_statistics_outlier_change.setText(f"({lower_bound:5.1f}, {upper_bound:5.1f})")
        else:
            self.btn_statistics_outlier_change.setText(f"change")


    def cb_btn_statistics_outlier_change(self):
        
        self.set_current_outlier_indices(data=self.current_selected_x_arr.value, edge_indices=self.current_selected_x_arr.edge_indices, edge_info_dict=self.current_selected_x_arr.edge_info_dict)
        self.update_index()
        return


    def cb_btn_statistics_outlier_prev(self):
        if not isinstance(self.current_outlier_indices, np.ndarray):
            return
        candidates = self.current_outlier_indices[(self.current_outlier_indices-self.current_index)<0]
        if len(candidates)==0:
            return
        prev_idx = candidates[-1]
        self.current_index = prev_idx
        self.update_index()

    def cb_statistics_outlier_next(self):
        if not isinstance(self.current_outlier_indices, np.ndarray):
            return
        candidates = self.current_outlier_indices[(self.current_outlier_indices-self.current_index)>0]
        if len(candidates)==0:
            return

        next_idx = candidates[0]
        self.current_index = next_idx
        self.update_index()


    def cb_btn_show_plt(self):
        ## to plt.show(), its range is depends on the display range
        if not isinstance(self.current_selected_x_arr.value, np.ndarray): return
        
        width = self.get_ledit_value(self.ledit_show_plt_width)
        height = self.get_ledit_value(self.ledit_show_plt_height)

        if width == '':
            self.PLT_SHOW_WIDTH = 2000
        else:
            self.PLT_SHOW_WIDTH = width
        if height == '':
            self.PLT_SHOW_HEIGHT = 1000
        else:
            self.PLT_SHOW_HEIGHT = height

        self.make_plot(start=self.current_display_start_idx, end=self.current_display_end_idx, current_idx=self.current_index)
        if not isinstance(self.plt_img, np.ndarray): 
            return
        
        self.set_lbl_img(self.plt_show_palette.lbl_main_palette1,self.plt_img)
        if self.plt_show_palette.isVisible():
            self.plt_show_palette.close()
        else:
            self.plt_show_palette.show()


    def cb_btn_prev_5_index(self):
        self.current_index = self.get_valid_index(self.current_index-5)
        self.update_index()
    def cb_btn_next_5_index(self):
        self.current_index = self.get_valid_index(self.current_index+5)
        self.update_index()
    def cb_btn_prev_10_index(self):
        self.current_index = self.get_valid_index(self.current_index-10)
        self.update_index()
    def cb_btn_next_10_index(self):
        self.current_index = self.get_valid_index(self.current_index+10)
        self.update_index()
    def cb_btn_prev_60_index(self):
        self.current_index = self.get_valid_index(self.current_index-60)
        self.update_index()
    def cb_btn_next_60_index(self):
        self.current_index = self.get_valid_index(self.current_index+60)
        self.update_index()
    def cb_btn_prev_display_gap_index(self):
        self.current_index = self.get_valid_index(self.current_index-self.current_display_scale)
    def cb_btn_next_display_gap_index(self):
        self.current_index = self.get_valid_index(self.current_index+self.current_display_scale)

    def cb_radio_full_video_speed(self):
        if self.radio_full_video_speed.isChecked():
            self.current_time_interval = 0.03
    def cb_radio_half_video_speed(self):
        if self.radio_half_video_speed.isChecked():
            self.current_time_interval = 0.06
    def cb_radio_quarter_video_speed(self):
        if self.radio_quarter_video_speed.isChecked():
            self.current_time_interval = 0.12
    def cb_radio_1tenth_video_speed(self):
        if self.radio_1tenth_video_speed.isChecked():
            self.current_time_interval = 0.3
        

    def cb_btn_goto_index(self):
        index = self.ledit_goto_index.text()
        if not index.isdigit():
            return
        self.current_index = self.get_valid_index(int(index))
        self.update_index()

    def cb_btn_prev_index(self):
        add_value = self.get_ledit_value(self.ledit_index_add_value)
        if add_value == '':
            add_value = 1
        self.current_index = self.get_valid_index(self.current_index-add_value)
        self.update_index()

    def cb_btn_next_index(self):
        add_value = self.get_ledit_value(self.ledit_index_add_value)
        if add_value == '':
            add_value = 1
        self.current_index = self.get_valid_index(self.current_index+add_value)
        self.update_index()

    def help_play_frames(self, int_):
        self.current_index = int_
        self.update_index()

    def cb_btn_back_frames(self):
        # check label frame pushed
        if not self.FLAG_LABEL_PUSHED:
            # self.FLAG_LABEL_PUSHED = True
            # self.frame.setFrameShadow(QFrame.Sunken)
            self.toggle_label()
        # check play frame
        if self.FLAG_BACK_FRAMES: 
            self.cb_btn_stop_frames()
            return
        # play frame
        self.FLAG_BACK_FRAMES = True
        self.btn_back_frames.setIcon(QApplication.style().standardIcon(getattr(QStyle, 'SP_MediaPause')))
        self.thread_ = Mythread(self.current_index, self.maximum_index)
        self.thread_.time_interval = self.current_time_interval
        self.thread_.FLAG_DIRECTION = False
        self.thread_.progress.connect(self.help_play_frames)
        self.thread_.done.connect(self.cb_btn_stop_frames)
        self.btn_play_frames.setDisabled(True)
        self.thread_.start()

    def cb_btn_play_frames(self):
        # check label frame pushed
        if not self.FLAG_LABEL_PUSHED:
            # self.FLAG_LABEL_PUSHED = True
            # self.frame.setFrameShadow(QFrame.Sunken)
            self.toggle_label()
        # check play frame
        if self.FLAG_PLAY_FRAMES: 
            self.cb_btn_stop_frames()
            return
        # play frame
        self.FLAG_PLAY_FRAMES = True
        self.btn_play_frames.setIcon(QApplication.style().standardIcon(getattr(QStyle, 'SP_MediaPause')))
        self.thread_ = Mythread(self.current_index, self.maximum_index)
        self.thread_.time_interval = self.current_time_interval
        self.thread_.progress.connect(self.help_play_frames)
        self.thread_.done.connect(self.cb_btn_stop_frames)
        self.btn_back_frames.setDisabled(True)
        self.thread_.start()

    def cb_btn_stop_frames(self):
        self.FLAG_BACK_FRAMES = False
        self.FLAG_PLAY_FRAMES = False 
        self.btn_play_frames.setIcon(QApplication.style().standardIcon(getattr(QStyle, 'SP_MediaPlay')))
        self.btn_back_frames.setIcon(QApplication.style().standardIcon(getattr(QStyle, 'SP_MediaSeekBackward')))
        self.btn_back_frames.setDisabled(False)
        self.btn_play_frames.setDisabled(False)
        
        if not self.thread_:
            return
        self.thread_.FLAG_RUNNING = False
        

    def cb_slide_index(self):
        self.current_index = self.slide_index.value()
        if self.is_slider_user_interact == True:
            self.update_index()

    def cb_btn_reset_indices(self):
        self.ledit_start_index.clear()
        self.ledit_end_index.clear()


    def help_vlist_indices_record(self):
        n = self.vlist_indices.count()

        l_item = []
        for i in range(n):
            item = self.vlist_indices.item(i)
            # item not work well, use item.text()
            # l_item.append(item)
            l_item.append(item.text())

        self.undo_list.append(l_item)

    def cb_btn_push_indices(self):
        start = self.ledit_start_index.text()
        if start == '':
            start = 0
        elif start.isdigit():
            start = int(start)
        else:
            QMessageBox.information(None,"QMessageBox",f"from cb_btn_push_indices : not valid value")
            return
        end = self.ledit_end_index.text()
        if end == '':
            end = 0
        elif end.isdigit():
            end = int(end)
            if end > self.maximum_index:
                end = self.maximum_index
                self.ledit_end_index.setText(str(end))
        else:
            QMessageBox.information(None,"QMessageBox",f"from cb_btn_push_indices : not valid value")
            return
        
        if start > end:
            QMessageBox.information(None,"QMessageBox",f"from cb_btn_push_indices : start({start}) value is bigger than end({end})")
            return
        count_ = self.vlist_indices.count()
        self.vlist_indices.addItem(f'{count_}, start:{start}, end:{end}')
        self.help_vlist_indices_record()


    def cb_btn_push_current_display_indices(self):
        
        start = self.current_display_start_idx
        end = self.current_display_end_idx

        count_ = self.vlist_indices.count()
        self.vlist_indices.addItem(f'{count_}, start:{start}, end:{end}')
        self.help_vlist_indices_record()
    
    def cb_vlist_indices_dclicked(self):
        if not self.FLAG_LABEL_PUSHED:
            self.toggle_label()
        if self.FLAG_PLAY_FRAMES: 
            self.cb_btn_stop_frames()
            return
        current_row = self.vlist_indices.currentRow()
        text = self.vlist_indices.item(current_row).text()
        l_ = text.split(',')
        self.current_index = int(l_[1].split(':')[1])
        self.update_plt_show_palette()
        # self.update_index()
        end_index = int(l_[2].split(':')[1])
        self.thread_ = Mythread(self.current_index, end_index)
        self.FLAG_PLAY_FRAMES = True
        self.thread_.FLAG_DIRECTION = True
        self.thread_.time_interval = self.current_time_interval
        self.thread_.progress.connect(self.help_play_frames)
        self.thread_.done.connect(self.cb_btn_stop_frames)
        self.btn_play_frames.setDisabled(True)
        self.btn_back_frames.setDisabled(True)
        self.thread_.start()

    def cb_btn_erase_vlist(self):
        current_row = self.vlist_indices.currentRow()
        current_item = self.vlist_indices.takeItem(current_row)
        self.vlist_indices.removeItemWidget(current_item)

        n = self.vlist_indices.count()
        new_list = []
        for i in range(n):
            item = self.vlist_indices.item(i)
            l_ = item.text().split(',')
            l_.pop(0)
            l_.insert(0,str(i))
            str_ = ','.join(l_)
            new_list.append(str_)

        self.vlist_indices.clear()
        for i in range(n):
            self.vlist_indices.addItem(new_list[i])

        self.help_vlist_indices_record()

    def cb_btn_clear_vlist(self):
        n = self.vlist_indices.count()
        if n == 0:
            return
        reply = QMessageBox.question(self,"warning","do you want clear the list view?", QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self.vlist_indices.clear()
        self.help_vlist_indices_record()

    def cb_btn_undo_vlist(self):
        if len(self.undo_list) < 2:
            return
        self.vlist_indices.clear()
        l_items = self.undo_list[-2]
        for i in l_items:
            self.vlist_indices.addItem(i)
        self.undo_list.pop()

################ methods
    def update_index(self, **obskargs):
        self.signal_current_index.emit(self.current_index)
        self.set_current_index_lbl()
        self.is_slider_user_interact = False
        self.slide_index.setValue(self.current_index)
        self.is_slider_user_interact = True
        self.set_main_canvas1()
        self.update_display()
        self.update_plt_show_palette()
        self.set_display_canvas1()
        self.set_display_info_lbl()
    

    def set_display_index_parameter(self) -> bool:
        scale = self.get_ledit_value(self.ledit_display_scale)
        if scale == '':
            scale = self.current_display_scale
        is_refreshed = False

        ## check if the display should be refreshed
        if self.current_display_scale != scale:
            self.current_display_scale = scale
            self.current_display_start_idx = self.current_index
            self.current_display_end_idx = self.current_display_start_idx + self.current_display_scale
            is_refreshed = True

        if self.current_index < self.current_display_start_idx or self.current_index >= self.current_display_end_idx:
            self.current_display_start_idx = self.current_index
            self.current_display_end_idx = self.current_display_start_idx + self.current_display_scale
            is_refreshed = True
        if self.current_display_end_idx > self.maximum_index:
            self.current_display_end_idx = self.maximum_index
        
        return is_refreshed

    def update_display(self):
        if not isinstance(self.current_selected_x_arr.value, np.ndarray): 
            self.npimg_current_display = self.no_display_img
            return
        if not isinstance(self.npimg_base_display,np.ndarray): return

        self.set_display_index_parameter()
        # self.make_plot(self.current_display_start_idx,self.current_display_end_idx,plt_display=True)
        self.make_display_img(self.current_display_start_idx,self.current_display_end_idx)
    
        ## add red_vertical line
        idx = self.current_index - self.current_display_start_idx
        display_range = self.current_display_end_idx - self.current_display_start_idx
        if display_range == 0:
            line_width = self.DISPLAY_RESIZE_WIDTH
        else:
            line_width = self.DISPLAY_RESIZE_WIDTH / (self.current_display_end_idx - self.current_display_start_idx)
            line_width = int(line_width)
        self.npimg_current_display = self.npimg_base_display.copy()
        self.npimg_current_display[:,idx*line_width : idx*line_width+line_width] += np.array([127,0,0],dtype=np.uint8)

    
        ## currently, update_plt_show_palette would be worked similarly with update_display
    def update_plt_show_palette(self):
        if not self.plt_show_palette.isVisible():
            return

        # this update takes too much time to use on fast time interval. so limit it.
        # this would work only x1/10 speed
        # if self.current_time_interval < 0.12 and (self.FLAG_PLAY_FRAMES or self.FLAG_BACK_FRAMES):
        #     return

        ## no longer support play image
        if self.FLAG_PLAY_FRAMES or self.FLAG_BACK_FRAMES:
            return

        is_refresh = self.set_display_index_parameter()

        self.plt_show_start_idx = self.current_display_start_idx
        self.plt_show_end_idx = self.current_display_end_idx
    
        ## set the self.plt_img. below function will set it
        ## at this time, make_plot not support is_refresh == True. so put the value as always True
        self.make_plot(start=self.plt_show_start_idx, end=self.plt_show_end_idx, current_idx=self.current_index, is_refresh=True)
        self.set_lbl_img(self.plt_show_palette.lbl_main_palette1,self.plt_img)

    def set_display_info_lbl(self):
        if not isinstance(self.current_selected_x_arr.value,np.ndarray):
            return
        if self.init_x_list[self.current_index] == None:
            data = 'None'
        else:
            data = round(self.init_x_list[self.current_index],1)
        
        r_catched = True if self.current_index in self.base_roundness_catched_x_indices else False    
        out_catched = True if self.current_index in self.base_outlier_catched_x_indices else False
        merged = True if self.current_index in self.base_merged_none_x_indices else False
        text = f'data({data:5}), merged_none({merged:5}), processed({self.base_x_arr[self.current_index]:5.1f}), r_catched({r_catched:5}), out_catched({out_catched:5})'
        self.lbl_display_info.setText(text)

    def get_ledit_value(self,ledit_widget):
        text = ledit_widget.text()
        if text == '':
            return ''
        elif text.isdigit():
            return int(text)
        else:
            ## without regex, input(1.5) will be cause exception of not digit.
            ## detecting float value required

            rec = re.compile('\d+[.]\d')
            match = rec.match(text)
            if match:
                return float(text) 
            else:
                raise Exception(f"from get_ledit_value: value has not numeric character ({text})")
    
    def get_valid_index(self, int_):
        if int_ < 0:
            return 0
        if int_ > self.maximum_index:
            return self.maximum_index
        return int_

    def set_name(self,name):
        if not isinstance(name,str):
            return
        self.lbl_name.setText(name)

    def set_pic_size(self,width,height):
        if not isinstance(width,int) or not isinstance(height,int):
            return
        self.lbl_pic.setFixedWidth(width)
        self.lbl_pic.setFixedHeight(height)

    def set_data(self,data):
        if not isinstance(data,(list,tuple)):
            return
        self.rgb_npimgs = data

    def revent_frame_mouse(self, e):
        if e.buttons() == QtCore.Qt.LeftButton:
            self.toggle_label(e)
        elif e.button() == QtCore.Qt.RightButton:
            if not self.ledit_start_index.text():
                # QMessageBox.information(None,"message","something")
                self.ledit_start_index.setText(str(self.current_index))
            else:
                self.ledit_end_index.setText(str(self.current_index))

    def toggle_label(self,e=None):
        # called by without e
        if not e:
            if not self.FLAG_PLAY_FRAMES and not self.FLAG_BACK_FRAMES:
                if self.FLAG_LABEL_PUSHED:
                    self.FLAG_LABEL_PUSHED = False
                    self.frame.setFrameShadow(QFrame.Raised)
                    self.btn_back_frames.clearFocus()
                    self.btn_play_frames.clearFocus()
                else:
                    self.FLAG_LABEL_PUSHED = True
                    self.frame.setFrameShadow(QFrame.Sunken)
        # called by event handler
        elif e.buttons() == QtCore.Qt.LeftButton:
            if not self.FLAG_PLAY_FRAMES and not self.FLAG_BACK_FRAMES:
                if self.FLAG_LABEL_PUSHED:
                    self.FLAG_LABEL_PUSHED = False
                    self.frame.setFrameShadow(QFrame.Raised)
                    self.btn_back_frames.clearFocus()
                    self.btn_play_frames.clearFocus()
                else:
                    self.FLAG_LABEL_PUSHED = True
                    self.frame.setFocus()
                    self.frame.setFrameShadow(QFrame.Sunken)

    def set_current_index_lbl(self):
        self.lbl_current_index.setText(f'{self.current_index} / {self.maximum_index}')

    def set_lbl_img(self, lbl, npimg_rgb:np.ndarray):
        lbl.clear()
        npimg_copy = npimg_rgb.copy()
        npimg_cvt = npimg_copy
        # if np.ndim(npimg) == 3:
        #     npimg_cvt = cv2.cvtColor(npimg_copy, cv2.COLOR_BGR2RGB)
        # elif np.ndim(npimg) == 2:
        #     npimg_cvt = cv2.cvtColor(npimg_copy, cv2.COLOR_GRAY2RGB)
        # else:
        #     print('np.ndim is not 2 or 3')
        #     return
        qimg = QtGui.QImage(npimg_cvt.data, npimg_cvt.shape[1], npimg_cvt.shape[0], npimg_cvt.shape[1]*3, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        lbl.setPixmap(pixmap)   

    def set_main_canvas1(self):
        if self.inferred_info_imgs != None:
            if not isinstance(self.inferred_info_imgs[self.current_index], np.ndarray):
                return
            self.set_lbl_img(self.lbl_pic,self.inferred_info_imgs[self.current_index])
        else:
            if not isinstance(self.rgb_npimgs[self.current_index], np.ndarray):
                return
            self.set_lbl_img(self.lbl_pic,self.rgb_npimgs[self.current_index])
        
        # if not isinstance(self.rgb_npimgs[self.current_index], np.ndarray):
        #     return
        # self.set_lbl_img(self.lbl_pic,self.rgb_npimgs[self.current_index])

    def set_display_canvas1(self):
        if not isinstance(self.npimg_current_display, np.ndarray):
            return
        self.set_lbl_img(self.lbl_display, self.npimg_current_display)


    def get_vlist_items(self):
        n = self.vlist_indices.count()
        result = []
        if n == 0:
            return result
        else:
            for i in range(n):
                text = self.vlist_indices.item(i).text()
                l_ = text.split(',')
                start_index = int(l_[1].split(':')[1])
                end_index = int(l_[2].split(':')[1])
                result.append([start_index,end_index])

        return result

    def run_frames(self):
        if self.FLAG_LABEL_PUSHED:
                if self.FLAG_PLAY_FRAMES or self.FLAG_BACK_FRAMES:
                    self.cb_btn_stop_frames()
                    return
                elif not self.FLAG_PLAY_FRAMES:
                    self.cb_btn_play_frames()
                    return

    def revent_key_press1(self,e):
        # return False

        # super(QLineEdit,self.ledit_goto_index).keyPressEvent(e)
        super().keyPressEvent(e)
        
        # self.ledit_goto_index.keyPressEvent = lambda x : x.ignore() if x.key() == QtCore.Qt.Key_Space else super(QLineEdit,self.ledit_goto_index).keyPressEvent(x)
        # if e.key() == QtCore.Qt.Key_Space:
            # e.ignore()
        # else:
            # super(QLineEdit,self.ledit_goto_index).keyPressEvent(e)
            # a = self.sender()
            # t= type(a)
            # super(type(self.sender()),self.sender()).keyPressEvent(e)


    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        # if a0.key() == QtCore.Qt.Key_Escape:
            # self.close()
        if a0.key() == QtCore.Qt.Key_Space:
            self.signal_space.emit()
            self.run_frames()
        # return super().keyPressEvent(a0)
        # super().keyPressEvent(a0)
        
            # QLineEdit().keyPressEvent(a0)
        # return super(QLineEdit,self.ledit_goto_index).keyPressEvent(a0)
        return super(MyGuiModule,self).keyPressEvent(a0)
        # return super(QLineEdit,self.ledit_goto_index).keyPressEvent(a0)

    def eventFilter(self, a0: 'QObject', a1: 'QEvent') -> bool:
        if a1.type() == QtCore.QEvent.KeyPress:
            if a1.key() == QtCore.Qt.Key_Space:
                self.run_frames()
        # return False
        return super().eventFilter(a0, a1)
        # return super(QWidget,self).eventFilter(a0, a1)


    def get_base_arrays(self, d_inferred_info:dict) -> 'list[np.ndarray, np.ndarray]':
        if not isinstance(d_inferred_info,dict):
            return

        x_points, y_points = zip(*d_inferred_info['centers'])
        self.init_x_list = x_points[:]
        self.init_y_list = y_points[:]
        roundnesses = d_inferred_info['roundnesses']

        self.x_none_indices = np.array([index for index, item in enumerate(x_points) if item == None])
        
        x_points = self.myfe.fill_na(list(x_points), 'tip')
        y_points = self.myfe.fill_na(list(y_points), 'tip')

        x_points, roundness_catched_x_indices = self.myfe.mask_with_roundness(x_points, roundnesses)
        self.base_roundness_catched_x_indices = roundness_catched_x_indices
        y_points, roundness_catched_y_indices = self.myfe.mask_with_roundness(y_points, roundnesses)

        x_points, outlier_catched_x_indices = self.myfe.mask_with_value_outlier(x_points)
        self.base_outlier_catched_x_indices = outlier_catched_x_indices
        y_points, outlier_catched_y_indices = self.myfe.mask_with_value_outlier(y_points)

        x_points, merged_none_x_indices = self.myfe.merge_none(x_points, y_points)
        self.base_merged_none_x_indices = merged_none_x_indices
        # y_points, merged_none_y_indices = self.myfe.merge_none(y_points, x_points)

        x_points = self.myfe.fill_na(x_points, 'all')
        y_points = self.myfe.fill_na(y_points, 'all')

        return [np.array(x_points), np.array(y_points)]


    def get_base_arrays_imgs(self):
        if not isinstance(self.rgb_npimgs, (list,tuple)):
            return
        frame_length = len(self.rgb_npimgs)
        if frame_length == 0 :
            return
        if not isinstance(self.d_inferred_info, dict):
            return
        
        try:
            centers = self.d_inferred_info['centers']
            roundnesses = self.d_inferred_info['roundnesses']
            widths = self.d_inferred_info['widths']
            heights = self.d_inferred_info['heights']
            radians = self.d_inferred_info['radians']
        except Exception as e :
            QMessageBox.information(None,'QMessageBox', f'from get_base_arrays_imgs: Exception {e}')
            return

        if len(centers) != frame_length:
            QMessageBox.information(None, 'QMEssageBox', f'from get_base_arrays_imgs: the number of frames({frame_length}) and inferred_infos({len(centers)}) are not same length')
            return

        h,w,c = self.rgb_npimgs[0].shape

        info_imgs = []
        for i in range(frame_length):
            temp_img = self.rgb_npimgs[i].copy()
            x, y = centers[i]
            if x != None and y != None:
                x = int(x * 2) 
                y = int(y * 2)
                temp_img[y,x,:] = np.array([255,0,0], dtype=np.uint8) 
                roundness = round(roundnesses[i],3)
                width = int(widths[i] * 2)
                height = int(heights[i] * 2)
                radian = radians[i]
                temp_img = cv2.putText(temp_img, f'roundness: {roundness}', (10,10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
                temp_img = cv2.putText(temp_img, f'x: {x}, y: {y}', (10,h-10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
                temp_img = cv2.ellipse(temp_img, (x,y), (width, height), int(np.rad2deg(radian)),0,360, (255,0,0),1)
            
            else:
                temp_img = cv2.putText(temp_img, f'None', (10,10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
            
            info_imgs.append(temp_img)

        self.inferred_info_imgs = info_imgs


    def help_make_display_make_indices_resized(self,data:np.ndarray,start:int,end:int,resized_width:int) -> np.ndarray:
        if isinstance(data,list):
            data = np.array(data)
        if not isinstance(data,np.ndarray):
            return np.array([])
        inner_range_indices = self.myfe.get_inner_values(data,start,end)
        if len(inner_range_indices) == 0:
            return np.array([])
        new_indices = inner_range_indices - start
        new_indices *= resized_width
        new_indices_resized = np.array([],dtype=np.int32)
        for i in range(resized_width+1):
            new_indices_resized = np.append(new_indices_resized,new_indices+i)

        return new_indices_resized


    def make_display_img(self, start:int=0, end:int=None) -> np.ndarray:
        data = self.current_selected_x_arr.value
        if not isinstance(data,np.ndarray):
            return

        if start == None:
            start = 0
        if start >= self.maximum_index:
            return
        if end == None:
            end = len(data)
        if end <= start:
            end = start + 100
        if end > self.maximum_index:
            end = self.maximum_index

        x_datas = data[start:end+1]
        max_x = max(x_datas)
        min_x = min(x_datas)
        width = len(x_datas)
        height = (max_x - min_x)
        if height == 0: return
        ## rescale for the size of display. -1 is require for access image height range and another -1 is require for filling inner range(height)
        adjusted_x_datas = (x_datas - min_x) / height * (self.DISPLAY_RESIZE_HEIGHT-1)
        
        adjusted_x_datas_int = adjusted_x_datas.copy().astype(np.int32)
        adjusted_x_datas_decimal = adjusted_x_datas - adjusted_x_datas_int
        adjusted_x_datas_decimal = np.where(adjusted_x_datas_decimal >= 0.5, 1, 0)
        adjusted_x_datas_rounded = adjusted_x_datas_int + adjusted_x_datas_decimal
        
        ## upside down values to make img correctly. each of new_x_datas value will be height row of image  
        new_x_datas = (self.DISPLAY_RESIZE_HEIGHT-1) - adjusted_x_datas_rounded
        
        white_point_height = int((self.DISPLAY_RESIZE_HEIGHT-1) / height)
        white_point_width = int(self.DISPLAY_RESIZE_WIDTH / width)

        palette = np.zeros((self.DISPLAY_RESIZE_HEIGHT,self.DISPLAY_RESIZE_WIDTH,3), dtype=np.uint8)

        new_x_datas = new_x_datas.astype(np.int32)
        
        none_indices_resized = self.help_make_display_make_indices_resized(self.x_none_indices, start, end, white_point_width)
        # roundness_catched_indices_resized = self.help_make_display_make_indices_resized(self.base_roundness_catched_x_indices, start, end, white_point_width)
        # outlier_catched_indices_resized = self.help_make_display_make_indices_resized(self.base_outlier_catched_x_indices, start, end, white_point_width)
        # merged_none_indices_resized = self.help_make_display_make_indices_resized(self.base_merged_none_x_indices, start, end, white_point_width)
        current_outlier_indices_resized = self.help_make_display_make_indices_resized(self.current_outlier_indices, start, end, white_point_width)
        edge_info_manual_nystagmus_indices = self.current_selected_x_arr.edge_info_manual_nystagmus_indices
        manual_nystagmus_indices = self.help_make_display_make_indices_resized(edge_info_manual_nystagmus_indices, start, end, white_point_width)
        edge_info_statistical_nystagmus_indices = self.current_selected_x_arr.edge_info_statistical_nystagmus_indices
        statistical_nystagmus_indices = self.help_make_display_make_indices_resized(edge_info_statistical_nystagmus_indices, start, end, white_point_width)
        
        ## draw statistical_nystagmus_indices
        if isinstance(statistical_nystagmus_indices, np.ndarray) and len(statistical_nystagmus_indices) != 0:
            palette[:, statistical_nystagmus_indices] += np.array([127,127,60], dtype=np.uint8)
 
 
        ## draw manual_nystagmus_indices
        if isinstance(manual_nystagmus_indices, np.ndarray) and len(manual_nystagmus_indices) != 0:
            palette[:, manual_nystagmus_indices] += np.array([128,0,128], dtype=np.uint8)
 
 
        for i in range(width):
            # palette[new_x_datas[i]-white_point_height:new_x_datas[i], i*white_point_width:i*white_point_width+white_point_width] = np.array([255,255,255], dtype=np.uint8)
            palette[new_x_datas[i], i*white_point_width:i*white_point_width+white_point_width] = np.array([255,255,255], dtype=np.uint8)



        if isinstance(none_indices_resized, np.ndarray) and len(none_indices_resized) != 0:
            palette[:, none_indices_resized] += np.array([0,60,0], dtype=np.uint8)
        
        # if isinstance(current_outlier_indices_resized, np.ndarray):
        #     palette[:, current_outlier_indices_resized] += np.array([60,0,90], dtype=np.uint8)
        
        self.npimg_base_display = palette
        return


    # ## this function is no longer used. will be deprecated.
    # def make_plot_with_animate(self, start:int=0, end:int=None, current_idx:int=None, is_refresh:bool=True) -> None:
    #     ## prepare to draw
    #     fig = self.fig
    #     axes = self.ax
    #     axes.clear()   

    #     if is_refresh == False:
    #         if self.temporal_plt_background is None:
    #             self.make_plot(start,end,current_idx,is_refresh=True)
    #             # print(1)

    #         fig.canvas.restore_region(self.temporal_plt_background)
            
    #         # if current_idx != None:
    #         #     ani_vline = axes.axvline(current_idx-start,0,1,color='red', linestyle='solid', animated=True)
    #         #     axes.draw_artist(ani_vline)
    #         if current_idx != None:
    #             axes.axvline(current_idx-start,0,1,color='red', linestyle='solid')
                
    #         ## save plt to numpy
    #         fig.canvas.draw()
    #         plt_img = np.array(fig.canvas.renderer._renderer)
    #         plt_img_rgb = cv2.cvtColor(plt_img, cv2.COLOR_RGBA2RGB)
    #         plt_img_rgb_resized = cv2.resize(plt_img_rgb, (self.PLT_SHOW_WIDTH,self.PLT_SHOW_HEIGHT), interpolation=cv2.INTER_LINEAR)
    #         self.plt_img = plt_img_rgb_resized.copy()
    #         # print(2)
    #         return
        
    #     current_x_arr = self.current_selected_x_arr.value
    #     if not isinstance(current_x_arr,np.ndarray):
    #         raise Exception("from make_plot: self.current_selected_x_arr.value is no np.ndarray")
    #     if start == None:
    #         start = 0
    #     if start >= self.maximum_index:
    #         return
    #     if end == None:
    #         end = len(current_x_arr)
    #     if end <= start:
    #         end = start + 100
    #     if end > self.maximum_index:
    #         end = self.maximum_index
        
    #     ## prepare basic datas
    #     base_x_inner_datas = self.base_x_arr[start:end+1]
    #     meaned_base_x_inner_datas = base_x_inner_datas - base_x_inner_datas.mean()
    #     current_selected_x_inner_datas = current_x_arr[start:end+1]
    #     meaned_current_selected_x_inner_datas = current_selected_x_inner_datas - current_selected_x_inner_datas.mean()
    #     current_edge_indices = self.current_selected_x_arr.edge_indices
    #     current_edge_indices_inner_range = self.myfe.get_inner_values(current_edge_indices,start,end)
    #     current_erased_indices = self.current_selected_x_arr.erased_edge_indices
    #     current_erased_indices_inner_range = self.myfe.get_inner_values(current_erased_indices,start,end)
    #     edge_info_statistical_nystagmus_indices = self.current_selected_x_arr.edge_info_statistical_nystagmus_indices
    #     edge_info_statistical_nystagmus_indices_inner_range = self.myfe.get_inner_values(edge_info_statistical_nystagmus_indices,start,end)
    #     edge_info_manual_nystagmus_indices = self.current_selected_x_arr.edge_info_manual_nystagmus_indices
    #     edge_info_manual_nystagmus_indices_inner_range = self.myfe.get_inner_values(edge_info_manual_nystagmus_indices, start, end)
        
        
    #     max_y = max(meaned_current_selected_x_inner_datas)
    #     min_y = min(meaned_current_selected_x_inner_datas)
    #     y_ticks_ = np.linspace(min_y,max_y,5)
    #     x_ticks_ = np.linspace(start,end,5)
    #     y_gap = y_ticks_[4]-y_ticks_[0]
    #     plt_y_gap = 0 
    #     if y_gap > plt_y_gap:
    #         plt_y_gap = y_gap

    #     bottom, top = plt.ylim()

    #     # if self.temporal_plt_background == None:
    #     #     fig.canvas.draw()
    #     #     self.temporal_plt_background = fig.canvas.copy_from_bbox(axes.bbox)

    #     ## draw x, y ticks with animate
    #     # art_text_xticks = []
    #     # art_text_yticks = []
    #     # for i in x_ticks:
    #     #     art_text_xticks.append(axes.text(i-start,min_y-y_gap*0.1,f'{i:.0f}', animated=True))
    #     # for i in y_ticks:
    #     #     art_text_yticks.append(axes.text(0,i,f'{i:.2f}', animated=True))
    #     # for i in range(len(art_text_xticks)):
    #     #         art_text = art_text_xticks[i]
    #     #         axes.draw_artist(art_text)
    #     # for i in art_text_yticks:
    #     #         axes.draw_artist(i)
        
    #     ## draw basic plot with animate. must this plot follow drawing one of scatter or text series. otherwise if this plot comes at first, not working proferly
    #     # ani_plot_current_selected_x = axes.plot(range(end-start+1), meaned_current_selected_x_inner_datas,animated=True)[0]
    #     # axes.draw_artist(ani_plot_current_selected_x)



    #     ## draw basic plot
    #     axes.text(20,min_y-0.1, f'plt_y_gap:{plt_y_gap:.2f}')
    #     axes.text(60,min_y-0.2, f'yticks:{bottom,top}')
    #     for i in x_ticks_:
    #         axes.text(i-start,min_y-0.2,f"{i:.0f}")
    #         axes.axvline(i-start,0,1,c='gray',linestyle='--')
    #     axes.plot(range(end-start+1), meaned_base_x_inner_datas, c='cyan')
    #     axes.plot(range(end-start+1), meaned_current_selected_x_inner_datas, c='blue')



    #     ## draw curve_indices info with animate
    #     # if isinstance(self.curve_indices, np.ndarray):
    #     #     curve_indices_inner_range = self.myfe.get_inner_values(self.curve_indices, start, end)
    #     #     art_texts1 = []
    #     #     texts1 = []
    #     #     for i in curve_indices_inner_range:
    #     #         curve_idx = np.where(self.curve_indices == i)[0][0]
    #     #         if curve_idx > len(self.y_diffs)-1:
    #     #             continue
    #     #         art_texts1.append(axes.text(i-start, meaned_current_selected_x_inner_datas[i-start], '', rotation=30, color='red',animated=True))
    #     #         texts1.append(f'{self.y_diffs[curve_idx]:.1f}, {self.grad_ratios[curve_idx]:5.2f}')
    #     #     for i in range(len(art_texts1)):
    #     #         art_text = art_texts1[i]
    #     #         art_text.set_text(texts1[i])
    #     #         axes.draw_artist(art_text)

    #     #     ani_scat1 = axes.scatter(curve_indices_inner_range-start, meaned_current_selected_x_inner_datas[curve_indices_inner_range-start], color='g', animated=True)
    #     #     axes.draw_artist(ani_scat1)
        

    #     ## draw edge_indices info
    #     if len(current_edge_indices_inner_range) != 0:
    #         axes.scatter(current_edge_indices_inner_range-start, meaned_current_selected_x_inner_datas[current_edge_indices_inner_range-start], color='g')

    #     edge_indices = current_edge_indices
    #     edge_inner_indices = current_edge_indices_inner_range
    #     grads = self.current_selected_x_arr.edge_info_grads
    #     grad_ratios = self.current_selected_x_arr.edge_info_grad_ratios
    #     distances1 = self.current_selected_x_arr.edge_info_distances1
    #     distances2 = self.current_selected_x_arr.edge_info_distances2
    #     x_diffs = self.current_selected_x_arr.edge_info_x_diffs
    #     y_diffs = self.current_selected_x_arr.edge_info_y_diffs
        
    #     for i in edge_inner_indices:
    #         edge_idx = np.where(edge_indices == i)[0][0]
    #         if edge_idx > len(grads)-1:
    #             continue
    #         axes.text(i-start, meaned_current_selected_x_inner_datas[i-start], f"g({grads[edge_idx]:.1f}), gr({grad_ratios[edge_idx]:.2f})", rotation=30)
    #         # axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-min(y_gap*0.1, 1), f"d1({distances1[edge_idx]:.2f}), d2({distances2[edge_idx]:.2f})", rotation=30)
    #         axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-0.2, f"d1({distances1[edge_idx]:.2f}), d2({distances2[edge_idx]:.2f})", rotation=30)
    #         # axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-min(y_gap*0.2, 2), f"x_d({x_diffs[edge_idx]:.2f}), y_d({y_diffs[edge_idx]:.2f})", rotation=30)
    #         axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-0.4, f"x_d({x_diffs[edge_idx]:.2f}), y_d({y_diffs[edge_idx]:.2f})", rotation=30)

    #     ## draw x_none_indices with animate
    #     # if isinstance(self.x_none_indices, np.ndarray):
    #     #     none_indices_inner_range = self.myfe.get_inner_values(self.x_none_indices, start, end)
    #     #     ani_scat2 = axes.scatter(none_indices_inner_range-start, meaned_current_selected_x_inner_datas[none_indices_inner_range-start], color='black', animated=True)
    #     #     axes.draw_artist(ani_scat2)

    #     ## draw x_none_indices
    #     if isinstance(self.x_none_indices, np.ndarray):
    #         none_indices_inner_range = self.myfe.get_inner_values(self.x_none_indices, start, end)
    #         axes.scatter(none_indices_inner_range-start, meaned_current_selected_x_inner_datas[none_indices_inner_range-start], color='black')


    #     ## draw erased_curves_by_filter with animate
    #     # if isinstance(self.erased_curves_by_filter, np.ndarray):
    #     #     erased_curves_by_filter_inner_range = self.myfe.get_inner_values(self.erased_curves_by_filter,start,end)
    #     #     ani_scat_erased = axes.scatter(erased_curves_by_filter_inner_range-start, meaned_base_x_inner_datas[erased_curves_by_filter_inner_range-start], color='gray', animated=True)
    #     #     axes.draw_artist(ani_scat_erased)

    #     ## draw current_erased_indices_inner_range
    #     if len(current_erased_indices_inner_range) != 0:
    #         axes.scatter(current_erased_indices_inner_range-start, meaned_base_x_inner_datas[current_erased_indices_inner_range-start], color='gray')
            

    #     ## draw current_outliers with animate
    #     # if isinstance(self.current_outlier_indices, np.ndarray):
    #     #     current_outlier_indices_inner_range = self.myfe.get_inner_values(self.current_outlier_indices,start,end)
    #     #     art_text_current_outliers = []
    #     #     for i in current_outlier_indices_inner_range:
    #     #         art_text_current_outliers.append(axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-y_gap*0.2, 'n2', color='blue',animated=True))
    #     #     for i in range(len(art_text_current_outliers)):
    #     #         axes.draw_artist(art_text_current_outliers[i])

    #     ## draw current_outliers
    #     if isinstance(self.current_outlier_indices, np.ndarray):
    #         current_outlier_indices_inner_range = self.myfe.get_inner_values(self.current_outlier_indices,start,end)
    #         if len(current_outlier_indices_inner_range) != 0:
    #             for i in current_outlier_indices_inner_range:
    #                 # axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-min(y_gap*0.1, 1), 'o', color='orange')
    #                 axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-0.2, 'o', color='orange')
            
    #     if len(edge_info_statistical_nystagmus_indices_inner_range) != 0:
    #         for i in edge_info_statistical_nystagmus_indices_inner_range:
    #             # axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-min(y_gap*0.2, 2), 'stat_n', color='y')
    #             axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-0.4, 'stat_n', color='y')

    #     if len(edge_info_manual_nystagmus_indices_inner_range) != 0:
    #         for i in edge_info_manual_nystagmus_indices_inner_range:
    #             # axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-min(y_gap*0.3,3), 'man_n', color='purple')
    #             axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-0.6, 'man_n', color='purple')


    #     fig.canvas.draw()
    #     self.temporal_plt_background = fig.canvas.copy_from_bbox(axes.bbox)
    #     # fig.canvas.blit(axes.bbox)
    #     # fig.canvas.flush_events()
    #     if current_idx != None:
    #         axes.axvline(current_idx-start,0,1,color='red', linestyle='solid')
    #     fig.canvas.draw()
    #     plt_img = np.array(fig.canvas.renderer._renderer)
    #     plt_img_rgb = cv2.cvtColor(plt_img, cv2.COLOR_RGBA2RGB)
    #     plt_img_rgb_resized = cv2.resize(plt_img_rgb, (self.PLT_SHOW_WIDTH,self.PLT_SHOW_HEIGHT), interpolation=cv2.INTER_LINEAR)
    #     self.plt_img = plt_img_rgb_resized.copy()
    #     # print('3')
    #     return

    def make_plot(self, start:int=0, end:int=None, current_idx:int=None, is_refresh:bool=True) -> None:
        ## prepare to draw
        fig = self.fig
        axes = self.ax
        axes.clear()   
        
        current_x_arr = self.current_selected_x_arr.value
        if not isinstance(current_x_arr,np.ndarray):
            raise Exception("from make_plot: self.current_selected_x_arr.value is no np.ndarray")
        if start == None:
            start = 0
        if start >= self.maximum_index:
            return
        if end == None:
            end = len(current_x_arr)
        if end <= start:
            end = start + 100
        if end > self.maximum_index:
            end = self.maximum_index
        
        ## prepare basic datas
        base_x_inner_datas = self.base_x_arr[start:end+1]
        meaned_base_x_inner_datas = base_x_inner_datas - base_x_inner_datas.mean()
        current_selected_x_inner_datas = current_x_arr[start:end+1]
        meaned_current_selected_x_inner_datas = current_selected_x_inner_datas - current_selected_x_inner_datas.mean()
        current_edge_indices = self.current_selected_x_arr.edge_indices
        current_edge_indices_inner_range = self.myfe.get_inner_values(current_edge_indices,start,end)
        current_erased_indices = self.current_selected_x_arr.erased_edge_indices
        current_erased_indices_inner_range = self.myfe.get_inner_values(current_erased_indices,start,end)
        edge_info_statistical_nystagmus_indices = self.current_selected_x_arr.edge_info_statistical_nystagmus_indices
        edge_info_statistical_nystagmus_indices_inner_range = self.myfe.get_inner_values(edge_info_statistical_nystagmus_indices,start,end)
        edge_info_manual_nystagmus_indices = self.current_selected_x_arr.edge_info_manual_nystagmus_indices
        edge_info_manual_nystagmus_indices_inner_range = self.myfe.get_inner_values(edge_info_manual_nystagmus_indices, start, end)
        
        
        ## draw basic plot first. this will set xticks, yticks, xlim, ylim automatically.
        axes.plot(range(end-start+1), meaned_base_x_inner_datas, c='cyan')
        axes.plot(range(end-start+1), meaned_current_selected_x_inner_datas, c='blue')
        
        ## calculate tick_values to draw text.
        max_y = max(meaned_current_selected_x_inner_datas)
        min_y = min(meaned_current_selected_x_inner_datas)
        y_ticks_ = np.linspace(min_y,max_y,5)
        x_ticks_ = np.linspace(start,end,5)
        y_gap = y_ticks_[4]-y_ticks_[0]
        
        if y_gap > self.plt_y_max_gap:
            self.plt_y_max_gap = y_gap
        
        loc, labels = plt.yticks()
        bottom, top = axes.get_ylim()
        # axes.text(10,min_y-y_gap*0.05, f'plt_y_gap:{self.plt_y_max_gap:.2f}')
        axes.text(40,min_y-y_gap*0.05, f'bottom, top:{bottom:.2f}, {top:.2f}')
        # axes.text(70,min_y-y_gap*0.05, f'yticks:{loc, labels}')
        for i in x_ticks_:
            axes.text(i-start,min_y-y_gap*0.05,f"{i:.0f}")
            axes.axvline(i-start,0,1,c='gray',linestyle='--')


        ## draw edge_indices info
        if len(current_edge_indices_inner_range) != 0:
            axes.scatter(current_edge_indices_inner_range-start, meaned_current_selected_x_inner_datas[current_edge_indices_inner_range-start], color='g')

        ## prepare edge infos.
        grads = self.current_selected_x_arr.edge_info_grads
        grad_ratios = self.current_selected_x_arr.edge_info_grad_ratios
        distances1 = self.current_selected_x_arr.edge_info_distances1
        distances2 = self.current_selected_x_arr.edge_info_distances2
        x_diffs = self.current_selected_x_arr.edge_info_x_diffs
        y_diffs = self.current_selected_x_arr.edge_info_y_diffs
        
        for i in current_edge_indices_inner_range:
            edge_idx = np.where(current_edge_indices == i)[0][0]
            if edge_idx > len(grads)-1:
                continue
            axes.text(i-start, meaned_current_selected_x_inner_datas[i-start], f"g({grads[edge_idx]:.1f}), gr({grad_ratios[edge_idx]:.2f})", rotation=30)
            # axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-self.plt_y_max_gap*0.02, f"d1({distances1[edge_idx]:.2f}), d2({distances2[edge_idx]:.2f})", rotation=30)
            axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-y_gap*0.02, f"d1({distances1[edge_idx]:.2f}), d2({distances2[edge_idx]:.2f})", rotation=30)
            # axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-self.plt_y_max_gap*0.04, f"x_d({x_diffs[edge_idx]:.2f}), y_d({y_diffs[edge_idx]:.2f})", rotation=30)
            axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-y_gap*0.04, f"x_d({x_diffs[edge_idx]:.2f}), y_d({y_diffs[edge_idx]:.2f})", rotation=30)

        ## draw x_none_indices
        if isinstance(self.x_none_indices, np.ndarray):
            none_indices_inner_range = self.myfe.get_inner_values(self.x_none_indices, start, end)
            axes.scatter(none_indices_inner_range-start, meaned_current_selected_x_inner_datas[none_indices_inner_range-start], color='black')

        ## draw current_erased_indices_inner_range
        if len(current_erased_indices_inner_range) != 0:
            axes.scatter(current_erased_indices_inner_range-start, meaned_base_x_inner_datas[current_erased_indices_inner_range-start], color='gray')


        
        ## draw statistical_nystagmus_indices
        if len(edge_info_statistical_nystagmus_indices_inner_range) != 0:
            for i in edge_info_statistical_nystagmus_indices_inner_range:
                axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-y_gap*0.06, 'stat_n', color='y')

        ## draw manual nystagmus indices
        if len(edge_info_manual_nystagmus_indices_inner_range) != 0:
            for i in edge_info_manual_nystagmus_indices_inner_range:
                axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-y_gap*0.08, 'man_n', color='purple')


        ## draw current_outliers
        if isinstance(self.current_outlier_indices, np.ndarray):
            current_outlier_indices_inner_range = self.myfe.get_inner_values(self.current_outlier_indices,start,end)
            if len(current_outlier_indices_inner_range) != 0:
                for i in current_outlier_indices_inner_range:
                    # axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-self.plt_y_max_gap*0.02, 'o', color='orange')
                    axes.text(i-start, meaned_current_selected_x_inner_datas[i-start]-y_gap*0.1, 'o', color='orange')


        ## draw vertical red line to indicate current index on plt.
        if current_idx != None:
            axes.axvline(current_idx-start,0,1,color='red', linestyle='solid')
        
        ## draw and make it to nparray to show through label image.
        fig.canvas.draw()
        plt_img = np.array(fig.canvas.renderer._renderer)
        plt_img_rgb = cv2.cvtColor(plt_img, cv2.COLOR_RGBA2RGB)
        plt_img_rgb_resized = cv2.resize(plt_img_rgb, (self.PLT_SHOW_WIDTH,self.PLT_SHOW_HEIGHT), interpolation=cv2.INTER_LINEAR)
        self.plt_img = plt_img_rgb_resized.copy()
        
        return 

if __name__ == '__main__':
    app = QApplication(sys.argv)
    arr = np.zeros((240*2, 320 * 2), dtype=np.uint8)
    l_ = []
    for h in range(240 * 2):
        arr[h,:] = 255
        l_.append(arr.copy())
    view = MyGuiModule(l_, None)
    # view.installEventFilter(view)
    view.show()
    view.set_name("Rt")
    view.set_pic_size(320 * 2, 240 * 2)

    
    sys.exit(app.exec_())