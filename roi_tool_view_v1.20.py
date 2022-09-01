import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtCore 
from roi_tool_ctl import MODEL_HEIGHT, MODEL_WIDTH, MainCtl

MODEL_HEIGHT
MODEL_WIDTH

class MainGui(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()


    def init_ui(self):
        self.setWindowTitle('roi tool')

        self.vlayout = QVBoxLayout()

        self.info1 = Info1()
        self.info2 = Info2()
        self.info3 = Info3()

        self.vlayout.addWidget(self.info1)

        self.hlayout1 = QHBoxLayout()
        self.hlayout1.addWidget(self.info2)
        self.hlayout1.addWidget(self.info3)

        self.vlayout.addLayout(self.hlayout1)

        # self.vlayout.addWidget(self.info3)

        self.setLayout(self.vlayout)

class Info1(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.hlayout = QHBoxLayout()

        self.btn_open_video = QPushButton()
        self.btn_open_video.setText('open video')
        self.btn_open_folder = QPushButton("open folder")

        self.btn_set_lt = QPushButton()
        self.btn_set_lt.setText('set Lt')
        self.btn_set_rt = QPushButton()
        self.btn_set_rt.setText('set Rt')
        self.btn_set_lt_rt = QPushButton()
        self.btn_set_lt_rt.setText('set Lt_Rt')
        self.btn_set_lt_rt.setDisabled(True)
        self.lbl_current_location = QLabel()
        self.lbl_current_location.setText('Lt')
        self.btn_extract_index = QPushButton("extract_index")

        self.hlayout.addWidget(self.btn_open_video)
        self.hlayout.addWidget(self.btn_open_folder)
        self.hlayout.addWidget(self.btn_set_lt)
        self.hlayout.addWidget(self.btn_set_rt)
        self.hlayout.addWidget(self.btn_set_lt_rt)
        self.hlayout.addWidget(self.lbl_current_location)
        self.hlayout.addWidget(self.btn_extract_index)

        self.setLayout(self.hlayout)
    


class Info2(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()


    def init_ui(self):
        self.vlayout = QVBoxLayout()

        self.lbl_main_canvas1 = QLabel()
        self.lbl_main_canvas1.setText('no Video')
        self.lbl_main_canvas1.setFixedSize(320,240)
        
        self.hlayout1 = QHBoxLayout()
        self.lbl_current_index = QLabel()
        self.lbl_current_index.setText('? / ? : ?')
        self.lbl_current_index.setFixedWidth(120)
        self.ledit_goto_index = QLineEdit()
        self.ledit_goto_index.setFixedWidth(30)
        self.btn_goto_index = QPushButton()
        self.btn_goto_index.setText('move index')
        self.btn_prev_index = QPushButton()
        self.btn_prev_index.setText('prev')
        self.btn_next_index = QPushButton()
        self.btn_next_index.setText('next')
        self.btn_play_frames = QPushButton()
        self.btn_play_frames.setText('play')
        self.btn_stop_frames = QPushButton()
        self.btn_stop_frames.setText('stop')

        self.hlayout1.addWidget(self.lbl_current_index)
        self.hlayout1.addWidget(self.ledit_goto_index)
        self.hlayout1.addWidget(self.btn_goto_index)
        self.hlayout1.addWidget(self.btn_prev_index)
        self.hlayout1.addWidget(self.btn_next_index)
        self.hlayout1.addWidget(self.btn_play_frames)
        self.hlayout1.addWidget(self.btn_stop_frames)


        self.hlayout2 = QHBoxLayout()
        self.btn_set_orig = QPushButton()
        self.btn_set_orig.setText('set orig')
        self.btn_set_rois = QPushButton()
        self.btn_set_rois.setText('set rois')
        self.btn_set_ellipse = QPushButton()
        self.btn_set_ellipse.setText('set ellipse')
        self.btn_set_calib_ellipse1 = QPushButton()
        self.btn_set_calib_ellipse1.setText('set c_ellipse1')
        self.btn_set_calib_ellipse2 = QPushButton()
        self.btn_set_calib_ellipse2.setText('set c_ellipse2')

        self.hlayout2.addWidget(self.btn_set_orig)
        self.hlayout2.addWidget(self.btn_set_rois)
        self.hlayout2.addWidget(self.btn_set_ellipse)
        self.hlayout2.addWidget(self.btn_set_calib_ellipse1)
        self.hlayout2.addWidget(self.btn_set_calib_ellipse2)
        
        self.hlayout3 = QHBoxLayout()
        self.btn_get_mask1 = QPushButton()
        self.btn_get_mask1.setText('get mask1')
        self.btn_get_mask1.setDisabled(True)
        self.btn_get_mask2 = QPushButton()
        self.btn_get_mask2.setText('get 5mask2')
        self.btn_get_mask2.setDisabled(True)
        
        self.hlayout3.addWidget(self.btn_get_mask1)
        self.hlayout3.addWidget(self.btn_get_mask2)

        self.hlayout4 = QHBoxLayout()
        self.btn_resize_2x = QPushButton()
        self.btn_resize_2x.setText('resize 2x')
        self.btn_resize_4x = QPushButton()
        self.btn_resize_4x.setText('resize 4x')

        self.hlayout4.addWidget(self.btn_resize_2x)
        self.hlayout4.addWidget(self.btn_resize_4x)

        self.lbl_ellipse_info = QLabel()
        self.lbl_ellipse_info.setText('no ellipse info')

        self.hlayout5 = QHBoxLayout()
        self.ledit_ellipse_info_center_x = QLineEdit()
        self.ledit_ellipse_info_center_x.setFixedWidth(40)
        self.ledit_ellipse_info_center_x.setPlaceholderText('x')
        self.ledit_ellipse_info_center_y = QLineEdit()
        self.ledit_ellipse_info_center_y.setFixedWidth(40)
        self.ledit_ellipse_info_center_y.setPlaceholderText('y')
        self.ledit_ellipse_info_width = QLineEdit()
        self.ledit_ellipse_info_width.setFixedWidth(40)
        self.ledit_ellipse_info_width.setPlaceholderText('w')
        self.ledit_ellipse_info_height = QLineEdit()
        self.ledit_ellipse_info_height.setFixedWidth(40)
        self.ledit_ellipse_info_height.setPlaceholderText('h')
        self.ledit_ellipse_info_radian = QLineEdit()
        self.ledit_ellipse_info_radian.setFixedWidth(40)
        self.ledit_ellipse_info_radian.setPlaceholderText('rad')
        self.btn_load_current_ellipse_info = QPushButton()
        self.btn_load_current_ellipse_info.setText('load info')
        self.btn_ellipse_info_change = QPushButton()
        self.btn_ellipse_info_change.setText('change')
        self.btn_ellipse_info_save = QPushButton()
        self.btn_ellipse_info_save.setText('save')

        self.hlayout5.addWidget(self.ledit_ellipse_info_center_x)
        self.hlayout5.addWidget(self.ledit_ellipse_info_center_y)
        self.hlayout5.addWidget(self.ledit_ellipse_info_width)
        self.hlayout5.addWidget(self.ledit_ellipse_info_height)
        self.hlayout5.addWidget(self.ledit_ellipse_info_radian)
        self.hlayout5.addWidget(self.btn_load_current_ellipse_info)
        self.hlayout5.addWidget(self.btn_ellipse_info_change)
        self.hlayout5.addWidget(self.btn_ellipse_info_save)

        self.vlist_ellipse_info_list = QListWidget()
        # self.vlist_ellipse_info_list.setFlow(0)
        

        self.hlayout6 = QHBoxLayout()
        self.btn_erase_ellipse_info_list = QPushButton()
        self.btn_erase_ellipse_info_list.setText('erase')
        self.btn_clear_ellipse_info_list = QPushButton()
        self.btn_clear_ellipse_info_list.setText('clear')

        self.hlayout6.addWidget(self.btn_erase_ellipse_info_list)
        self.hlayout6.addWidget(self.btn_clear_ellipse_info_list)

        self.hlayout7 = QHBoxLayout()

        self.btn_draw_ellipse_with_points_4x = QPushButton('draw_ellipse 4x')
        self.btn_draw_ellipse_with_points_8x = QPushButton('draw_ellipse 8x')

        self.hlayout7.addWidget(self.btn_draw_ellipse_with_points_4x)
        self.hlayout7.addWidget(self.btn_draw_ellipse_with_points_8x)

        self.vlayout.addWidget(self.lbl_main_canvas1)
        self.vlayout.addLayout(self.hlayout1)
        self.vlayout.addLayout(self.hlayout2)
        self.vlayout.addLayout(self.hlayout3)
        self.vlayout.addLayout(self.hlayout4)
        self.vlayout.addWidget(self.lbl_ellipse_info)
        self.vlayout.addLayout(self.hlayout5)
        self.vlayout.addWidget(self.vlist_ellipse_info_list)
        self.vlayout.addLayout(self.hlayout6)
        self.vlayout.addLayout(self.hlayout7)

        self.setLayout(self.vlayout)

class Info3(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):

        self.vlayout = QVBoxLayout()
        self.lbl_main_canvas2 = QLabel()
        self.lbl_main_canvas2.setText('no Mask')
        self.lbl_main_canvas2.setFixedSize(320,240)


        self.hlayout1 = QHBoxLayout()
        self.btn_edit_mask_empty = QPushButton()
        self.btn_edit_mask_empty.setText('empty')
        self.btn_edit_mask_left = QPushButton()
        self.btn_edit_mask_left.setText('left')
        self.btn_edit_mask_right = QPushButton()
        self.btn_edit_mask_right.setText('right')
        self.btn_edit_mask_up = QPushButton()
        self.btn_edit_mask_up.setText('up')
        self.btn_edit_mask_down = QPushButton()
        self.btn_edit_mask_down.setText('down')
        self.btn_edit_mask_island = QPushButton()
        self.btn_edit_mask_island.setText('island')
        self.btn_edit_mask_fill_contour = QPushButton('fill cnt')


        self.hlayout1.addWidget(self.btn_edit_mask_empty)
        self.hlayout1.addWidget(self.btn_edit_mask_left)
        self.hlayout1.addWidget(self.btn_edit_mask_right)
        self.hlayout1.addWidget(self.btn_edit_mask_up)
        self.hlayout1.addWidget(self.btn_edit_mask_down)
        self.hlayout1.addWidget(self.btn_edit_mask_island)
        self.hlayout1.addWidget(self.btn_edit_mask_fill_contour)

        
        self.hlayout2 = QHBoxLayout()
        self.btn_edit_mask_1x = QPushButton('edit 1x')
        self.btn_edit_mask_2x = QPushButton()
        self.btn_edit_mask_2x.setText('edit 2x')
        self.btn_edit_mask_4x = QPushButton()
        self.btn_edit_mask_4x.setText('edit 4x')
        self.btn_edit_mask_8x = QPushButton()
        self.btn_edit_mask_8x.setText('edit 8x')
        self.btn_edit_mask_16x = QPushButton()
        self.btn_edit_mask_16x.setText('edit 16x')


        self.hlayout2.addWidget(self.btn_edit_mask_1x)
        self.hlayout2.addWidget(self.btn_edit_mask_2x)
        self.hlayout2.addWidget(self.btn_edit_mask_4x)
        self.hlayout2.addWidget(self.btn_edit_mask_8x)
        self.hlayout2.addWidget(self.btn_edit_mask_16x)


        self.hlayout3 = QHBoxLayout()

        self.btn_load_current_mask = QPushButton()
        self.btn_load_current_mask.setText('load_mask')
        self.btn_load_mask_file = QPushButton()
        self.btn_load_mask_file.setText('load_mask_file')
        self.btn_save_temp_mask = QPushButton()
        self.btn_save_temp_mask.setText('save temp')
        self.btn_save_file_mask = QPushButton()
        self.btn_save_file_mask.setText('save file')

        self.hlayout3.addWidget(self.btn_load_current_mask)
        self.hlayout3.addWidget(self.btn_load_mask_file)
        self.hlayout3.addWidget(self.btn_save_temp_mask)
        self.hlayout3.addWidget(self.btn_save_file_mask)

        self.vlist_maskimgs = QListWidget()

        self.hlayout4 = QHBoxLayout()
        self.btn_erase_vlist_maskimgs = QPushButton()
        self.btn_erase_vlist_maskimgs.setText('erase')
        self.btn_clear_vlist_maskimgs = QPushButton()
        self.btn_clear_vlist_maskimgs.setText('clear')

        self.hlayout4.addWidget(self.btn_erase_vlist_maskimgs)
        self.hlayout4.addWidget(self.btn_clear_vlist_maskimgs)

        self.vlayout.addWidget(self.lbl_main_canvas2)
        self.vlayout.addLayout(self.hlayout1)
        self.vlayout.addLayout(self.hlayout2)
        self.vlayout.addWidget(self.vlist_maskimgs)
        self.vlayout.addLayout(self.hlayout3)
        self.vlayout.addLayout(self.hlayout4)


        self.setLayout(self.vlayout)






if __name__ == '__main__':
    app = QApplication(sys.argv)
    view = MainGui()
    view.show()
    
    ctl = MainCtl(view)
    sys.exit(app.exec_())