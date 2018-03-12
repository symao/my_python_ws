import sys
import cv2
import os
import time
import numpy as np
import sys

from video_reverse import video_reverse
from video_cut import video_cut
from video_combine import video_combine
from video_compose import video_compose
from video2img import video2img

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi

class VideoToolGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_file = ''
        self.open_video = False
        loadUi('videotool.ui', self)
        self.initUI()

    def initUI(self):
        self.selectButton.clicked.connect(self.select)
        self.saveButton.clicked.connect(self.save)
        self.cutSelectButton.clicked.connect(self.cut)
        self.loadAppendButton.clicked.connect(self.load_append)
        self.loadInsertButton.clicked.connect(self.load_insert)
        self.grabImgButton.clicked.connect(self.grab_img)
        self.strideSlider.valueChanged.connect(self.downsample_slide)
        self.resizeRateSlider.valueChanged.connect(self.resize_slide)
        self.resizeRateEdit.textChanged.connect(self.set_resize_rate)

        self.downsampleCheckBox.setEnabled(False)
        self.resizeCheckBox.setEnabled(False)
        self.cutCheckBox.setEnabled(False)
        self.reverseCheckBox.setEnabled(False)
        self.appendCheckBox.setEnabled(False)
        self.insertCheckBox.setEnabled(False)

        self.inputEdit.setText(self.video_file)
        self.videoInfoEdit.setText('No video opened.')

        self.setWindowTitle('Video Process Tools')
        self.show()

    def init_open_dir(self):
        return '.' if not os.path.exists(self.video_file) else os.path.dirname(self.video_file)

    def select(self):
        self.open_video = False
        self.videoInfoEdit.setText('No video opened.')
        self.inputEdit.setText('')
        video_file = QFileDialog.getOpenFileName(self, "Open Video", self.init_open_dir(), "Video Files (*.avi *.mp4)")[0]
        if video_file and os.path.exists(video_file):
            self.video_file = video_file
            self.inputEdit.setText(self.video_file)
            self.open_video = self.read_video_info(self.video_file)

        self.downsampleCheckBox.setChecked(False)
        self.resizeCheckBox.setChecked(False)
        self.cutCheckBox.setChecked(False)
        self.reverseCheckBox.setChecked(False)
        self.appendCheckBox.setChecked(False)
        self.insertCheckBox.setChecked(False)

        self.downsampleCheckBox.setEnabled(self.open_video)
        self.resizeCheckBox.setEnabled(self.open_video)
        self.cutCheckBox.setEnabled(self.open_video)
        self.reverseCheckBox.setEnabled(self.open_video)
        self.appendCheckBox.setEnabled(self.open_video)
        self.insertCheckBox.setEnabled(self.open_video)

    def read_video_info(self, video_file):
        cap = cv2.VideoCapture(video_file)
        if cap.isOpened():
            self.frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.videoInfoEdit.setText("Open success.\nFrames: %d  FPS: %.1f \nImage size: %d x %d"%
                                        (self.frame_cnt, self.fps, self.width, self.height))
            self.startIdxEdit.setText(str(0))
            self.endIdxEdit.setText(str(self.frame_cnt-1))
            self.xMinEdit.setText(str(0))
            self.xMaxEdit.setText(str(self.width))
            self.yMinEdit.setText(str(0))
            self.yMaxEdit.setText(str(self.height))
            return True
        else:
            self.videoInfoEdit.setText('Open video falied.\n')
            return False

    def parsing_options(self):
        option_str = 'Save coniguration:\n'
        # stride
        stride = 1
        if self.downsampleCheckBox.isChecked():
            stride = int(self.strideEdit.text())
            option_str += '  - downsample, stride:%d\n'%stride
        # resize
        resize_rate = 1
        resize_size = (self.width,self.height)
        if self.resizeCheckBox.isChecked():
            resize_rate = float(self.resizeRateEdit.text())
            resize_size = (int(self.width*resize_rate), int(self.height*resize_rate))
            option_str += '  - resize, %f(%d x %d)\n'%(resize_rate, resize_size[0], resize_size[1])
        # cut
        start_idx = 0
        end_idx = self.frame_cnt-1
        img_rect = (0,0,self.width,self.height)
        rect = img_rect
        if self.cutCheckBox.isChecked():
            start_idx = int(self.startIdxEdit.text())
            end_idx = int(self.endIdxEdit.text())
            rect = (int(self.xMinEdit.text()),int(self.yMinEdit.text()),\
                    int(self.xMaxEdit.text()),int(self.yMaxEdit.text()))
            option_str += '  - cut, range:[%d,%d], rect:[%d,%d,%d,%d]\n'%(start_idx,end_idx,
                                                    rect[0], rect[1], rect[2], rect[3])
        #reverse
        reverse = self.reverseCheckBox.isChecked()
        if reverse:
            option_str += '  - reverse\n'
        #append
        append_file = None
        if self.appendCheckBox.isChecked() and self.appendEdit.text():
            append_file = self.appendEdit.text()
            option_str += '  - append, file:%s\n'%append_file
        #insert
        insert_file = None
        if self.insertCheckBox.isChecked() and self.insertCheckBox.text():
            insert_file = self.insertEdit.text()
            option_str += '  - insert, file:%s\n'%insert_file
        return option_str, stride, resize_rate, resize_size, start_idx, end_idx, rect, reverse, append_file, insert_file

    def save(self):
        if not self.open_video:
            QMessageBox.warning(self, "Save failed", "No video opened.")
            return
        option_str, stride, resize_rate, resize_size, start_idx, end_idx,\
                rect, reverse, append_file, insert_file = self.parsing_options()
        reply = QMessageBox.question(self, 'Save coniguration', option_str + 
                                            "Click 'Yes' to continue, 'No' to cancel.")

        if reply == QMessageBox.Yes:
            save_file = QFileDialog.getSaveFileName(self, "Save Video",os.path.join(self.init_open_dir(),'out.avi'),
                                                    'AVI Video (*.avi);;MP4 Video (*.mp4)')[0]
            if not save_file:
                return
            # start saving
            progress = QProgressDialog(self)
            progress.setWindowTitle("Video Save")  
            progress.setLabelText("Process video and saving...               ")
            progress.setMinimumDuration(1)
            progress.setWindowModality(Qt.WindowModal)
            progress.setRange(0,100)
            progress.setFixedWidth(600)
            cap = cv2.VideoCapture(self.video_file)
            for frame_idx in range(start_idx+1):
                ret, frame = cap.read()
            save_rect = (0,0)+resize_size
            img_rect = (0,0,self.width,self.height)
            if rect != img_rect:
                save_rect = tuple([int(x*resize_rate) for x in rect])
            writer = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*"XVID"), self.fps,
                                    (save_rect[2]-save_rect[0], save_rect[3]-save_rect[1]))
            while ret and frame_idx<=end_idx:
                progress.setValue(3 + (frame_idx-start_idx)*67.0/(end_idx-start_idx))
                if progress.wasCanceled():
                    return
                if resize_rate != 1 or rect != img_rect:
                    frame = cv2.resize(frame, resize_size)[save_rect[1]:save_rect[3],save_rect[0]:save_rect[2]]
                writer.write(frame)
                for _ in range(int(stride)):
                    ret, frame = cap.read()
                    frame_idx+=1
            writer.release()
            if reverse:
                video_reverse(save_file, save_file)
            progress.setValue(80)
            if progress.wasCanceled():
                return
            if append_file:
                if os.path.exists(append_file):
                    tmp_file = 'tmp'
                    while os.path.exists(tmp_file+'.avi'):
                        tmp_file +='_1'
                    tmp_file+='.avi'
                    os.rename(save_file, tmp_file)
                    video_combine(tmp_file,append_file,save_file)
                    os.remove(tmp_file)
                else:
                    QMessageBox.warning(self, 'Append failed', 'Append file %s not exists.'%append_file)
                    return
            progress.setValue(90)
            if progress.wasCanceled():
                return
            if insert_file:
                if os.path.exists(insert_file):
                    tmp_file = 'tmp'
                    while os.path.exists(tmp_file+'.avi'):
                        tmp_file +='_1'
                    tmp_file+='.avi'
                    os.rename(save_file, tmp_file)
                    video_compose(tmp_file,insert_file,save_file)
                    os.remove(tmp_file)
                else:
                    QMessageBox.warning(self, 'Insert failed', 'Insert file %s not exists.'%insert_file)
                    return
            # saved
            progress.setValue(100)
            QMessageBox.information(self, 'Save success', 'Save success to %s.'%save_file)

    def grab_img(self):
        if not self.open_video:
            QMessageBox.warning(self, "Grab failed", "No video opened.")
            return
        save_dir = QFileDialog.getExistingDirectory(self, "Open Save Dir", self.init_open_dir())
        if save_dir:
            _, stride, resize_rate, resize_size, start_idx, end_idx, rect, _, _, _ = self.parsing_options()
            progress = QProgressDialog(self)
            progress.setWindowTitle("Video Save")  
            progress.setLabelText("Process video and saving...                ")
            progress.setMinimumDuration(1)
            progress.setWindowModality(Qt.WindowModal)
            progress.setFixedWidth(600)
            progress.setRange(0,100)
            cap = cv2.VideoCapture(self.video_file)
            for frame_idx in range(start_idx+1):
                ret, frame = cap.read()
            save_rect = (0,0)+resize_size
            img_rect = (0,0,self.width,self.height)
            if rect != img_rect:
                save_rect = tuple([int(x*resize_rate) for x in rect])
            save_cnt = 0
            while ret and frame_idx<=end_idx:
                progress.setValue(5 + (frame_idx-start_idx)*95.0/(end_idx-start_idx))
                if progress.wasCanceled():
                    return
                if resize_rate != 1 or rect != img_rect:
                    frame = cv2.resize(frame, resize_size)[save_rect[1]:save_rect[3],save_rect[0]:save_rect[2]]
                cv2.imwrite(os.path.join(save_dir, '%06d.jpg'%save_cnt),frame)
                save_cnt+=1
                for _ in range(int(stride)):
                    ret, frame = cap.read()
                    frame_idx+=1
            progress.setValue(100)
            QMessageBox.information(self, 'Save success', 'Save success to %s.'%save_dir)

    def load_append(self):
        video_file = QFileDialog.getOpenFileName(self, "Open Append Video", self.init_open_dir(),"Video Files (*.avi *.mp4)")[0]
        if video_file:
            self.appendEdit.setText(video_file)

    def load_insert(self):
        video_file = QFileDialog.getOpenFileName(self, "Open Insert Video", self.init_open_dir(),"Video Files (*.avi *.mp4)")[0]
        if video_file:
            self.insertEdit.setText(video_file)

    def downsample_slide(self):
        stride = self.strideSlider.value()
        self.strideEdit.setText(str(stride))

    def resize_slide(self):
        val = self.resizeRateSlider.value()
        self.resizeRateEdit.setText(str(2**val))

    def set_resize_rate(self):
        if self.resizeCheckBox.isChecked() and self.resizeRateEdit.text():
            rate = float(self.resizeRateEdit.text())
            self.resizeSizeEdit.setText('%d x %d'%(int(self.width*rate),int(self.height*rate)))

    def cut(self):
        if self.cutCheckBox.isChecked():
            QMessageBox.information(self, 'Video cut', 'Shortcut:\n - b: select begin frame\n - e: select end frame\n'
                                       + ' - ctrl+left mouse button: crop video\n - s: save configuration')
            start_idx, end_idx, cut_rect = video_cut(self.video_file)
            self.startIdxEdit.setText(str(start_idx))
            self.endIdxEdit.setText(str(end_idx))
            self.xMinEdit.setText(str(cut_rect[0]))
            self.xMaxEdit.setText(str(cut_rect[2]))
            self.yMinEdit.setText(str(cut_rect[1]))
            self.yMaxEdit.setText(str(cut_rect[3]))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoToolGUI()
    sys.exit(app.exec_())