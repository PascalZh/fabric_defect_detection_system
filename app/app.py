import sys
import os
import time
import cv2
from PyQt5.QtGui import QImage, QPixmap, QStandardItem
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QDialog, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import QTimer
from realtime_main_window import *
from histories_dialog import *
from efficient_net_predict import predict
from database import insert_image, get_all_images


class Dialog_Histories(QDialog, Ui_Dialog_Histories):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.lookup_histories()
        self.tableWidget.itemSelectionChanged.connect(self.show_image)
        self.rows = []

    def lookup_histories(self):
        rows = get_all_images()
        self.rows = rows
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setRowCount(len(rows))
        self.tableWidget.setHorizontalHeaderLabels(
            ['ID', 'Image Name', 'Datetime'])
        self.tableWidget.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tableWidget.setSelectionBehavior(QTableWidget.SelectRows)
        self.tableWidget.setSelectionMode(QTableWidget.SingleSelection)
        row_idx = 0
        for id, image_name, _, datetime in rows:
            # print(id)
            # print(image_name)
            # print(datetime)
            for col_idx, item in enumerate([str(id), image_name, datetime]):
                self.tableWidget.setItem(
                    row_idx, col_idx, QTableWidgetItem(item))
            row_idx += 1

    def show_image(self):
        if len(self.tableWidget.selectedItems()) == 0:
            return
        image = QImage()
        # print(self.tableWidget.selectedIndexes()[0].data())
        image.loadFromData(
            self.rows[self.tableWidget.selectedIndexes()[0].row()][2], format='JPG')
        self.label_Image.setPixmap(QPixmap.fromImage(image))
        self.label_Image.adjustSize()


class RealtimeMainWindow(QMainWindow, Ui_RealtimeMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.pushButton_OpenCamera.clicked.connect(self.open_camera)
        self.pushButton_CloseCamera.clicked.connect(self.close_camera)
        self.pushButton_Capture.clicked.connect(self.capture)
        self.pushButton_LookUpHistories.clicked.connect(
            self.open_histories_dialog)
        self.histories_dialog = Dialog_Histories(self)

        self.image_path = './tmp/captured.jpg'

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_video)

    def open_camera(self):
        self.timer.start(30)

    def close_camera(self):
        self.timer.stop()

    def show_video(self):
        os.system('libcamera-jpeg --width 600 --height 600 --nopreview -o '+self.image_path)
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(
            image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        self.label_Video.setPixmap(QPixmap.fromImage(image))
        self.label_Video.adjustSize()

    def capture(self):
        # capture image and run neural network model to predict
        # show image
        image = QImage(self.image_path)
        if image.isNull():
            QMessageBox.information(
                self, 'Capture Error', 'Cannot open file %s.' % os.path.abspath(self.image_path))
            return False

        self.label_Image.setPixmap(QPixmap.fromImage(image))
        self.label_Image.adjustSize()

        # eval model
        self.label_Status.setText('Evaluating...')
        res, _ = predict(self.image_path)

        # show prediction results in the gui
        self.label_Status.setText('Detection result: ' + res)

        # save image to database if the result is defection
        image.save('./tmp/tmp.jpg')
        insert_image('./tmp/tmp.jpg')

    def open_histories_dialog(self):
        self.histories_dialog.lookup_histories()
        self.histories_dialog.show()


if __name__ == "__main__":
    os.makedirs('tmp', exist_ok=True)

    app = QApplication(sys.argv)
    mainWin = RealtimeMainWindow()
    mainWin.show()
    sys.exit(app.exec_())
