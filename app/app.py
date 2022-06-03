import sys
import os
import time
from io import BytesIO
from picamera import PiCamera
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

        # self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_video)

    def open_camera(self):
        # self.cap.open(0)

        self.camera = PiCamera()
        self.camera.start_preview()
        time.sleep(2)
        self.timer.start(1000//30)

    def close_camera(self):
        # self.cap.release()
        self.timer.stop()
        self.camera.close()
        self.label_Video.clear()

    def show_video(self):
        self.stream = BytesIO()
        self.camera.capture(self.stream, format='jpeg')

        image = QImage()
        image.loadFromData(self.stream.getvalue(), format='jpeg')
        self.label_Video.setPixmap(QPixmap.fromImage(image))
        self.label_Video.adjustSize()

    def capture(self):
        # capture image and run neural network model to predict
        # show image
        with open(self.image_path, 'wb') as f:
            f.write(self.stream.getvalue())
        image = QImage()
        image.loadFromData(self.stream.getvalue(), format='jpeg')
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
        if res == 'defection':
            insert_image(self.image_path)

    def open_histories_dialog(self):
        self.histories_dialog.lookup_histories()
        self.histories_dialog.show()


if __name__ == "__main__":
    os.makedirs('tmp', exist_ok=True)

    app = QApplication(sys.argv)
    mainWin = RealtimeMainWindow()
    mainWin.show()
    sys.exit(app.exec_())
