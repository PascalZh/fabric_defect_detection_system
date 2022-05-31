# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'app/realtime_main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_RealtimeMainWindow(object):
    def setupUi(self, RealtimeMainWindow):
        RealtimeMainWindow.setObjectName("RealtimeMainWindow")
        RealtimeMainWindow.resize(1200, 800)
        self.centralwidget = QtWidgets.QWidget(RealtimeMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(50, 30, 1011, 441))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(24)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox_Camera = QtWidgets.QGroupBox(self.layoutWidget)
        self.groupBox_Camera.setObjectName("groupBox_Camera")
        self.label_Video = QtWidgets.QLabel(self.groupBox_Camera)
        self.label_Video.setGeometry(QtCore.QRect(0, 20, 57, 15))
        self.label_Video.setText("")
        self.label_Video.setObjectName("label_Video")
        self.horizontalLayout.addWidget(self.groupBox_Camera)
        self.groupBox_Detection = QtWidgets.QGroupBox(self.layoutWidget)
        self.groupBox_Detection.setObjectName("groupBox_Detection")
        self.label_Image = QtWidgets.QLabel(self.groupBox_Detection)
        self.label_Image.setGeometry(QtCore.QRect(0, 20, 57, 15))
        self.label_Image.setText("")
        self.label_Image.setObjectName("label_Image")
        self.horizontalLayout.addWidget(self.groupBox_Detection)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(810, 510, 201, 171))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_OpenCamera = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_OpenCamera.setMinimumSize(QtCore.QSize(180, 0))
        self.pushButton_OpenCamera.setObjectName("pushButton_OpenCamera")
        self.verticalLayout.addWidget(self.pushButton_OpenCamera)
        self.pushButton_CloseCamera = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_CloseCamera.setObjectName("pushButton_CloseCamera")
        self.verticalLayout.addWidget(self.pushButton_CloseCamera)
        self.pushButton_Capture = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_Capture.setMinimumSize(QtCore.QSize(180, 0))
        self.pushButton_Capture.setObjectName("pushButton_Capture")
        self.verticalLayout.addWidget(self.pushButton_Capture)
        self.label_Status = QtWidgets.QLabel(self.centralwidget)
        self.label_Status.setGeometry(QtCore.QRect(130, 580, 300, 15))
        self.label_Status.setText("")
        self.label_Status.setObjectName("label_Status")
        self.pushButton_LookUpHistories = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_LookUpHistories.setGeometry(QtCore.QRect(100, 680, 131, 31))
        self.pushButton_LookUpHistories.setObjectName("pushButton_LookUpHistories")
        RealtimeMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(RealtimeMainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 20))
        self.menubar.setObjectName("menubar")
        RealtimeMainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(RealtimeMainWindow)
        self.statusbar.setObjectName("statusbar")
        RealtimeMainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(RealtimeMainWindow)
        QtCore.QMetaObject.connectSlotsByName(RealtimeMainWindow)

    def retranslateUi(self, RealtimeMainWindow):
        _translate = QtCore.QCoreApplication.translate
        RealtimeMainWindow.setWindowTitle(_translate("RealtimeMainWindow", "Realtime Fabric Defect Detection"))
        self.groupBox_Camera.setTitle(_translate("RealtimeMainWindow", "Camera"))
        self.groupBox_Detection.setTitle(_translate("RealtimeMainWindow", "Detection"))
        self.pushButton_OpenCamera.setText(_translate("RealtimeMainWindow", "Open Camera"))
        self.pushButton_CloseCamera.setText(_translate("RealtimeMainWindow", "Close Camera"))
        self.pushButton_Capture.setText(_translate("RealtimeMainWindow", "Capture"))
        self.pushButton_LookUpHistories.setText(_translate("RealtimeMainWindow", "Lookup Histories"))
