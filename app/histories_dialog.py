# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'app/histories_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog_Histories(object):
    def setupUi(self, Dialog_Histories):
        Dialog_Histories.setObjectName("Dialog_Histories")
        Dialog_Histories.resize(800, 600)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog_Histories)
        self.buttonBox.setGeometry(QtCore.QRect(390, 530, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.tableWidget = QtWidgets.QTableWidget(Dialog_Histories)
        self.tableWidget.setGeometry(QtCore.QRect(30, 30, 341, 541))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.label_Image = QtWidgets.QLabel(Dialog_Histories)
        self.label_Image.setGeometry(QtCore.QRect(430, 40, 57, 15))
        self.label_Image.setText("")
        self.label_Image.setObjectName("label_Image")

        self.retranslateUi(Dialog_Histories)
        self.buttonBox.accepted.connect(Dialog_Histories.accept)
        self.buttonBox.rejected.connect(Dialog_Histories.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog_Histories)

    def retranslateUi(self, Dialog_Histories):
        _translate = QtCore.QCoreApplication.translate
        Dialog_Histories.setWindowTitle(_translate("Dialog_Histories", "Histories"))