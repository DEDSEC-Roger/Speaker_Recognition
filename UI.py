# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'UI.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import Resource_rc

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(800, 444)
        self.verticalLayout = QVBoxLayout(Form)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.widget = QWidget(Form)
        self.widget.setObjectName(u"widget")
        self.horizontalLayout_3 = QHBoxLayout(self.widget)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.display_prompt = QTextBrowser(self.widget)
        self.display_prompt.setObjectName(u"display_prompt")
        self.display_prompt.setMinimumSize(QSize(400, 240))
        font = QFont()
        font.setPointSize(18)
        self.display_prompt.setFont(font)

        self.horizontalLayout_3.addWidget(self.display_prompt)

        self.widget_2 = QWidget(self.widget)
        self.widget_2.setObjectName(u"widget_2")
        self.widget_2.setMinimumSize(QSize(150, 0))
        self.widget_2.setMaximumSize(QSize(200, 16777215))
        self.verticalLayout_2 = QVBoxLayout(self.widget_2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.display_debug = QTextBrowser(self.widget_2)
        self.display_debug.setObjectName(u"display_debug")
        font1 = QFont()
        font1.setPointSize(16)
        self.display_debug.setFont(font1)

        self.verticalLayout_2.addWidget(self.display_debug)

        self.input_username = QLineEdit(self.widget_2)
        self.input_username.setObjectName(u"input_username")
        self.input_username.setMinimumSize(QSize(0, 38))
        self.input_username.setMaximumSize(QSize(16777215, 38))
        self.input_username.setFont(font1)

        self.verticalLayout_2.addWidget(self.input_username)


        self.horizontalLayout_3.addWidget(self.widget_2)


        self.verticalLayout.addWidget(self.widget)

        self.widget_3 = QWidget(Form)
        self.widget_3.setObjectName(u"widget_3")
        self.widget_3.setMaximumSize(QSize(16777215, 150))
        self.horizontalLayout_2 = QHBoxLayout(self.widget_3)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.display_result = QTextBrowser(self.widget_3)
        self.display_result.setObjectName(u"display_result")
        self.display_result.setFont(font1)

        self.horizontalLayout_2.addWidget(self.display_result)

        self.widget_4 = QWidget(self.widget_3)
        self.widget_4.setObjectName(u"widget_4")
        self.widget_4.setMinimumSize(QSize(300, 0))
        self.widget_4.setMaximumSize(QSize(400, 16777215))
        self.gridLayout = QGridLayout(self.widget_4)
        self.gridLayout.setObjectName(u"gridLayout")
        self.button_auto = QCheckBox(self.widget_4)
        self.button_auto.setObjectName(u"button_auto")
        self.button_auto.setFont(font1)

        self.gridLayout.addWidget(self.button_auto, 1, 0, 1, 1)

        self.button_recognize = QPushButton(self.widget_4)
        self.button_recognize.setObjectName(u"button_recognize")
        self.button_recognize.setFont(font1)

        self.gridLayout.addWidget(self.button_recognize, 0, 0, 1, 1)

        self.button_enroll = QPushButton(self.widget_4)
        self.button_enroll.setObjectName(u"button_enroll")
        self.button_enroll.setFont(font1)

        self.gridLayout.addWidget(self.button_enroll, 0, 1, 1, 1)

        self.button_delete = QPushButton(self.widget_4)
        self.button_delete.setObjectName(u"button_delete")
        self.button_delete.setFont(font1)

        self.gridLayout.addWidget(self.button_delete, 1, 1, 1, 1)


        self.horizontalLayout_2.addWidget(self.widget_4)


        self.verticalLayout.addWidget(self.widget_3)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Speaker Recognition", None))
        self.button_auto.setText(QCoreApplication.translate("Form", u"\u81ea\u52a8\u6a21\u5f0f", None))
        self.button_recognize.setText(QCoreApplication.translate("Form", u"\u5f00\u59cb\u8bc6\u522b", None))
        self.button_enroll.setText(QCoreApplication.translate("Form", u"\u6ce8\u518c", None))
        self.button_delete.setText(QCoreApplication.translate("Form", u"\u5220\u9664", None))
    # retranslateUi

