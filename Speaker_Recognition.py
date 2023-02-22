import os
import sys
import typing

import numpy as np
import PySide2
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from Audio import Audio
from Model import Model
from Profile import Profile
from UI import Ui_Form


class MyWidget(QWidget):
    record_signal = Signal(int)
    save_signal = Signal(str)
    infer_signal = Signal(type(np.array))
    recognize_signal = Signal(type(np.array))
    enroll_signal = Signal(type(np.array), str)
    delete_signal = Signal(str)

    def __init__(self,
                 parent: typing.Optional[PySide2.QtWidgets.QWidget] = ...,
                 f: PySide2.QtCore.Qt.WindowFlags = ...) -> None:
        super().__init__(parent, f)

        self.setup_ui()
        # stable indicates signal that always connect
        self.setup_stable_signal()
        # unstable indicates signal that will be disconnected
        self.setup_unstable_signal()
        # prevent disconnecting error
        self.operate("connect")

        self.normal()

    def setup_ui(self):
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.recognize_text = self.ui.button_recognize.text()
        self.enroll_text = self.ui.button_enroll.text()
        self.delete_text = self.ui.button_delete.text()
        self.auto_text = self.ui.button_auto.text()
        self.cancel_text = "取消"
        self.keyword = "你好，小P"

        self.ui.input_username.setPlaceholderText("在此输入用户名")
        self.ui.display_debug.setPlaceholderText("调试信息，用户可忽略。")
        self.ui.display_result.setPlaceholderText("在此显示识别结果。")
        self.ui.display_result.document().setMaximumBlockCount(10)

        self.delete_query = QMessageBox(self)
        self.delete_query.setIcon(QMessageBox.Question)
        self.delete_query.setInformativeText("是否删除该用户？")
        self.delete_query.setStandardButtons(QMessageBox.Ok
                                             | QMessageBox.Cancel)
        self.delete_query.setDefaultButton(QMessageBox.Ok)

    def setup_stable_signal(self):
        self.vad_mode = 3
        self.audio_dir = "Audio"
        self.enroll_duration = 6
        self.enr_rec_ratio = 3
        self.recognize_duration = self.enroll_duration // self.enr_rec_ratio

        modelname = "ECAPA_TDNN_GLOB_c512-ASTP-emb192-ArcMargin-LM"
        for config in modelname.split('-'):
            if "emb" in config:
                self.embeding_size = int(config[3:])

        self.model_path = os.path.join("Model", f"{modelname}.onnx")
        self.profile_dir = os.path.join("Profile", f"{modelname}")

        # ---- rate <= 0.001 -----
        # thres = 0.17244 @ fnr = 0.0009573449633017764, fpr = 0.1182944335158701
        # thres = 0.45377 @ fpr = 0.0009569886756339852, fnr = 0.13397510903095416
        # ---- rate <= 0.0001 -----
        # thres = 0.06691 @ fnr = 5.318583129454313e-05, fpr = 0.3009197724493593
        # thres = 0.49913 @ fpr = 5.316603753524607e-05, fnr = 0.21981704074034678
        # ---- rate <= 0 -----
        # thres = 0.05753 @ fnr = 0.0, fpr = 0.32356850443936414
        # thres = 0.53752 @ fpr = 0.0, fnr = 0.3117753430486119
        # ---- vox1_O_cleaned.kaldi.score -----
        # EER = 1.972
        # thres = 0.31796
        # minDCF (p_target:0.01 c_miss:1 c_fa:1) = 0.194
        self.recognize_threshold = 0.45377
        self.save_threshold = 0.53752

        # audio
        self.audio_thread = QThread()
        self.audio = Audio(None, self.vad_mode)
        self.audio.moveToThread(self.audio_thread)
        self.audio_thread.start()
        self.audio_thread.finished.connect(self.audio_thread.deleteLater)

        self.record_signal.connect(self.audio.record)
        self.save_signal.connect(self.audio.save)

        # model
        self.model_thread = QThread()
        self.model = Model(None, self.model_path, self.audio.sample_rate)
        self.model.moveToThread(self.model_thread)
        self.model_thread.start()
        self.model_thread.finished.connect(self.model_thread.deleteLater)

        self.infer_signal.connect(self.model.infer)

        # profile
        self.profile = Profile(self, self.profile_dir, self.embeding_size)

        self.recognize_signal.connect(self.profile.recognize)
        self.enroll_signal.connect(self.profile.enroll)
        self.delete_signal.connect(self.profile.delete)

        # timer
        self.prompt_timer = QTimer(self)
        self.prompt_interval = 10 * 1000
        self.prompt_timer.timeout.connect(self.prompt_timeout)
        self.prompt_timer.start(self.prompt_interval)

        self.prompt_texts = []
        self.prompt_texts.append(f"点击“{self.recognize_text}”按键，即可进行识别，" +
                                 f"识别需要录制{self.recognize_duration}秒音频。")
        self.prompt_texts.append("点击左下角菜单，选择附件，选择Keyboard，即可使用虚拟键盘，" +
                                 f"输入用户名后，关闭虚拟键盘，再回到本界面。")
        self.prompt_texts.append(f"点击“{self.enroll_text}”按键之前，需要先在右下方输入用户名，"
                                 f"注册需要录制{self.enroll_duration}秒音频，")
        self.prompt_texts.append(f"识别和注册建议说一样的话，比如“{self.keyword}”，" +
                                 f"已注册的用户可以继续注册其他常用语。")
        self.prompt_texts.append(f"点击“{self.delete_text}”按键之前，需要先在右下方输入用户名，" +
                                 f"只能删除已注册的用户。")
        self.prompt_texts.append(f"勾选“{self.auto_text}”按键，即可持续识别，" +
                                 f"反选恢复手动模式。")
        self.prompt_texts.append(f"要中途退出识别、注册或自动模式，点击“{self.cancel_text}”按键即可。")

        self.prompt_count = 0
        self.ui.display_prompt.setPlaceholderText(
            self.prompt_texts[self.prompt_count])

        # button
        self.ui.button_recognize.clicked.connect(self.recognize)
        self.ui.button_enroll.clicked.connect(self.enroll)
        self.ui.button_auto.clicked.connect(self.auto)
        # prevent disconnecting error
        self.ui.button_delete.clicked.connect(self.cancel)

    def setup_unstable_signal(self):
        self.signals = []
        self.slots = []
        # audio
        self.signals.append(self.audio.after_vad_signal)
        self.slots.append(self.after_vad)
        self.signals.append(self.audio.recorded_signal)
        self.slots.append(self.recorded)

        # model
        self.signals.append(self.model.inferred_signal)
        self.slots.append(self.inferred)

        # profile
        self.signals.append(self.profile.recognized_signal)
        self.slots.append(self.recognized)
        self.signals.append(self.profile.enrolled_signal)
        self.slots.append(self.enrolled)
        self.signals.append(self.profile.deleted_signal)
        self.slots.append(self.deleted)

    def operate(self, operation: str):
        assert operation in ["connect", "disconnect"]

        for signal, slot in zip(self.signals, self.slots):
            getattr(signal, operation)(slot)

    @Slot()
    def prompt_timeout(self):
        self.prompt_count += 1
        self.prompt_count %= len(self.prompt_texts)
        self.ui.display_prompt.setPlaceholderText(
            self.prompt_texts[self.prompt_count])

    @Slot()
    def normal(self):
        # ui
        self.ui.button_recognize.setDisabled(False)
        self.ui.button_enroll.setDisabled(False)
        self.ui.button_delete.setDisabled(False)
        self.ui.button_delete.setText(self.delete_text)

        # switch signal
        self.ui.button_delete.clicked.disconnect()
        self.ui.button_delete.clicked.connect(self.delete)

        # logic
        self.state = "normal"
        self.operate("disconnect")
        self.ui.input_username.setFocus()
        self.ui.display_prompt.clear()
        if self.ui.button_auto.isChecked():
            self.recognize()

    @Slot()
    def recognize(self):
        # ui
        self.ui.button_recognize.setDisabled(True)
        self.ui.button_enroll.setDisabled(True)
        self.ui.button_delete.setDisabled(False)
        self.ui.button_delete.setText(self.cancel_text)

        # switch signal
        self.ui.button_delete.clicked.disconnect()
        self.ui.button_delete.clicked.connect(self.cancel)

        # logic
        self.state = "recognize"
        self.operate("connect")
        self.record()

    @Slot()
    def auto(self):
        if self.ui.button_auto.isChecked():
            if "normal" == self.state:
                self.recognize()
        else:
            if "recognize" == self.state:
                self.cancel()

    @Slot()
    def enroll(self):

        def validate_username(username: str):
            for char in username:
                if "a" <= char <= "z" or "A" <= char <= "Z":
                    continue
                if "_" == char:
                    continue
                if "0" <= char <= "9":
                    continue
                return False
            return True

        # ui
        self.ui.button_recognize.setDisabled(True)
        self.ui.button_enroll.setDisabled(True)
        self.ui.button_delete.setDisabled(False)
        self.ui.button_delete.setText(self.cancel_text)

        # switch signal
        self.ui.button_delete.clicked.disconnect()
        self.ui.button_delete.clicked.connect(self.cancel)

        # logic
        self.username = self.ui.input_username.text()
        if 0 == len(self.username):
            self.ui.display_prompt.setText("请先在右下方输入用户名。")
            self.normal()
        elif not validate_username(self.username):
            self.ui.display_prompt.setText("用户名只能由字母、下划线_或数字组成。")
            self.normal()
        else:
            self.state = "enroll"
            self.operate("connect")
            self.record()

    @Slot()
    def delete(self):
        # ui
        self.ui.button_recognize.setDisabled(True)
        self.ui.button_enroll.setDisabled(True)
        self.ui.button_delete.setDisabled(True)

        # logic
        self.username = self.ui.input_username.text()
        if 0 == len(self.username):
            self.ui.display_prompt.setText("请先在右下方输入用户名。")
            self.normal()
        elif self.username not in self.profile.user_embeddings.keys():
            self.ui.display_prompt.setText(f"{self.username}，未注册。")
            self.normal()
        else:
            self.state = "delete"
            self.operate("connect")
            self.delete_query.setText(f"{self.username}，已注册。")
            ret = self.delete_query.exec_()
            if QMessageBox.Ok == ret:
                self.delete_signal.emit(self.username)
            else:
                self.normal()

    @Slot()
    def cancel(self):
        self.ui.button_auto.setChecked(False)
        self.normal()
        self.audio.running = False

    def record(self):
        if "recognize" == self.state:
            duration = self.recognize_duration
            string = f"正在录音，需要录制{duration}秒，" + f"识别和注册建议说一样的话。"
        elif "enroll" == self.state:
            duration = self.enroll_duration
            string = f"正在录音，需要录制{duration}秒，"
            if self.username in self.profile.user_embeddings.keys():
                string += f"可以说其他常用语。"
            else:
                string += f"可以重复说“{self.keyword}”。"
        self.audio.running = True
        self.record_signal.emit(duration)

        self.ui.display_prompt.setText(string)

    @Slot()
    def after_vad(self, need_duration: float):
        self.ui.display_prompt.setText(f"还需要录制{(need_duration):.1f}秒，" +
                                       f"您有{self.audio.record_duration}秒时间。")

    @Slot()
    def recorded(self, voiced_frame: np.array):
        voiced_frames = []
        if "recognize" == self.state:
            voiced_frames.append(voiced_frame)
        elif "enroll" == self.state:
            length = len(voiced_frame) // self.enr_rec_ratio
            for i in range(self.enr_rec_ratio):
                voiced_frames.append(voiced_frame[i * length:(i + 1) * length])
            self.save_signal.emit(
                os.path.join(self.audio_dir, f"{self.username}_enroll.wav"))
        voiced_frames = np.stack(voiced_frames, axis=0)

        self.infer_signal.emit(voiced_frames)

        self.ui.display_prompt.setText("录制完成，正在推理。")

    @Slot()
    def inferred(self, embeddings: np.ndarray):
        if "recognize" == self.state:
            self.ui.display_prompt.append("推理完成，正在识别。")
            self.recognize_signal.emit(np.squeeze(embeddings, axis=0))
        elif "enroll" == self.state:
            self.ui.display_prompt.append("推理完成，正在注册。")
            self.enroll_signal.emit(embeddings, self.username)

    @Slot()
    def recognized(self, user_score_sorted: list):
        # no enrolled user
        if not user_score_sorted:
            self.ui.display_result.append("用户未注册。")
            self.save_signal.emit(os.path.join(self.audio_dir, "unknown.wav"))
            self.normal()
            return

        max_score = user_score_sorted[0][1]
        if max_score >= self.recognize_threshold:
            max_username = user_score_sorted[0][0]
            self.ui.display_result.append(f"{max_username}，您好！")

            if max_score >= self.save_threshold:
                self.save_signal.emit(
                    os.path.join(self.audio_dir, f"{max_username}_certain.wav"))
            else:
                self.save_signal.emit(
                    os.path.join(self.audio_dir,
                                 f"{max_username}_uncertain.wav"))
        else:
            self.ui.display_result.append("用户未注册。")
            self.save_signal.emit(os.path.join(self.audio_dir, "unknown.wav"))

        self.normal()

        # python helps with the out of index problem
        user_score_sorted = user_score_sorted[:10]
        string = ""
        for username, score in user_score_sorted:
            string += f"{username}: {score:.2f}, "
        string = string[:-2]
        self.ui.display_debug.setText(string)

    @Slot()
    def enrolled(self, enrolled_count: int):
        self.ui.display_result.append(
            f"{self.username}，已注册{enrolled_count}个嵌入码。")
        self.normal()

    @Slot()
    def deleted(self):
        self.ui.display_result.append(f"{self.username}，已删除。")
        self.normal()

    def closeEvent(self, event: PySide2.QtGui.QCloseEvent) -> None:
        print(f"{self} closed")

        if "normal" != self.state:
            self.cancel()

        self.audio_thread.quit()
        self.audio_thread.wait()
        self.audio.p.terminate()

        self.model_thread.quit()
        self.model_thread.wait()


if "__main__" == __name__:
    app = QApplication(sys.argv)
    window = MyWidget(None, Qt.WindowFlags())

    window.showMaximized()
    sys.exit(app.exec_())
