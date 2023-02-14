import os
import shutil
import typing

import numpy as np
import PySide2
from PySide2.QtCore import QObject, Signal, Slot


class Profile(QObject):
    recognized_signal = Signal(list)
    enrolled_signal = Signal(int)
    deleted_signal = Signal()

    def __init__(self,
                 parent: typing.Optional[PySide2.QtCore.QObject] = ...,
                 file_dir: str = ...,
                 embedding_size: int = ...) -> None:
        super().__init__(parent)

        self.embedding_size = embedding_size
        self.file_dir = file_dir
        self.delete_dir = f"{self.file_dir}_Delete"
        self.load()

    def load(self):
        self.user_embeddings = {}
        self.user_profile = {}
        self.user_norm = {}

        if not os.path.exists(self.file_dir):
            os.mkdir(self.file_dir)

        for filaname in os.listdir(self.file_dir):
            with open(os.path.join(self.file_dir, filaname), 'rb') as f:
                embeddings = []
                try:
                    embeddings.append(
                        np.load(f, allow_pickle=False, fix_imports=False))
                except:
                    continue
                while embeddings[-1].size == self.embedding_size:
                    try:
                        embeddings.append(
                            np.load(f, allow_pickle=False, fix_imports=False))
                    except:
                        username = filaname.split('.')[0]
                        self.user_embeddings[username] = embeddings
                        self.user_profile[username] = np.mean(np.array(
                            embeddings, copy=False),
                                                              axis=0)
                        self.user_norm[username] = np.linalg.norm(
                            self.user_profile[username], ord=2)
                        break

    @Slot()
    def recognize(self, embedding: np.ndarray):
        """
        embedding: np.ndarray with shape [embedding_size, ]
        """
        user_score = {}
        x_norm = np.linalg.norm(embedding, ord=2)
        for username in self.user_profile.keys():
            user_score[username] = np.dot(embedding,
                                          self.user_profile[username]) / (
                                              x_norm * self.user_norm[username])
            user_score[username] = float(user_score[username])

        user_score_sorted = sorted(user_score.items(),
                                   key=lambda item: item[1],
                                   reverse=True)

        self.recognized_signal.emit(user_score_sorted)

        return user_score_sorted

    @Slot()
    def enroll(self, embeddings: np.ndarray, username: str):
        """
        embeddings: np.ndarray with shape [bs, embedding_size]
        username: user's name

        return: enrolled_count
        """
        with open(os.path.join(self.file_dir, f"{username}.npy"),
                  mode="ab") as f:
            for embedding in embeddings:
                self.user_embeddings.setdefault(username, []).append(embedding)
                np.save(f, embedding, allow_pickle=False, fix_imports=False)

        self.user_profile[username] = np.mean(np.array(
            self.user_embeddings[username], copy=False),
                                              axis=0)
        self.user_norm[username] = np.linalg.norm(self.user_profile[username],
                                                  ord=2)

        enrolled_count = len(self.user_embeddings[username])

        self.enrolled_signal.emit(enrolled_count)

        return enrolled_count

    @Slot()
    def delete(self, username: str):
        """
        username: user's name
        """
        del self.user_embeddings[username]
        del self.user_profile[username]
        del self.user_norm[username]

        if not os.path.exists(self.delete_dir):
            os.mkdir(self.delete_dir)

        shutil.move(os.path.join(self.file_dir, f"{username}.npy"),
                    os.path.join(self.delete_dir, f"{username}.npy"))

        self.deleted_signal.emit()


if "__main__" == __name__:
    profile = Profile(None)

    print("Done")
