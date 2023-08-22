import os, sys
from .mnist import MamlMnist


class MamlKMnist(MamlMnist):
    mirrors = ["http://codh.rois.ac.jp/kmnist/dataset/kmnist/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "bdb82020997e1d708af4cf47b453dcf7"),
        ("train-labels-idx1-ubyte.gz", "e144d726b3acfaa3e44228e80efcd344"),
        ("t10k-images-idx3-ubyte.gz", "5c965bf0a639b31b8f53240b1b52f4d7"),
        ("t10k-labels-idx1-ubyte.gz", "7320c461ea6c1c855c0b718fb2a4b134"),
    ]
    classes = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]