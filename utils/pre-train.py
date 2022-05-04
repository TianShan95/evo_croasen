# pre-train predictor
import numpy as np
from acc_predictor.factory import get_acc_predictor
from search_space.myofa import OFASearchSpace
import json
import os
import torch

search_space = OFASearchSpace()


def _fit_acc_predictor(archive, save_dir):
    inputs = np.array([search_space.encode(x[0]) for x in archive])  # arch samples
    targets = np.array([x[1] for x in archive])  # acc
    # assert len(inputs) > len(inputs[0]), "# of training samples have to be > # of dimensions"

    acc_predictor = get_acc_predictor('mlp', inputs, targets)  # input是模型 targets是top-1错误率

    # 保存预测模型的参数
    torch.save(acc_predictor.model.state_dict(), save_dir)

    return acc_predictor, acc_predictor.predict(inputs)  # 返回精度预测器 和 预测器预测出来的精度值


def _resume_from_dir(resume):
    """ resume search from a previous iteration """
    import glob

    archive = []
    for file in glob.glob(os.path.join(resume, "net_*_subnet.txt")):
        arch = json.load(open(file))
        pre, ext = os.path.splitext(file)
        stats = json.load(open(pre + ".stats"))
        archive.append((arch, 100 - stats['top1']))

    return archive

if __name__ == '__main__':
    save_model_dict_path = '../predictor_dict.pth'
    archive = _resume_from_dir('../.tmp_01/iter_0')
    _fit_acc_predictor(archive, save_model_dict_path)