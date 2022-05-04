import json
import os
import re
import subprocess
import shutil

import future.builtins.disabled


def _resume_from_dir(resume):
    """ resume search from a previous iteration """
    import glob

    archive = []
    for file in glob.glob(os.path.join(resume, "net_*_subnet.txt")):
        arch = json.load(open(file))
        # pre, ext = os.path.splitext(file)
        # stats = json.load(open(pre + ".txt"))
        try:
            stats = json.load(open(file.replace('subnet', 'stats')))
        except FileNotFoundError:
            continue

        archive.append((arch, 100 - stats['acc']*100, file))

    return archive

if __name__ == '__main__':
    archive = _resume_from_dir('../.tmp_01/iter_0')
    # 把个体 按照精度从大到小排列
    archive.sort(key=lambda x: float(x[1]))
    # 选出精度最高的 前 60 个个体 存入 .tmp/iter_0/
    os.makedirs('../.tmp/iter_0', exist_ok=True)
    for indivi_index, indivi in enumerate(archive[:60]):
        # subprocess.call(f'cp {indivi[2]} ../.tmp/iter_0/')
        shutil.copyfile(indivi[2], f'../.tmp/iter_0/{os.path.basename(indivi[2])}')
        # subprocess.call(f'cp {indivi[2].replace("subnet", "stats")} ../.tmp/iter_0/')
        shutil.copyfile(indivi[2].replace("subnet", "stats"), f'../.tmp/iter_0/{os.path.basename(indivi[2].replace("subnet", "stats"))}')