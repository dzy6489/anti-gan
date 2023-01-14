import os
import os.path
import xml.dom.minidom as md

path = '/home/dzy-lab/projects/TBdata/TBX11K/voc_TBX11K_MMDET/annotations_origin/'
files = os.listdir(path)  # 得到文件夹下所有文件名称

# srcfile 需要复制、移动的文件
# dstpath 目的地址

import os
import shutil
from glob import glob


def mycopyfile(srcfile, dstpath):  # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, dstpath + fname)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstpath + fname))



img_path = '/home/dzy-lab/projects/TBdata/TBX11K/TBX11K/imgs/tb/'
dstpath = '/home/dzy-lab/projects/TBdata/TBX11K/TBX11K/imgs/all_active_imgs/'
def main():
    for xmlFile in files:  # 遍历文件夹
        if not os.path.isdir(xmlFile):  # 判断是否是文件夹，不是文件夹才打开

            dom = md.parse(os.path.join(path, xmlFile))
            root = dom.documentElement
            names = root.getElementsByTagName('name')  # 对某个标签进行修改
            # print(name[0].firstChild.data)
            flag = 1
            for i in range(len(names)):
                if (names[i].firstChild.data)=='ObsoletePulmonaryTuberculosis':
                    flag = 0
            if flag:
                srcfile = img_path+xmlFile.split('.')[0]+'.png'
                mycopyfile(srcfile, dstpath)


if __name__ == '__main__':
    main()