import os
import shutil

"""
通过清洗后的数据list文件名，挑选出原图像数据，用于白内障分级
"""

def move_select_data_to_target(origin, target, save_path):
    origin_list = os.listdir(origin)

    for i in range(len(origin_list)):
        img_name = origin_list[i]
        src_img = os.path.join(target, img_name)
        target_img = os.path.join(save_path, img_name)

        if os.path.exists(src_img):
            shutil.move(src_img, target_img)
            print("Move " + img_name)

if __name__ == '__main__':
    origin_data = "H:\Release\output\out_LGS_N_crop"   # 筛查后的分割结果图像路径
    target_data = "H:\Release\output\out_LGS_N_origin"    # 原始数据路径
    save_path = "H:\origin_LGS_N"      # 保存的目标路径

    move_select_data_to_target(origin_data, target_data, save_path)