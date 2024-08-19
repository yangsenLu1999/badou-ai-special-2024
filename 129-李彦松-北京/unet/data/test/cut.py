import os

def delete_images_with_two_res(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            if filename.count("_res") == 2:
                os.remove(os.path.join(directory, filename))
                print(f"Deleted {filename}")

# 获取当前脚本的路径
script_path = os.path.dirname(os.path.realpath(__file__))
# 使用方法：删除与脚本同名的文件夹中文件名包含两个"_res"的所有图片
delete_images_with_two_res(script_path)
