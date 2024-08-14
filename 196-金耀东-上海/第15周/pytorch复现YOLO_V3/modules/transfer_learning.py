from modules.CONST import *
import numpy as np
import torch

def load_pretrained_model(model, path_pretraind_model, show_details=True):
    # ------------------------------------------------------#
    #   根据预训练权重的shape和模型的shape对应进行加载
    #   要求：模型的结构与预训练模型相同
    # ------------------------------------------------------#
    model_dict = model.state_dict()
    pretrained_model_dict = torch.load(path_pretraind_model, map_location=DEVICE)
    model_keys = list(model_dict.keys())

    load_key, no_load_key, update_dict= [], [], {}
    i = 0
    for pretrained_key, pretrained_value in pretrained_model_dict.items():
        if model_keys[i].find("num_batches_tracked") != -1 and pretrained_key.find("num_batches_tracked") == -1:
            # 如果模型中含有"num_batches_tracked"字段参数，而预训练模型中没有，则跳过模型的"num_batches_tracked"字段参数
            i+=1
        elif pretrained_key.find("num_batches_tracked") == -1 and pretrained_key.find("num_batches_tracked") != -1:
            # 如果模型中没有"num_batches_tracked"字段参数，而预训练模型中含有，则跳过预训练模型的"num_batches_tracked"字段参数
            no_load_key.append(pretrained_key)
            continue

        if np.shape(model_dict[ model_keys[i] ]) == np.shape(pretrained_value):
            update_dict[ model_keys[i] ] = pretrained_value
            # model_dict[ model_keys[i] ].copy_(pretrained_value)
            load_key.append(pretrained_key)
        else:
            no_load_key.append(pretrained_key)
        i+=1
    # 更新模型参数字典
    model_dict.update(update_dict)
    # 将更新完的模型参数字典加载回模型
    model.load_state_dict(model_dict)

    # 显示没有加载成功的权重
    if show_details:
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))