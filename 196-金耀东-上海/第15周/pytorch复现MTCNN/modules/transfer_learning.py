import torch

def load_pretrained_weight(model, path_pretrained_model, device):
    # ------------------------------------------------------#
    #   根据预训练权重的shape和模型的shape对应进行加载
    #   要求：模型的结构与预训练模型相同
    # ------------------------------------------------------#
    pretrained_dict = torch.load(path_pretrained_model, map_location=device)
    model_dict = model.state_dict()
    keys = list( model_dict.keys() )

    update_dict = {}
    num_model_key, i = len(keys), 0
    for pretrained_key, pretrained_value in pretrained_dict.items():
        if not i < num_model_key:
            # 参数加载完毕
            break

        if pretrained_dict[pretrained_key].shape == model_dict[keys[i]].shape:
            # 形状相同，参数匹配成功
            update_dict[keys[i]] = pretrained_value
        else:
            return False
        i+=1

    # 更新模型参数字典
    model_dict.update(update_dict)
    #加载模型参数
    model.load_state_dict(model_dict)
    return True
