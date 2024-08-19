import torch.nn as nn
'''
构件网络模型的几种方法
torch下两种子模块：nn模型训练部分，function指激活函数等部分
'''
# 方法1：
model = nn.Sequential()
model.add_module('fc1', nn.Linear(3, 4))
model.add_module('fc2', nn.Linear(4, 2))
model.add_module('out', nn.Softmax(2))

# 方法2：
model2 = nn.Sequential(nn.Conv2d(1, 20, 5),
                       nn.ReLU(),
                       nn.Conv2d(20, 64, 5),
                       nn.ReLU()
                        )

# 方法3
model3 = nn.ModuleList([nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)])
