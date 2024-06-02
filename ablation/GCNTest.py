import torch
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn import metrics as ms  

class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c, graph,n_classes):
        super(GCN, self).__init__()
        self.linear_1 = nn.Linear(in_c, hid_c)   #8  #8
        self.linear_2 = nn.Linear(hid_c, out_c)  #8  #1
        self.linear_3 = nn.Linear(12, n_classes)

        self.act = nn.ReLU()

        self.graph_data = graph
        self.n_classes = n_classes

    def forward(self, X):
        graph_data = GCN.process_graph(self.graph_data)

        flow_x = X  

        B, N = flow_x.size(0), flow_x.size(1)
     
        flow_x = flow_x.view(B, N, -1)  
    
        output_1 = self.linear_1(flow_x)  # [B, N, hid_C]


        output_1 = torch.matmul(graph_data, output_1)


        output_1 = self.act(output_1)





        #A^ * Y1 * W
        output_2 = self.linear_2(output_1)
        output_2 = torch.matmul(graph_data, output_2)#降维
        # [12,8] w=[8,1] w就是之前训练的


        # [B, N, 1, Out_C]  #激活函数
        # Y1 = 64个样本 12节点 1维度特征


        output_2 = output_2.view(output_2.size(0), -1) # 原论文1 * 307  等效于我们64 *12 把最后两维合并
        #然后64*12  *  W3(12*9) 九类病症
        #one-hot :64 * 9   0 0 0 0 0 0 0 0 0独热编码


        # 得出结果再softmax
        output_3 = self.linear_3(output_2)

        return torch.softmax(output_3,dim=1)

    @staticmethod
    def process_graph(graph_data):
        N = graph_data.size(0)
        # 对邻接矩阵处理变成A帽
        matrix_i = torch.eye(N, dtype=graph_data.dtype, device=graph_data.device)
        graph_data += matrix_i  # A~ [N, N]

        degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)  # [N]
        degree_matrix = degree_matrix.pow(-1)
        degree_matrix[degree_matrix == float("inf")] = 0.  # [N]

        degree_matrix = torch.diag(degree_matrix)  # [N, N]
        #原老代码 把邻接矩阵进行处理
        return torch.mm(degree_matrix, graph_data)  # D^(-1) * A = \hat(A)

#自定义交叉熵损失
class MyCrossEntropy(nn.Module):
    def  __init__(self):
        super(MyCrossEntropy, self).__init__()

    def forward(self, x, y):
        P_i = torch.nn.functional.softmax(x, dim=1)
        loss = y * torch.log(P_i + 0.0000001)
        loss = -torch.mean(torch.sum(loss, dim=1), dim=0)
        return loss

#归一化
def normalize(X):
    # 归一化
    X_nor = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1)
    return X_nor


if __name__ == '__main__':
    # graph = np.zeros([12,12])
    #A 邻接矩阵
    graph = np.array([[0,1,1,1,0,0,0,0,1,0,0,0],
                      [1,0,1,1,1,0,0,0,0,1,0,0],
                      [1,1,0,1,1,1,0,0,0,0,1,0],
                      [0,1,1,0,1,1,1,0,0,0,0,1],
                      [0,0,1,1,0,1,1,1,0,0,1,0],
                      [0,0,0,1,1,0,1,1,0,1,0,0],
                      [0,0,0,0,1,1,0,1,1,0,0,0],
                      [0,0,0,0,0,1,1,0,0,1,0,0],
                      [1,0,0,0,0,0,0,0,0,1,0,1],
                      [0,1,1,0,1,0,0,0,1,0,1,0],
                      [0,0,1,1,0,1,0,0,0,1,0,1],
                      [1,0,1,0,1,1,0,0,1,0,1,0]])
    #胸导联之间（前六个关联分为正极负极电流，在邻接矩阵中我认为其之间存在关联）
    #四肢导联分为外导联和内导联，外导联分加压肢体是是AVR导联、AVF导联和AVL导联，内导联与胸导联相关
    #根据物理位置进行关联
    graph[np.eye(12,dtype=bool)] = 0
    graph = torch.from_numpy(np.float32(graph))

    #模型的输入
    X_r = np.load('PeMS_04/pred_nnet_r.npy')
    X_t = np.load('PeMS_04/pred_nnet_t.npy')
    X_y = np.load('PeMS_04/pred_nnet_v.npy')

    # 新加入数据
    x_r_combined = np.load("PeMS_04/man_features_r.npy")
    x_t_combined = np.load("PeMS_04/man_features_t.npy")
    x_v_combined = np.load("PeMS_04/man_features_v.npy")

    # 清除NAN和空格
    x_r_combined[np.isnan(x_r_combined)] = 0
    x_t_combined[np.isnan(x_t_combined)] = 0
    x_v_combined[np.isnan(x_v_combined)] = 0

    # 归一化
    x_r_combined = normalize(x_r_combined[:, :10])
    x_t_combined = normalize(x_t_combined[:, :10])
    x_v_combined = normalize(x_v_combined[:, :10])

    #添加空列并转换
    x_r_combined = (np.concatenate((x_r_combined,np.zeros((x_r_combined.shape[0],12 - x_r_combined.shape[1]))),axis = 1)).reshape(-1,12,1,1)
    x_t_combined = (np.concatenate((x_t_combined,np.zeros((x_t_combined.shape[0],12 - x_t_combined.shape[1]))),axis = 1)).reshape(-1,12,1,1)
    x_v_combined = (np.concatenate((x_v_combined,np.zeros((x_v_combined.shape[0],12 - x_v_combined.shape[1]))),axis = 1)).reshape(-1,12,1,1)

    # 导入数据
    label_r = np.load('PeMS_04/lb_r.npy')  # 0 ,1 ,2, 3,4 ....
    label_t = np.load('PeMS_04/lb_t.npy')  # 0 ,1 ,2, 3,4 ....
    label_v = np.load('PeMS_04/lb_v.npy')  # 0 ,1 ,2, 3,4 ....
    print(X_r.shape)
    #对输入完成的数据进行处理，把原来的96维分成12x8符合邻接矩阵模型
    #（数据量，通道，特征，标签）
    #train_x = X_r.reshape(8388,12,8,1)
    train_x = np.concatenate((X_r.reshape(8388, 12, 8, 1), x_r_combined), axis=2)

    print(train_x.shape)
    ohe = OneHotEncoder()
    ohe.fit(label_r.reshape(-1, 1))
    train_y = ohe.transform(label_r.reshape(-1, 1)).toarray()
    train_y = torch.from_numpy(np.float32(train_y))

    # train_y = label_r.reshape(8388,12,1,1)
    print(train_y.shape)
    train_x = torch.from_numpy(np.float32(train_x))
    x_r_combined = torch.from_numpy(np.float32(x_r_combined))

    batch_size = 64
    epochs = 10
    model = GCN(in_c=9, hid_c=24, out_c=1, graph=graph,n_classes=train_y.shape[1])
    print(model)
    device = torch.device("cpu")

    # 损失函数
    #弄两个交叉熵
    loss_function = nn.CrossEntropyLoss()
    loss_function2 = nn.BCELoss()

    # 优化器
    #优化模型的参数
    optimizer = torch.optim.Adam(model.parameters(),lr= 0.01)
    # 开始训练
    print(X_r.shape)
    print(label_r.reshape(-1, 1).shape)
    torch_dataset = Data.TensorDataset(train_x, train_y)  # 得到一个元组(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,  # 每次训练打乱数据， 默认为False
        num_workers=2,  # 使用多进行程读取数据， 默认0，为不使用多进程
    )
    #循环
    for epoch in range(epochs):
        for step, (batch_x, batch_y) in enumerate(loader):
            # 1.8000  优化1次
            # 2.1     优化8000次
            # 3.64    8000/64次
            prd = model(batch_x)
            #预测y和真实y的区别
            #梯度下降，先算损失，再将损失反向传播，给到正向传播
            # loss = loss_function(prd,batch_y)
            loss2 = loss_function2(prd,batch_y)
            optimizer.zero_grad()
            #反向传播
            loss2.backward()
            #优化 用偏导数来算，优化下参数W
            optimizer.step()
            if(step%50==0):
                print(epoch,step, "loss:", loss2.data)

    # t部分
    X_t = np.load('PeMS_04/pred_nnet_t.npy')
    test_x = np.concatenate((X_t.reshape(-1, 12, 8, 1), x_t_combined), axis=2)
    test_x = torch.from_numpy(np.float32(test_x))

    # 测试部分
    label_t = np.load('PeMS_04/lb_t.npy')
    ohe = OneHotEncoder()
    ohe.fit(label_t.reshape(-1, 1))
    test_y = ohe.transform(label_t.reshape(-1, 1)).toarray()
    test_y = torch.from_numpy(np.float32(test_y))

    y_pred = model(test_x)
    print((test_y.argmax(axis=1) == y_pred.argmax(axis=1)).sum() / len(test_y))
    y_pred = y_pred.argmax(axis=1)
    test_y = test_y.argmax(axis=1)

    print("准确率为:{0:%}".format(ms.accuracy_score(test_y, y_pred)))
    print("精确率为:{0:%}".format(ms.precision_score(test_y, y_pred, average='macro')))
    print("召回率为:{0:%}".format(ms.recall_score(test_y, y_pred, average='macro')))
    print("F1分数为:{0:%}".format(ms.f1_score(test_y, y_pred, average='macro')))
    print("Fbeta为:{0:%}".format(ms.fbeta_score(test_y, y_pred, beta=1.2, average='macro')))
    # prd = model(train_x)
    # print(prd.shape)