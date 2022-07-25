import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = '5' #使用GPU 5

print(tf.__version__)


def read_data():
    com_path = '/home/zlc/PycharmWorkSpace/GestureSoli/Data/'
    data_time = '200422'
    data_length = 'short'
    who_without_list = [
        "dzh",
        # "ZLC",
        # "PMC",
        # "WYH",
        # "MZD",
        # "YT",
        # "ZC",
        # "DWY",
        # "FS",
        # "GHL",
        # "HJZ",
        # "LK",
        # "LL",
        # "LPZ",
        # "LQX",
        # "LRZ",
        # "LYR",
        # "MZ",
        # "QSQ",
        # "TQM",
        # "WKN",
        # "ZJH",
        # "ZJW",
        # "ZLN",
        # "ZXY",
        # "ZYF",
        # "ZYS",
        # "ZYT",
        # "ZYX",
        # "ZZY",
        # "YYF",
        # "MZY",
        # "FYH",
        # "QKY",
        #                    "allppl",
    ]

    for iteration_value in range(len(who_without_list)):
        str_temp = who_without_list[iteration_value]
        x_norm = np.loadtxt(com_path + "/0512MobileTwirlData.txt")
        # x_norm = case_switch(np.loadtxt(
        #     com_path + str_temp + "/0FirstAnglegesture_" + str_temp + "_" + data_length + "_" + data_time + ".txt"), 0, str_temp)
        if str_temp == "dzh":
            print(str_temp)
            x_all = x_norm
        else:
            print(str_temp)
            x_all = np.vstack((x_all,x_norm))
    print(x_all.shape)
    for iteration_value in range(len(who_without_list)):
        str_temp = who_without_list[iteration_value]
        y_norm = np.loadtxt(com_path + "/0512MobileTwirlLabel.txt")
        # y_norm = case_switch(np.loadtxt(
        #     com_path + str_temp + "/0feature_" + str_temp + "_" + data_length + "_" + data_time + ".txt"), 1, str_temp)
        if str_temp == "dzh":
            y_all = y_norm
            print("*********************")
        else:
            y_all = np.hstack((y_all,y_norm))
    print(y_all.shape)
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=1)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

'''
   build dense layer 
        构建dense layer
    @param units:当前层的神经元个数
    @param last_units:上一层的神经元个数
    @param activation_function:激活函数
'''
def dense_layer(units,last_units,inputs,activation_function="relu"):
    W = tf.Variable(tf.truncated_normal([last_units,units],stddev = 0.001,dtype=tf.float32))
    b = tf.Variable(tf.zeros([units],dtype=tf.float32))
    logits1 = tf.matmul(inputs,W) + b
    if activation_function != None:
        activation_function = activation_function.strip().lower()
    if activation_function == "relu":
        return tf.nn.relu(logits1)
    elif activation_function == "softmax":
        return tf.nn.softmax(logits1)
    elif activation_function == "sigmoid":
        return tf.nn.sigmoid(logits1)
    else:
        #print("activation function '"+str(activation_function)+"'"+" not found, logits will be returned")
        return logits1

'''
    build ResNet block, two dense layer in one block
        构建ResNet block，2个dense layer为一个block
    @param units:神经元个数，当前层和上一层神经元个数保持一致，为了好相加
    @param inputs: 上一层的输出，本层的输入
    @param activation_function: 激活函数
'''
def resNet_block_with_two_layer(units,inputs,activation_function ="relu"):
    if activation_function == None:
        raise Exception("activation function can't be None")
    activation_function = activation_function.strip().lower()
    if activation_function != "relu" and activation_function != "sigmoid":
        raise Exception("Unsupported activation function, only 'sigmoid' or 'relu' are Candidate")
    if inputs.shape[1] != units:
        raise Exception("the rank 2 of inputs must equal units")
    d1_output = dense_layer(units, units, inputs, activation_function)
    d2_output = dense_layer(units, units, d1_output, activation_function = 'relu')
    #ResNet的体现，先相加再送入到激活函数
    #H(x) = f(x) + x,使得训练的目标f(x) = H(x) - x即残差
    d2_output = d2_output + inputs
    if activation_function == "sigmoid":
        d2_output = tf.nn.sigmoid(d2_output)
    else:
        d2_output = tf.nn.relu(d2_output)
    return d2_output


class SimpleResNet(object):
    '''
        @param input_dim: 数据的维度 / the dimension of data
        @param num_classes: 类别个数 / number of classes
        @param units: ResNet block的神经元个数 / number of neurons in ResNet block
        @param num_resNet_blocks: ResNet block 的个数 / number of ResNet block
        @param lr: 学习率  / learning rate
        @param activation_function: resNet block中的激活函数 / activation function in ResNet block
    '''

    # 初始化模型参数
    def __init__(self, input_dim, num_classes, units, num_resNet_blocks, lr, activation_function="relu"):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.units = units
        self.num_resNet_blocks = num_resNet_blocks
        self.lr = lr
        self.sess = tf.InteractiveSession()
        self.activation_function = activation_function
        self.build_inputs()
        self.build_model()

    # 设置模型的输入参数
    def build_inputs(self):
        # 设置输入的x，shape:[batch_size,dim]
        self.x_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 1, 220], name="x_inputs")
        self.x_inputs = tf.reshape(self.x_inputs, [-1, 220])

        # 设置输入的y，shape:[batch_size]
        self.y_inputs = tf.placeholder(dtype=tf.int32, shape=[None], name="y_inputs")

        # 设置验证集的x，shape:[batch_size,dim]
        self.x_val = tf.placeholder(dtype=tf.float32, shape=[None, 1, 220])
        self.x_val = tf.reshape(self.x_val, [-1, 220])

        # 设置验证集的y，shape:[batch_size]
        self.y_val = tf.placeholder(dtype=tf.int32, shape=[None])

    def build_resNet_blocks(self, inputs):
        inputs_ = inputs
        for _ in range(self.num_resNet_blocks):
            inputs_ = resNet_block_with_two_layer(self.units, inputs_, self.activation_function)
        return inputs_

    def build_model(self):

        #输入
        d1_output = dense_layer(units = self.units, last_units = self.input_dim, inputs = self.x_inputs, activation_function = "relu")
        #构建n个ResNet block
        outputs_blocks = self.build_resNet_blocks(d1_output)
        #构建输出层，softmax
        y_logits = dense_layer(units = self.num_classes, last_units = self.units, inputs = outputs_blocks, activation_function = "None")
        print(y_logits)
        ##############################################
        y_preds = tf.nn.softmax(y_logits, name="gesture_output")
        self.preds = y_preds
        # tf.Print(y_preds)
        # 找出概率最大的类别
        self.y_round = tf.argmax(y_preds, axis=1)
        self.y_round = tf.cast(self.y_round, dtype=tf.int32)
        y_comp = tf.equal(self.y_inputs, self.y_round)

        # 计算train_acc
        self.acc = tf.reduce_mean(tf.cast(y_comp, dtype=tf.float32))

        # 构建train_loss，优化器
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_inputs, logits=y_logits)
        self.loss = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

    # 训练模型
    def train(self, X_train, y_train, X_test, y_test, epochs):
        self.sess.run(tf.global_variables_initializer())
        Best_acc = 0  # 记录最高的验证集准确率
        epoch_thresh_acc = 30  # 从第几个epoch开始保存最优的模型

        lastLoss = 0  # 上一次训练epoch的Loss值
        time_not_desend = 0  # 连续几个epochLoss不下降
        time_thresh_lr = 5  # 连续几次Loss不下降则降低学习率
        min_lr_thresh = 0.00001  # 最小将学习率缩减到多少
        time_thresh_exit = 10  # 连续几次Loss不下降则退出训练

        for i in range(epochs):
            # 训练过程
            train_loss, train_acc, _ = self.sess.run([self.loss, self.acc, self.train_op],
                                                     feed_dict={self.x_inputs: X_train, self.y_inputs: y_train})

            # if语句用来控制多久输出一次训练Log
            if i != -1:
                print('')
                # 使用验证集进行验证
                y_pred, val_acc, softmaxResult, val_loss = model.accuracy(X_test, y_test, i)
                print("epoch:" + str(i + 1), "/" + str(epochs), ":\ntrain_loss:", train_loss, " train_acc:", train_acc,
                      " val_loss:", val_loss, " val_acc", val_acc)
                print('')

            # 保存准确度最高的模型（PB格式）
            if val_acc > Best_acc:
                Best_acc = val_acc
                if (i + 1 > epoch_thresh_acc and val_acc > 0.9):
                    filePath = 'epoch' + str(i + 1) + '_acc' + str(val_acc) + '.pb'
                    self.saveModelPB(filePath)

            # 连续几次Loss不下降则降低学习率
            if i == 0:
                lastLoss = train_loss
            if lastLoss <= train_loss:
                time_not_desend = time_not_desend + 1
            else:
                time_not_desend = 0
            lastLoss = train_loss
            # 降低学习率
            if (time_not_desend == 5 and self.lr > min_lr_thresh):
                print("The learning rate is setting to ", self.lr)
                self.lr = self.lr * 0.5

            # 连续几次Loss不下降则退出训练
            if (time_not_desend == time_thresh_exit):
                print("More than ", time_not_desend, " times the loss value did not desend, exit training.")
                break

    # 使用模型仅进行推断
    def inference(self, X_test):
        y_pred = self.sess.run([self.y_round], feed_dict={self.x_inputs: X_test})
        return y_pred

    # 使用模型进行推断并计算准确率
    def accuracy(self, x_test, y_test, epoch):
        y_pred, acc_test, softmaxResult, val_loss = self.sess.run([self.y_round, self.acc, self.preds, self.loss],
                                                                  feed_dict={self.x_inputs: x_test,
                                                                             self.y_inputs: y_test})
        #进行冗余手势的判断,如果不进行6分类我们现在先将其注释掉
        #accFilter, accNoFilter, accRedundedFiltered, miss = self.redundedGestureProcessing(softmaxResult, y_test)

        return y_pred, acc_test, softmaxResult, val_loss

    # 对冗余手势进行处理
    def redundedGestureProcessing(self, tempResult, y_test):
        numAllGesture = len(y_test)  # 总待分类的手势数量
        redundedGesture = 0  # 待分类手势中冗余手势的数量
        correctResult = 0  # 最终正确判断的手势数量
        wrongResult = 0  # 最终错误判断的手势数量
        redundedResult = 0  # 正常手势中被判断为冗余手势的数量
        redundedGestureBeFound = 0  # 冗余手势中成功被识别为冗余手势的数量
        for i in range(0, numAllGesture):
            predictResult = np.argmax(tempResult[i])
            if (y_test[i] == 5):
                redundedGesture = redundedGesture + 1
                if (predictResult == y_test[i]):
                    redundedGestureBeFound = redundedGestureBeFound + 1
            elif (tempResult[i][5] > 0.8 or tempResult[i][predictResult] < 0.15):
                redundedResult = redundedResult + 1
            elif (predictResult == y_test[i]):
                correctResult = correctResult + 1
            else:
                wrongResult = wrongResult + 1
        valiadeGesture = numAllGesture - redundedGesture  # 正常的手势的数量（不包括冗余）
        valiadeFilteredGesture = valiadeGesture - redundedResult  # 正常的手势被过滤后的数量
        accFilter = correctResult / valiadeFilteredGesture  # 正常的手势被过滤后的准确率
        miss = redundedResult / valiadeGesture  # 正常手势被错认为冗余的概率
        accNoFilter = correctResult / valiadeGesture  # 如果没有过滤机制,正常手势的准确率
        if (redundedGesture != 0):
            accRedundedFiltered = redundedGestureBeFound / redundedGesture  # 冗余手势被识别为冗余的概率
        else:
            accRedundedFiltered = 1

        print('accFilter(正常的手势被过滤后的准确率）:{}'.format(accFilter))
        print('accNoFilter(如果没有过滤机制,正常手势的准确率):{}'.format(accNoFilter))
        print('accRedundedFiltered(冗余手势被识别为冗余的概率）:{}'.format(accRedundedFiltered))
        print('miss（正常手势被认为冗余的概率）:{}'.format(miss))
        return accFilter, accNoFilter, accRedundedFiltered, miss

    # 保存模型为ckpt模式
    def savemodel(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "model.ckpt")
        print("Model saved in path {}".format(save_path))

    # 保存模型为PB模式
    def saveModelPB(self, filePath):
        # 保存二进制模型
        output_graph_def = tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def,
                                                                        output_node_names=['gesture_output'])
        with tf.gfile.FastGFile(filePath, mode='wb') as f:
            f.write(output_graph_def.SerializeToString())


#labels代表训练数据使用哪些标签，传入None使用全部0~9
x_train,y_train,x_val,y_val = read_data()
#以下四行可以给训练集或验证集加入新数据
# x_old = np.loadtxt(open("/content/drive/My Drive/ColabData/Gesture_Lab_Data/dzh/x_all.csv","rb"), delimiter=",", skiprows=0)
# y_old = np.loadtxt(open("/content/drive/My Drive/ColabData/Gesture_Lab_Data/dzh/y_all.csv","rb"), delimiter=",", skiprows=0)
# x_train = np.vstack((x_train,x_old))
# y_train = np.hstack((y_train,y_old))

#设置初始学习率
lr = 0.00001
#设置训练epoch数量
epochs = 500
#设置分类的数量
num_classes = 2


#num_resNet_blocks*2+2 层
num_resNet_blocks = 10

#生成模型
model = SimpleResNet(input_dim = 220, num_classes = num_classes, units = 512, num_resNet_blocks = num_resNet_blocks, lr = lr, activation_function = "relu")

#训练模型
model.train(x_train, y_train, x_val, y_val, epochs = epochs)

#保存模型为pb格式，此处不再调用，在训练过程中会保存准确率最高的
#model.saveModelPB('convNetMobileData0511.pb')
