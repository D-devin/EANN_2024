import numpy as np
import argparse
import time, os

from triton import TritonError

import process_data_weixin as process_data
import copy
from random import sample
import torchvision
from sklearn.model_selection import train_test_split
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import scipy.io as sio
from gensim.models import Word2Vec
from torchvision import datasets, models, transforms
import gensim
import pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Rumor_Data(Dataset):
    #数据转换类转换为张量
    def __init__(self, dataset):
        self.text = torch.from_numpy(np.array(dataset['post_text_index'].tolist()))
        #self.social_context = torch.from_numpy(np.array(dataset['social_feature']))
        self.mask = torch.from_numpy(np.array(dataset['mask'].tolist()))
        self.label = torch.from_numpy(np.array(dataset['label'].tolist()))
        self.event_label = torch.from_numpy(np.array(dataset['news_tag'].tolist()))
        print('TEXT: %d, labe: %d, Event: %d'
               % (len(self.text), len(self.label), len(self.event_label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.text[idx], self.mask[idx]), self.label[idx], self.event_label[idx]


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 返回的梯度数量应与输入数量相匹配
        return grad_output, None  # 如果只有一个输入，第二个返回 None

def grad_reverse(x, lambd):
    return ReverseLayerF.apply(x, lambd)


# Neural Network Model (1 hidden layer)
class CNN_Fusion(nn.Module):
    def __init__(self, args, W):
        super(CNN_Fusion, self).__init__()
        self.args = args

        self.event_num = args.event_num

        vocab_size = args.vocab_size
        emb_dim = args.embed_dim

        C = args.class_num
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim
        self.social_size = 19

        # TEXT RNN
        #文本处理的循环网络
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(W))
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
        # 双向lstm
        self.lstm_fc = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.text_encoder = nn.Linear(emb_dim, self.hidden_size)

        ### TEXT CNN
        channel_in = 1
        filter_num = 20
        window_size = [1, 2, 3, 4]
        self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, emb_dim)) for K in window_size])
        self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size)

        self.dropout = nn.Dropout(args.dropout)

        #IMAGE
        #图像处理的VGG19
        #hidden_size = args.hidden_dim
        vgg_19 = torchvision.models.vgg19(pretrained=True)
        for param in vgg_19.parameters():
            param.requires_grad = False
        # visual model
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vgg = vgg_19
        self.image_fc1 = nn.Linear(num_ftrs,  self.hidden_size)
        #self.image_fc2 = nn.Linear(512, self.hidden_size)
        self.image_adv = nn.Linear(self.hidden_size,  int(self.hidden_size))
        self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size)

        ###social context
        #分类器
        self.social = nn.Linear(self.social_size, self.hidden_size)

        ##ATTENTION
        #注意力机制（吧
        self.attention_layer = nn.Linear(self.hidden_size, emb_dim)

        ## Class  Classifier
        #分类器
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1',  nn.Linear(self.hidden_size, 2))
        #self.class_classifier.add_module('c_bn1', nn.BatchNorm2d(100))
        #self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        #self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        #self.class_classifier.add_module('c_fc2', nn.Linear(self.hidden_size, 2))
        #self.class_classifier.add_module('c_bn2', nn.BatchNorm2d(self.hidden_size))
        #self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        #self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        ###Event Classifier
        #事件分类器
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.hidden_size, self.hidden_size))
        #self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

        ####Image and Text Classifier
        #图像和文本分类器
        self.modal_classifier = nn.Sequential()
        self.modal_classifier.add_module('m_fc1', nn.Linear(self.hidden_size, self.hidden_size))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.modal_classifier.add_module('m_relu1', nn.LeakyReLU(True))
        self.modal_classifier.add_module('m_fc2', nn.Linear(self.hidden_size, 2))
        self.modal_classifier.add_module('m_softmax', nn.Softmax(dim=1))


    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (to_var(torch.zeros(1, batch_size, self.lstm_size)),
                to_var(torch.zeros(1, batch_size, self.lstm_size)))

    def conv_and_pool(self, x, conv):
        #核与池
        x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        #x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, text, mask):
        
        #########CNN##################
        text = self.embed(text)
        text = text * mask.unsqueeze(2).expand_as(text)
        text = text.unsqueeze(1)
        text = [F.relu(conv(text)).squeeze(3) for conv in self.convs]  # [(N,hidden_dim,W), ...]*len(window_size)
        #text = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,hidden_dim), ...]*len(window_size)
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = torch.cat(text, 1)
        text = F.relu(self.fc1(text))
        #text = self.dropout(text)

        ### Class
        #class_output = self.class_classifier(text_image)
        class_output = self.class_classifier(text)
        ## Domain
        reverse_feature = grad_reverse(text,self.args.lambd)
        domain_output = self.domain_classifier(reverse_feature)
     
        return class_output, domain_output

#显卡检验
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

#转数组
def to_np(x):
    return x.data.cpu().numpy()
#选择器
def select(train, selec_indices):
    temp = []
    for i in range(len(train)):
        print("length is "+str(len(train[i])))
        print(i)
        #print(train[i])
        ele = list(train[i])
        temp.append([ele[i] for i in selec_indices])
    return temp

def make_weights_for_balanced_classes(event, nclasses = 15):
    count = [0] * nclasses
    for item in event:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(event)
    for idx, val in enumerate(event):
        weight[idx] = weight_per_class[val]
    return weight



def split_train_validation(train, percent):
    whole_len = len(train[0])

    train_indices = (sample(range(whole_len), int(whole_len * percent)))
    train_data = select(train, train_indices)
    print("train data size is "+ str(len(train[3])))
    # print()

    validation = select(train, np.delete(range(len(train[0])), train_indices))
    print("validation size is "+ str(len(validation[3])))
    print("train and validation data set has been splited")

    return train_data, validation





def main(args):
    print('loading data')
    #    dataset = DiabetesDataset(root=args.training_file)
    #    train_loader = DataLoader(dataset=dataset,
    #                              batch_size=32,
    #                              shuffle=True,
    #                              num_workers=2)
    #载入数据
    # MNIST Dataset
    train, validation, W = load_data(args)

    #train, validation = split_train_validation(train,  1)

    #weights = make_weights_for_balanced_classes(train[-1], 15)
    #weights = torch.DoubleTensor(weights)
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))


    train_dataset = Rumor_Data(train)

    validate_dataset = Rumor_Data(validation)

    #test_dataset = Rumor_Data(test) # not used



    #数据读取类
    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

    validate_loader = DataLoader(dataset = validate_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)


    print('building model')
    model = CNN_Fusion(args, W)

    if torch.cuda.is_available():
        print("CUDA")
        model.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    #参数设置
    #Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                                 lr= args.learning_rate)
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, list(model.parameters())),
                                  #lr=args.learning_rate)
    #scheduler = StepLR(optimizer, step_size= 10, gamma= 1)

    #最佳设置
    iter_per_epoch = len(train_loader)
    print("loader size " + str(len(train_loader)))
    best_validate_acc = 0.000
    best_loss = 100
    best_validate_dir = ''

    print('training model')
    adversarial = True
    # Train the Model
    #训练模型
    for epoch in range(args.num_epochs):

        p = float(epoch) / 100
        #lambd = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 0.001
        #参数加载
        optimizer.lr = lr
        #rgs.lambd = lambd

        start_time = time.time()
        cost_vector = []
        class_cost_vector = []
        domain_cost_vector = []
        acc_vector = []
        valid_acc_vector = []
        test_acc_vector = []
        vali_cost_vector = []
        test_cost_vector = []
        #读取训练集的数据，数据文本，标签，事件标签，0是文本，1是图片
        for i, (train_data, train_labels, event_labels) in enumerate(train_loader):
            train_text,  train_mask, train_labels, event_labels = \
                to_var(train_data[0]),  to_var(train_data[1]), \
                to_var(train_labels), to_var(event_labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            #获得类输出
            class_outputs, domain_outputs = model(train_text, train_mask)
            # ones = torch.ones(text_output.size(0))
            # ones_label = to_var(ones.type(torch.LongTensor))
            # zeros = torch.zeros(image_output.size(0))
            # zeros_label = to_var(zeros.type(torch.LongTensor))

            #modal_loss = criterion(text_output, ones_label)+ criterion(image_output, zeros_label)
            train_labels = train_labels.long()
            class_loss = criterion(class_outputs, train_labels)
            event_labels = event_labels.long()
            domain_loss = criterion(domain_outputs, event_labels)
            #做验证
            loss = class_loss + domain_loss
            loss.backward()
            optimizer.step()
            _, argmax = torch.max(class_outputs, 1)

            cross_entropy = True
            #计算正确率
            if True:
                accuracy = (train_labels == argmax.squeeze()).float().mean()
            else:
                _, labels = torch.max(train_labels, 1)
                accuracy = (labels.squeeze() == argmax.squeeze()).float().mean()
            #每轮加进去
            class_cost_vector.append(class_loss.item())
            #domain_cost_vector.append(domain_loss.data[0])
            cost_vector.append(loss.item())
            acc_vector.append(accuracy.item())
            # if i == 0:
            #     train_score = to_np(class_outputs.squeeze())
            #     train_pred = to_np(argmax.squeeze())
            #     train_true = to_np(train_labels.squeeze())
            # else:
            #     class_score = np.concatenate((train_score, to_np(class_outputs.squeeze())), axis=0)
            #     train_pred = np.concatenate((train_pred, to_np(argmax.squeeze())), axis=0)
            #     train_true = np.concatenate((train_true, to_np(train_labels.squeeze())), axis=0)

        #验证模式  
        model.eval()
        validate_acc_vector_temp = []
        #读取验证集的数据，数据文本，标签，事件标签，0是文本，1是图
        for i, (validate_data, validate_labels, event_labels) in enumerate(validate_loader):
            validate_text,  validate_mask, validate_labels, event_labels = \
                to_var(validate_data[0]),  to_var(validate_data[1]), \
                to_var(validate_labels), to_var(event_labels)
            #与上文同理
            validate_outputs, domain_outputs = model(validate_text, validate_mask)
            #同理
            _, validate_argmax = torch.max(validate_outputs, 1)
            validate_labels = validate_labels.long()
            vali_loss = criterion(validate_outputs, validate_labels)
            #domain_loss = criterion(domain_outputs, event_labels)
                #_, labels = torch.max(validate_labels, 1)
            validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            vali_cost_vector.append(vali_loss.item())
                #validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            validate_acc_vector_temp.append(validate_accuracy.item())
        validate_acc = np.mean(validate_acc_vector_temp)
        valid_acc_vector.append(validate_acc)
        model.train()
        #训练模式
        print ('Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, validate loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
                % (
                epoch + 1, args.num_epochs,  np.mean(cost_vector), np.mean(class_cost_vector), np.mean(vali_cost_vector),
                    np.mean(acc_vector),   validate_acc, ))

        #保存模型
        if validate_acc > best_validate_acc:
            best_validate_acc = validate_acc
            if not os.path.exists(args.output_file):
                os.mkdir(args.output_file)
            best_validate_dir = args.output_file + str(epoch + 1) + '_text.pkl'
            torch.save(model.state_dict(), best_validate_dir)

    duration = time.time() - start_time
    # 测试
    # Test the Model
    print('testing model')
    model = CNN_Fusion(args, W)
    model.load_state_dict(torch.load(best_validate_dir))
    #    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    test_score = []
    test_pred = []
    test_true = []
    # 读取测试集的数据，数据文本，标签，事件标签，0是文本，1是图
    for i, (test_data, test_labels, event_labels) in enumerate(validate_loader):
        test_text, test_mask, test_labels = to_var(
            test_data[0]), to_var(test_data[1]), to_var(test_labels)
        test_outputs, _ = model(test_text, test_mask)
        _, test_argmax = torch.max(test_outputs, 1)
        if i == 0:
            test_score = to_np(test_outputs.squeeze())
            test_pred = to_np(test_argmax.squeeze())
            test_true = to_np(test_labels.squeeze())
        else:
            test_score = np.concatenate((test_score, to_np(test_outputs.squeeze())), axis=0)
            test_pred = np.concatenate((test_pred, to_np(test_argmax.squeeze())), axis=0)
            test_true = np.concatenate((test_true, to_np(test_labels.squeeze())), axis=0)

    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    test_precision = metrics.precision_score(test_true, test_pred, average='macro')
    test_recall = metrics.recall_score(test_true, test_pred, average='macro')

    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')

    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

    print("Classification Acc: %.4f, AUC-ROC: %.4f"
          % (test_accuracy, test_aucroc))
    print("Classification report:\n%s\n"
          % (metrics.classification_report(test_true, test_pred, digits=3)))
    print("Classification confusion matrix:\n%s\n"
          % (test_confusion_matrix))

    print('Saving results')


    duration = time.time() - start_time
    #print ('Epoch: %d, Mean_Cost: %.4f, Duration: %.4f, Mean_Train_Acc: %.4f, Mean_Test_Acc: %.4f'
           #% (epoch + 1, np.mean(cost_vector), duration, np.mean(acc_vector), np.mean(test_acc_vector)))
#best_validate_dir = args.output_file + 'baseline_text_weibo_GPU2_out.' + str(20) + '.pkl'

    print('Saving results')
   

#参数设置函数
def parse_arguments(parser):
    parser.add_argument('training_file', type=str, metavar='<training_file>', help='')
    parser.add_argument('validation_file', type=str, metavar='<validation_file>', help='')
    #parser.add_argument('testing_file', type=str, metavar='<testing_file>', help='')
    parser.add_argument('output_file', type=str, metavar='<output_file>', help='')

    parse.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default = 32, help='')
    parser.add_argument('--embed_dim', type=int, default=100, help='')# 我们每个词向量为100，修改为100
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--filter_num', type=int, default=20, help='')
    parser.add_argument('--lambd', type=int, default= 1, help='')
    parser.add_argument('--text_only', type=bool, default= True, help='')

    #    parser.add_argument('--sequence_length', type = int, default = 28, help = '')
    #    parser.add_argument('--input_size', type = int, default = 28, help = '')
    #    parser.add_argument('--hidden_size', type = int, default = 128, help = '')
    #    parser.add_argument('--num_layers', type = int, default = 2, help = '')
    #    parser.add_argument('--num_classes', type = int, default = 10, help = '')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=100, help='')
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')
    return parser


#词向量转换函数
def word2vec(post, cab, index, sequence_length):
    texts = post['Title clear'] + post['News Url clear'] + post['Report Content clear']
    text_index = []
    mask = []
    discarded_indices = []  # 用于记录丢弃的索引

    for idx, sentence in enumerate(texts):  # 使用 enumerate 获取索引
        sen_index = []
        if isinstance(sentence, float) or not sentence.strip():  # 检查是否为空句子
            discarded_indices.append(idx)  # 记录丢弃的索引
            continue  # 直接跳过
        else:
            sentence = sentence.split()
            mask_seq = np.zeros(sequence_length, dtype=np.float32)
            mask_seq[:len(sentence)] = 1.0
            for word in sentence:
                if word in cab:
                    word_index = index[word]
                else:
                    word_index = 0
                sen_index.append(word_index)
            # 填充至 sequence_len
            while len(sen_index) < sequence_length:
                sen_index.append(0)  # 用 0 填充
            sen_index = sen_index[:sequence_length]  # 截断至 sequence_len
            text_index.append(sen_index)
            mask.append(mask_seq)

    return text_index, mask, discarded_indices  # 返回丢弃的索引




def load_data(args):
    #加载数据，调用隔壁函数加载的是dataframe
    train_path, val_path,word2_path = process_data.read_data(args.text_only,32,path= '../Data/weixin')
    #print(train[4][0])
    #读取词向量
    #train_path = r'C:\Users\yjpan\Desktop\EANN_2024\Data\weixin\train.csv'
    #val_path = r'C:\Users\yjpan\Desktop\EANN_2024\Data\weixin\val.csv'
    #word2_path = r'C:\Users\yjpan\Desktop\EANN_2024\Data\weixin\word2vec.model'
    train = pd.read_csv(train_path,encoding = 'utf-8')
    val = pd.read_csv(val_path,encoding='utf-8')
    model = gensim.models.Word2Vec.load(word2_path)
    cab= model.wv.index_to_key
    index = model.wv.key_to_index
    vectors = model.wv.vectors
    args.vocab_size = len(cab)
    #sequence_len自己看着改，我这里写的100！
    args.sequence_len = 30
    print("translate data to embedding")
    #数据集与词向量的转换
    #读取验证集

    val_text_index, val_mask,discarded_indices = word2vec(val,cab,index,args.sequence_len)
    val.drop(index = discarded_indices,errors='ignore',inplace=True)
    val['post_text_index'] = val_text_index
    val['mask'] = val_mask
    print("Val Data Size is "+str(len(val['post_text_index'])))

    #读取训练集
    train_text_index, train_mask,discarded_indices = word2vec(train,cab,index,args.sequence_len)
    train.drop(index = discarded_indices,errors = 'ignore',inplace=True)
    train['post_text_index'] = train_text_index
    train['mask'] = train_mask
    print("sequence length " + str(args.sequence_length))
    print("Train Data Size is "+str(len(train['post_text_index'])))
    print("Finished loading data ")

    return train, val, vectors

def transform(event):
    matrix = np.zeros([len(event), max(event) + 1])
    #print("Translate  shape is " + str(matrix))
    for i, l in enumerate(event):
        matrix[i, l] = 1.00
    return matrix

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    train = '../Data/weixin/train.csv'
    val = '../Data/weixin/train.csv'
    output = '../Data/weibo/output/'
    args = parser.parse_args([train, val, output])
    #    print(args)
    main(args)

