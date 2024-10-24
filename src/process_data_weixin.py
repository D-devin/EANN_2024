# encoding=utf-8
import pickle as pickle
import random
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
from collections import defaultdict
import sys, re
import pandas as pd
import math
from types import *
from gensim.models import Word2Vec
import jieba
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os.path
import gensim
import cv2

os.chdir(os.path.dirname(os.path.abspath(__file__)))
## 清洗（洗去标点，分词，去掉停用词），图片转化，文本与图像匹配，合并文本获取词频率，获取词向量将这次数据集的词向量加入到其中
## 最后返回数据集的data，将新的词向量保存。
def stopwords(path=r'E:\fakenews\EANN_2024\Data\weixin\cn_stopwords.txt'):
    stopwords = []
    for line in open(path, 'r').readlines():
        stopwords.append(line.strip())
    return stopwords
  
def ssl_clean(string):
    if isinstance(string, float):
        #print(string)
        string = "无"
        return string

    # 定义一个包含所有要清除的特殊字符的字符类
    special_chars = u"[，。：；.`|“”——_/&;@、?？''’‘“”《》～（）#！%【】:,+(｜)丨●▶…! ✅㊙҈↑↓🤣\s*nbsp-]"
    # 使用re.sub()函数和字符类来清除所有特殊字符
    string = re.sub(special_chars, "", string)
    # 去除字符串两端的空白字符（注意：这里不处理中间的空白字符）
    return string.strip()


def update_image_url(output_df, images_dir):
    # 创建 DataFrame 的副本以避免修改原始数据
    for index, row in output_df.iterrows():
        id_value = row['id']
        # 构造相对路径
        relative_image_path = os.path.join(os.path.basename(images_dir), f'{id_value}.png')
        # 检查图片文件是否存在
        image_file = os.path.join(images_dir, f'{id_value}.png')
        if not os.path.exists(image_file):
            print(f"Warning: Image file for id {id_value} does not exist at {image_file}")
            relative_image_path = 'images/missing.png'  # 占位符图片路径

        # 更新 DataFrame 中的图片路径
        output_df.at[index, 'Image Url'] = relative_image_path

    return output_df

def check_content_for_errors(text):
    # 定义无法读取的关键词列表
    error_keywords = [
        "账号 屏蔽", "内容 无法 查看", "内容发布者 删除", "微信 公众 平台 运营 中心",
        "账号 迁移", "公众号 环境异常"
    ]
    # 检查文本中是否包含任何关键词
    for keyword in error_keywords:
        if keyword in text:
            return "1"
    return "2"

def images_process (image_url):
    image_list = []
    for path in image_url:
        image_name = os.path.splitext(path)[0]
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])])
        try:
            im = cv2.imread(image_name)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = data_transforms(im)
            image_list.append(im) 
        except:
            print("err_image_name:"+image_name)
            print("image_path:"+path)
    print("image_length:"+str(len(image_list)))
    return image_list

def get_w(model, word_text,index, k):
        # vocab_size = len(word_vecs)
    word_index_map= dict()
    W = np.zeros(shape=(len(index) + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_text:
        W[i] = model[word]
        word_index_map[word] = i
        i += 1
    return W,word_index_map

def text_to_word2vec(save_path,all_text,min_df = 5,vector_size = 100):
    """
    将词表转化为词向量
    然后保存词向量表为文件
    采用gensim读取
    加入的新词表的最低频率为1
    min_count是最低出现数，默认数值是5；
    size是gensim Word2Vec将词汇映射到的N维空间的维度数量（N）默认的size数是100；
    iter是模型训练时在整个训练语料库上的迭代次数，假如参与训练的文本量较少，就需要把这个参数调大一些。iter的默认值为5；
    sg是模型训练所采用的的算法类型：1 代表 skip-gram，0代表 CBOW，sg的默认值为0；
    window控制窗口，如果设得较小，那么模型学习到的是词汇间的组合性关系（词性相异）；如果设置得较大，会学习到词汇之间的聚合性关系（词性相同）。模型默认的window数值为5；
    """
    #trian
    word_vec = gensim.models.Word2Vec(all_text, vector_size=100,min_count=min_df,window = 5,sg = 0)
    index = word_vec.wv.index_to_key

    #val

    word_vec.save(save_path+'word2vec.bin')
    return index 

def word_cab(data_clear):
    all_text = []
    vocab = {}
    texts = data_clear['Title clear'] + data_clear['News Url clear'] + data_clear['Report Content clear']
            # 将词汇集合转换为排序后的列表，并创建词汇到索引的映射
    for text in texts:
                #print(i,"\n")
        setence = ''
        if isinstance(text, str):  # 检查文本是否为字符串（假设每段文本是字符串）
            for word in text.split(): # 将词汇列表添加到all_text中
                vocab[word] = vocab.get(word, 0) + 1
                setence = setence + word+' '


        else:
            print(type(text))
            raise ValueError(f"Text at index  is not a string: {text}")
        all_text.append(setence)

    return vocab, all_text
   
def get_data(path):
    """
    数据预处理，对每个数据表的每列进行去除标点符号，jieba分词，形成数据1
    然后合集这些词汇，并且统计词汇频率，形成词频率表
    最后返回两个csv的path，和全文本。
    dataframe列[id,标题，正文，图片路径，评论，判断标签，clear标题，clearn正文，clearn评论]
    图片处理是直接为路径
    """
    os.chdir(path)
    print(os.getcwd())
    # 需要清洁的列表
    clear_index = ['Title', 'News Url', 'Report Content']
    data_csv = pd.read_csv(path+r'\origin_train.csv', encoding='utf-8')
    # 复制一个副本方便做处理避免表格过大
    data_clear = data_csv.copy()
    # 对指定列进行清洗
    for i in clear_index:
        #print(i)
        #print(data_clear[i])
        data_clear[f'{i} clear'] = data_clear[i].apply(lambda x: ssl_clean(x))
        data_clear[f'{i} clear'] = data_clear[f'{i} clear'].apply(lambda x: ' '.join(jieba.cut_for_search(x)))
    #打上标签，是否正确读取news url
    data_clear['news_tag'] = data_clear['News Url'].apply(check_content_for_errors)
    # 更改image url 为path
    images_directory = path+r'\train\image'
    data_clear = update_image_url(data_clear,images_directory)
    #图片目前怎么处理不知道
    #images = images_process(data_clear['image_path'].tolist())
    #data_clear['image'] = images
    """
    目前暂定这样等会，看看哪些地方有问题的地方等会再改
    """
    # 合并词汇做词汇表和全文本
    word_to_ix, all_text = word_cab(data_clear)
    print(type(all_text))
    # 随机划分训练集和测试集
    val_ratio = 0.3
    total_samples = len(data_clear)
    initial_val_samples = int(total_samples * val_ratio)
    # 打乱索引
    indices = list(range(data_clear.shape[0]))
    random.shuffle(indices)
    # 筛选符合news_tag条件的样本索引
    filtered_indices = [i for i in indices if data_clear.loc[i, 'news_tag'] in [1, 2]]
    val_samples = int(len(filtered_indices) * val_ratio)
    train_samples = len(filtered_indices) - val_samples
    # 选择验证集训练集样本
    val_indices = filtered_indices[:val_samples]
    train_indices_initial = filtered_indices[val_samples:]
    half_val_indices = val_indices[:val_samples // 2]
    train_indices_final = train_indices_initial + half_val_indices

    # 根据索引划分数据集
    train_data = data_csv.loc[train_indices_final]
    val_data = data_csv.loc[val_indices[val_samples // 2:]]  # 验证集剩下的一半

    train_data.to_csv('train.csv', index=False)
    val_data.to_csv('val.csv', index=False)

    return train_data, val_data, word_to_ix, all_text
       
def read_data(text_only,min_df,path = "null"):
    """
    训练加载data用
    先进行调用get_data进行读取文件分出训练集和测试集，返回训练集和测试集的参数情况，以及文件路径，
    在这个函数内加载这个读取文件。
    然后调用做分别做两个全文合集丢入词向量函数得到词向量表，返回并且词向量参数情况

    """
        #text_only = False

    if text_only:
        print("Text only")
        image_list = []
    else:
        print("Text and image")
    print("loding data...")    
    train_data,val_data,word_cab,all_text=get_data(path)

    #输出参数情况
    print("number of sentences: " + str(len(all_text)))
    print("vocab size: " + str(len(word_cab)))
    max_l = len(max(all_text, key=len))
    print("max sentence length: " + str(max_l))    
    #加载词向量
    save_path = r'E:\fakenews\EANN_2024\Data\weixin'
    print("word2vec loaded!")
    index = text_to_word2vec(save_path,all_text)
    return train_data,val_data,all_text,word_cab
   
  
read_data(False,30,path= r'E:\fakenews\EANN_2024\Data\weixin')