# encoding=utf-8
import pickle as pickle
import random
from random import *
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
from collections import defaultdict
import sys, re
import pandas as pd
from PIL import Image
import math
from types import *
from gensim.models import Word2Vec
import jieba
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import os.path
import gensim
os.chdir(os.path.dirname(os.path.abspath(__file__)))
## 清洗（洗去标点，分词，去掉停用词），图片转化，文本与图像匹配，合并文本获取词频率，获取词向量将这次数据集的词向量加入到其中
## 最后返回数据集的data，将新的词向量保存。
def stopwords(path='NULL'):
    stopwords = []
    for line in  open(path ,'r').readline():
        stopwords.append(line)
    return stopwords
    
def ssl_clean(string):
    string = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", string)
    return string.strip().lower()
    
def images_process (image_url):
    image_list = {}
    for path in image_url:
        image_name = os.path.splitext(path)[0]
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])])
        try:
            im = Image.open(i).convert('RGB')
            im = data_transforms(im)
            image_list[image_name]= im 
        except:
            print("image_name:"+image_name)
            print("image_path:"+path)
    print("image_length:"+str(len(image_list)))
    return iamge_list
    
def get_w(model, word_text,index, k):
        # vocab_size = len(word_vecs)
    word_index_map= dict()
    W = np.zeros(shape=(len(index) + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
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
    word_vec = gensim.models.Word2Vec(all_text, vector_size=100,min_count=min_df,window = 5,sg = 1)
    index = word_vec.wv.key_to_index
    word_vec.save_word2vec_format(save_path+'word2vec.bin', binary=False)
    return index 

def word_cab(data,clear_index):
    pass
   
def get_data (path):
    """
    数据预处理，对每个数据表的每列进行去除标点符号，jieba分词，形成数据1
    然后合集这些词汇，并且统计词汇频率，形成词频率表
    最后返回两个csv的path，和全文本。
    dataframe列[id,标题，正文，图片id，评论，判断标签]
    图片处理是直接为路径，还是存储转化后的图像？
    """
    ### reading data to number 
    val_id = []
    val_true = 500
    #需要清洁的列表
    index = ['id','Title','news_text','text_consent','label','clear_text','clear_title','clear_consent','image()']
    clear_index = ['Title','news_text','text_consent']
    data_csv = pd.read_csv(path = path,encoding= 'utf-8')
    #复制一个副本方便做处理避免表格过大
    data_clear = data_csv.copy()
    for i in clear_index:
        #去除标点以及去除jieba词
        data_clear [i] = data_clear[i].apply(lambda x: ssl_clean(x))
        data_clear[i] = data_clear[i].apply(lambda x: ','.join(jieba.cut_for_search(x)))    
    #图像列表，先导入图片然后返回图片数据，最后加入到data中
    images = images_process(data_csv['image_path'].tolist())
    """
    目前暂定这样等会，看看哪些地方比较构式的地方等会再改
    """
    #合并词汇做词汇表和全文本
    word_cab,all_text = word_cab(data_clear,clear_index)

    #随机划分训练集和测试集

    return train_data,val_data,word_cab,all_text
       
def read_data(text_only,min_df):
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
    train_data,val_data,word_cab,all_text=get_data()

    #输出参数情况
    print("number of sentences: " + str(len(all_text)))
    print("vocab size: " + str(len(word_cab)))
    max_l = len(max(all_text, key=len))
    print("max sentence length: " + str(max_l))    
    #加载词向量
    save_path = 'null'
    print("word2vec loaded!")
    index = text_to_word2vec(save_path,all_text)
    return train_data,val_data,all_text,word_cab
   
  
word_enbedding_path = r'E:\fakenews\sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
model = gensim.models.KeyedVectors.load_word2vec_format(word_enbedding_path, binary=False)
print("information",models)
word_indice = model.wv.index2word
print("word_indice",word_indice)