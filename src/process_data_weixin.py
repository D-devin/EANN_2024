import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import jieba
import sklearn
import os
import re
import random
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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
    
def word_cab_id(text ):
    """
    将词表id化
    """
    pass

def text_to_word2vec(word):
    """
    将词表转化为词向量
    然后保存词向量表为文件
    格式暂不定
    """
    pass
    
def get_data (path):
    """
    数据预处理，对每个数据表的每列进行去除标点符号，jieba分词，形成数据1
    然后合集这些词汇，并且统计词汇频率，形成停用词汇表
    再将数据1丢入停用词再去重，后进行训练集和测试集分类。
    最后返回两个csv的path，和全文本。
    dataframe列[id,标题，正文，图片id，评论，判断标签]
    图片处理是直接为路径，还是存储转化后的图像？
    """
    ### reading data to number 
    val_id = []
    val_true = 500
    clear_index = ['Title','news_text','text_consent']
    data_csv = pd.read_csv(path = path,encoding= 'utf-8')
    for i in clear_index:
        #去除标点以及去除jieba词
        data_csv [i] = data_csv[i].apply(lambda x: ssl_clean(x))
        data_csv[i] = data_csv[i].apply(lambda x: ','.join(jieba.cut_for_search(x)))    
    text_title = data_csv['Title'].tolist()
    text_form = data_csv['news_text'].tolist()
    text_consent = data_csv['text_consent'].tolist()
    cab = text_consent + text_form + text_title

    text_to_word2vec(cab)
    """
    目前暂定这样等会，看看哪些地方比较构式的地方等会再改
    """
    image_url = data_csv['image_path'].tolist()
    label = data_csv['label'].tolist()

    id = random.shuffle(list(range(data_csv.index)))
    val_num = 0
    val_data = pd.DataFrame()
    for i in range(id):
        if data_csv['news_tag'][i] == 1 and val_num<=(val_true*3):
            val_data = val_data.append(data_csv.iloc[i], ignore_index=True)
            val_num +=1
        if data_csv['news_tag'][i] == 2 and val_num<=(val_true*3):
            val_data = val_data.append(data_csv.iloc[i], ignore_index=True)
            val_num +=1
        if val_num == val_true*3:
            break
    train_data = data_csv.drop(val_data.index)
    train_data.save('train.csv')
    val_data.save('val.csv')
    return train_data,val_data
       

def read_data(text_only):
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
    print("save data...")    
    train_data,val_data=get_data()

    print("loading data...")
    vocab, all_text = bulid_vocab(train_data,val_data)

    print("number of sentences: " + str(len(all_text)))
    print("vocab size: " + str(len(vocab)))
    max_l = len(max(all_text, key=len))
    print("max sentence length: " + str(max_l))    

    print("word2vec loaded!")

    print("num words already in word2vec: " + str(len(w2v)))

    return train_data,val_data
   
  
