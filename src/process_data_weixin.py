import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import jieba
import sklearn
import os
import re
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
    数据预处理，去除标点符号，jieba分词，合成词表以及转化词向量
    """
    ### reading data to number 
    index = ['Title','news_text','text_consent']
    data_csv = pd.read_csv(path = path,encoding= 'utf-8')
    for i in index:
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

def read_data():
      pass 
"""
    训练加载data用
"""
  
