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
from gensim.models import Word2Vec


#加载停用词表
def stopwordslist(filepath = '../Data/weibo/stop_words.txt'):
    stopwords = {}
    for line in open(filepath, 'r',encoding='utf-8').readlines():
        line = line.strip()
        stopwords[line] = 1
    #stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

#删除符号
def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", string)
    return string.strip().lower()


# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")
# readimage
def read_image():
    """
    读取图片，进行格式化处理，将图片与id对应，且返回含有图片向量的列表
    """
    image_list = {}
    file_list = ['../Data/weibo/nonrumor_images/', '../Data/weibo/rumor_images/']
    for path in file_list:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        for i, filename in enumerate(os.listdir(path)):  # assuming gif

            # print(filename)
            try:
                im = Image.open(path + filename).convert('RGB')
                im = data_transforms(im)
                #im = 1
                image_list[filename.split('/')[-1].split(".")[0].lower()] = im
            except:
                print(filename)
    print("image length " + str(len(image_list)))
    #print("image names are " + str(image_list.keys()))
    return image_list

def write_txt(data):
    f = open("../Data/weibo/top_n_data.txt", 'wb')
    for line in data:
        for l in line:
            f.write(l+"\n")
        f.write("\n")
        f.write("\n")
    f.close()
text_dict = {}
def write_data(flag, image, text_only):
    #读取文件返回dataframe
    def read_post(flag):
        stop_words = stopwordslist()
        pre_path = "../Data/weibo/tweets/"
        file_list = [pre_path + "test_nonrumor.txt", pre_path + "test_rumor.txt", \
                         pre_path + "train_nonrumor.txt", pre_path + "train_rumor.txt"]
        if flag == "train":
            id = pickle.load(open("../Data/weibo/train_id.pickle", 'rb'))
        elif flag == "validate":
            id = pickle.load(open("../Data/weibo/validate_id.pickle", 'rb'))
        elif flag == "test":
            id = pickle.load(open("../Data/weibo/test_id.pickle", 'rb'))

        post_content = []
        labels = []
        image_ids = []
        twitter_ids = []
        data = []
        #datframe的列
        column = ['post_id', 'image_id', 'original_post', 'post_text', 'label', 'event_label']
        key = -1
        map_id = {}
        top_data = []
        #读取文件筛词，利用了jieba还有去符号，可以不看
        for k, f in enumerate(file_list):

            f = open(f, 'r',encoding='utf-8')
            if (k + 1) % 2 == 1:
                label = 0  ### real is 0
            else:
                label = 1  ####fake is 1

            twitter_id = 0
            line_data = []
            top_line_data = []

            for i, l in enumerate(f.readlines()):


                if (i + 1) % 3 == 1:
                    line_data = []
                    twitter_id = l.split('|')[0]
                    line_data.append(twitter_id)
                if (i + 1) % 3 == 2:

                    line_data.append(l.lower())

                if (i + 1) % 3 == 0:
                    l = clean_str_sst(l.strip())

                    seg_list = jieba.cut_for_search(l)
                    new_seg_list = []
                    for word in seg_list:
                        if word not in stop_words:
                            new_seg_list.append(word)

                    clean_l = " ".join(new_seg_list)
                    if len(clean_l) > 10 and line_data[0] in id:
                        post_content.append(l)
                        line_data.append(l)
                        line_data.append(clean_l)
                        line_data.append(label)
                        event = int(id[line_data[0]])
                        if event not in map_id:
                            map_id[event] = len(map_id)
                            event = map_id[event]
                        else:
                            event = map_id[event]
                        line_data.append(event)
                        data.append(line_data)

            f.close()
        
        data_df = pd.DataFrame(np.array(data), columns=column)
        write_txt(top_data)

        return post_content, data_df

    post_content, post = read_post(flag)
    print("Original post length is " + str(len(post_content)))
    print("Original data frame is " + str(post.shape))

    #两个废弃函数
    def find_most(db):
        #寻找最多
        maxcount = max(len(v) for v in db.values())
        return [k for k, v in db.items() if len(v) == maxcount]

    def select(train, selec_indices):
        #进行选择
        temp = []
        for i in range(len(train)):
            ele = list(train[i])
            temp.append([ele[i] for i in selec_indices])
            #   temp.append(np.array(train[i])[selec_indices])
        return temp


    def paired(text_only = False):
        ordered_image = []
        ordered_text = []
        ordered_post = []
        ordered_event= []
        label = []
        post_id = []
        image_id_list = []
        #image = []
        #将图片和文本的id对应然后加入到data中
        image_id = ""
        for i, id in enumerate(post['post_id']):
            for image_id in post.iloc[i]['image_id'].split('|'):
                image_id = image_id.split("/")[-1].split(".")[0]
                if image_id in image:
                    break
            #处理对应的图片与帖子对应
            if text_only or image_id in image:
                if not text_only:
                    image_name = image_id
                    image_id_list.append(image_name)
                    ordered_image.append(image[image_name])
                ordered_text.append(post.iloc[i]['original_post'])
                ordered_post.append(post.iloc[i]['post_text'])
                ordered_event.append(post.iloc[i]['event_label'])
                post_id.append(id)


                label.append(post.iloc[i]['label'])

        label = np.array(label, dtype=np.int32)
        ordered_event = np.array(ordered_event, dtype=np.int32)
        #输出处理好的数量
        print("Label number is " + str(len(label)))
        print("Rummor number is " + str(sum(label)))
        print("Non rummor is " + str(len(label) - sum(label)))



        #
        if flag == "test":
            y = np.zeros(len(ordered_post))
        else:
            y = []


        data = {"post_text": np.array(ordered_post),
                "original_post": np.array(ordered_text),
                "image": ordered_image, "social_feature": [],
                "label": np.array(label), \
                "event_label": ordered_event, "post_id":np.array(post_id),
                "image_id":image_id_list}
        #print(data['image'][0])


        print("data size is " + str(len(data["post_text"])))
        
        return data

    paired_data = paired(text_only)

    print("paired post length is "+str(len(paired_data["post_text"])))
    print("paried data has " + str(len(paired_data)) + " dimension")
    return paired_data


def load_data(train, validate, test):
    vocab = defaultdict(float)#创立字典
    #合并
    all_text = list(train['post_text']) + list(validate['post_text'])+list(test['post_text'])
    for sentence in all_text:
        for word in sentence:
            vocab[word] += 1
    return vocab, all_text


def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    将数据分成10个交叉验证的集合
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    with open(pos_file, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str_sst(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 1,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)
    with open(neg_file, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str_sst(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 0,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)
    return revs, vocab


def get_W(word_vecs, k=32):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    用于从给定的词向量字典word_vecs中提取词向量矩阵W和词索引映射word_idx_map
    """
    # vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(len(word_vecs) + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    预载google的词向量
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocabte_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocabte_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=32):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    用于为在文档中出现频率不低于min_df的词汇创建单独的词向量
     0.25是选择的，所以未知向量具有（大约）与预训练向量相同的方差
     """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)



def get_data(text_only):
    #text_only = False
    """
    与外部的接口，最后返回dataframe。
    """
    if text_only:
        print("Text only")
        image_list = []
    else:
        print("Text and image")
        image_list = read_image()
    #读取数据获得dataframe
    train_data = write_data("train", image_list, text_only)
    valiate_data = write_data("validate", image_list, text_only)
    test_data = write_data("test", image_list, text_only)
    print("loading data...")

    # w2v_file = '../Data/GoogleNews-vectors-negative300.bin'
    #合并文本+获取频率热词
    vocab, all_text = load_data(train_data, valiate_data, test_data)
    # print(str(len(all_text)))
    print("number of sentences: " + str(len(all_text)))
    print("vocab size: " + str(len(vocab)))
    max_l = len(max(all_text, key=len))
    print("max sentence length: " + str(max_l))

    #加载预载的词向量
    word_embedding_path = "../Data/weibo/w2v.pickle"
    w2v = pickle.load(open(word_embedding_path, 'rb'),encoding='latin1')
    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))
    # w2v = add_unknown_words(w2v, vocab)
    # file_path = "../Data/weibo/event_clustering.pickle"
    # if not os.path.exists(file_path):
    #     train = []
    #     for l in train_data["post_text"]:
    #         line_data = []
    #         for word in l:
    #             line_data.append(w2v[word])
    #         line_data = np.matrix(line_data)
    #         line_data = np.array(np.mean(line_data, 0))[0]
    #         train.append(line_data)
    #     train = np.array(train)
    #     cluster = AgglomerativeClustering(n_clusters=15, affinity='cosine', linkage='complete')
    #     cluster.fit(train)
    #     y = np.array(cluster.labels_)
    #     pickle.dump(y, open(file_path, 'wb+'))
    # else:
    # y = pickle.load(open(file_path, 'rb'))
    # print("Event length is " + str(len(y)))
    # center_count = {}
    # for k, i in enumerate(y):
    #     if i not in center_count:
    #         center_count[i] = 1
    #     else:
    #         center_count[i] += 1
    # print(center_count)
    # train_data['event_label'] = y
    #加入本次的词到词向量中
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    W2 = rand_vecs = {}
    w_file = open("../Data/weibo/word_embedding.pickle", "wb")
    #写入文件保存
    pickle.dump([W, W2, word_idx_map, vocab, max_l], w_file)
    w_file.close()
    return train_data, valiate_data, test_data






# if __name__ == "__main__":
#     image_list = read_image()
#
#     train_data = write_data("train", image_list)
#     valiate_data = write_data("validate", image_list)
#     test_data = write_data("test", image_list)
#
#     # print("loading data...")
#     # # w2v_file = '../Data/GoogleNews-vectors-negative300.bin'
#     vocab, all_text = load_data(train_data, test_data)
#     #
#     # # print(str(len(all_text)))
#     #
#     # print("number of sentences: " + str(len(all_text)))
#     # print("vocab size: " + str(len(vocab)))
#     # max_l = len(max(all_text, key=len))
#     # print("max sentence length: " + str(max_l))
#     #
#     # #
#     # #
#     # word_embedding_path = "../Data/weibo/word_embedding.pickle"
#     # if not os.path.exists(word_embedding_path):
#     #     min_count = 1
#     #     size = 32
#     #     window = 4
#     #
#     #     w2v = Word2Vec(all_text, min_count=min_count, size=size, window=window)
#     #
#     #     temp = {}
#     #     for word in w2v.wv.vocab:
#     #         temp[word] = w2v[word]
#     #     w2v = temp
#     #     pickle.dump(w2v, open(word_embedding_path, 'wb+'))
#     # else:
#     #     w2v = pickle.load(open(word_embedding_path, 'rb'))
#     # # print(temp)
#     # # #
#     # print("word2vec loaded!")
#     # print("num words already in word2vec: " + str(len(w2v)))
#     # # w2v = add_unknown_words(w2v, vocab)
#     # Whole_data = {}
#     # file_path = "../Data/weibo/event_clustering.pickle"
#     # # if not os.path.exists(file_path):
#     # #     data = []
#     # #     for l in train_data["post_text"]:
#     # #         line_data = []
#     # #         for word in l:
#     # #             line_data.append(w2v[word])
#     # #         line_data = np.matrix(line_data)
#     # #         line_data = np.array(np.mean(line_data, 0))[0]
#     # #         data.append(line_data)
#     # #
#     # #     data = np.array(data)
#     # #
#     # #     cluster = AgglomerativeClustering(n_clusters=15, affinity='cosine', linkage='complete')
#     # #     cluster.fit(data)
#     # #     y = np.array(cluster.labels_)
#     # #     pickle.dump(y, open(file_path, 'wb+'))
#     # # else:
#     # # y = pickle.load(open(file_path, 'rb'))
#     # # print("Event length is " + str(len(y)))
#     # # center_count = {}
#     # # for k, i in enumerate(y):
#     # #     if i not in center_count:
#     # #         center_count[i] = 1
#     # #     else:
#     # #         center_count[i] += 1
#     # # print(center_count)
#     # # train_data['event_label'] = y
#     #
#     # #
#     # print("word2vec loaded!")
#     # print("num words already in word2vec: " + str(len(w2v)))
#     # add_unknown_words(w2v, vocab)
#     # W, word_idx_map = get_W(w2v)
#     # # # rand_vecs = {}
#     # # # add_unknown_words(rand_vecs, vocab)
#     # W2 = rand_vecs = {}
#     # pickle.dump([W, W2, word_idx_map, vocab, max_l], open("../Data/weibo/word_embedding.pickle", "wb"))
#     # print("dataset created!")



