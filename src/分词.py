import pandas as pd
import jieba
import re
# 读取CSV文件
df = pd.read_csv('train/news_train.csv')
#加载停用表
with open('train/cn_stopwords.txt','r',encoding='utf-8') as f:
 stopwords=f.read().splitlines()
print(len(stopwords))
#标点清理
# 文本清理函数
def ssl_clean(string):
        # 定义一个包含所有要清除的特殊字符的字符类
        special_chars = u"[，。：；.`|“”——_/&;@、《》～（）#！%【】:,+(｜)丨●▶…!�✅㊙҈↑↓🤣\s*nbsp-]"
        # 使用re.sub()函数和字符类来清除所有特殊字符
        string = re.sub(special_chars, "", string)
        # 去除字符串两端的空白字符（注意：这里不处理中间的空白字符）
        return string.strip()

    #文本预处理函数
def preprocess_text(text):
 text = ssl_clean(text)#清理标点
 # 使用正则表达式清除特殊字符，如项目符号“●”
 text = re.sub(r'[●]', '', text)
 text = re.sub(r'[▶]', '', text)
 text = re.sub(r'…+', '', text)
 text = re.sub(r'!+', '', text)
 text = re.sub(r'[�]', '', text)
 text = re.sub(r'[✅]', '', text)
 text = re.sub(r'[㊙]', '', text)
 text = re.sub(r'[҈]', '', text)
 text = re.sub(r'[↑]', '', text)
 text = re.sub(r'[↓]', '', text)
 text = re.sub(r'[🤣]', '', text)
 text = re.sub(r'[﻿]', '', text)
 text=re.sub(r'\s*[，。！？；："‘’“”（）《》【】\n\t]+\s*', '', text)
 text = re.sub(r'~+', '', text)
 text = re.sub(r'\d+', '', text) #清理数字
 words=jieba.cut(text)
 words=[word for word in words if word.strip() and word not in stopwords]
 return ' '.join(words)
#对文本进行预处理
df['News Url']=df['News Url'].apply(preprocess_text)
df['Report Content']=df['Report Content'].apply(preprocess_text)
df['Title']=df['Title'].apply(preprocess_text)
df.to_csv('news_train.csv', index=False)

print('处理完成')
#词表创建
#合并
texts = df['Title'] + df['News Url'] + df['Report Content']
# 创建一个空的列表来存储所有单词
all_words = []
# 遍历每条评论，并将单词添加到列表中
for text in texts:
    words = text.split()  # 因为我们已经用空格连接了单词
    all_words.extend(words)

# 创建一个词典，将每个单词映射到唯一的索引
# 我们可以使用Python的set来去除重复的单词，然后转换为列表并排序
unique_words = sorted(list(set(all_words)))
word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
idx_to_word = {idx: word for idx, word in enumerate(unique_words)}

# 现在，我们可以将文本转换为索引序列
def text_to_idx_sequence(text):
    words = text.split()
    return [word_to_idx[word] for word in words if word in word_to_idx]

from gensim.models import Word2Vec
sentences = df['Title'] + df['News Url'] + df['Report Content'].apply(
    lambda x: x if pd.notnull(x) else '')
sentences = sentences.apply(lambda x: x.split())  # 将文本拆分为单词列表
# 确保 sentences 是一个列表，其中每个元素都是一个单词列表（句子）
sentences_list = sentences.tolist()
# 打开文件以写入
with open('sentences.txt', 'w', encoding='utf-8') as f:
    for sentence in sentences_list:
        # 将单词列表转换为字符串，并用空格分隔
        sentence_str = ' '.join(sentence)
        # 写入句子到文件，每个句子占一行
        f.write(f'{sentence_str}\n')
# 训练word2Vec模型
# 注意：这里只是示例参数，您可能需要调整它们以适应您的数据和需求
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 保存模型（可选）
model.save("word2vec.model")

# 加载模型（如果需要的话）
# model = Word2Vec.load("word2vec.model")

# 获取某个词的词向量（例如“范冰冰”）
# 注意：由于分词和预处理的原因，你可能需要确保输入的词与模型中的词完全匹配
word_vector = model.wv['美女'] if '美女' in model.wv else None
print(word_vector)