# encoding: utf-8
# time    : 2021-06-02 15:09
# author  : 丛星星
"""
用于对新闻内容在输入模型之前数据预处理。
"""
import jieba
import re

# 创建停用字符list
def get_stop_words(file_path):
    return [line.strip() for line in open(file_path, 'r', encoding='utf-8').readlines()]


# 去除停用字符
def move_stop_words(sentence):
    r = ['https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
         '[a-zA-Z]{1,5}(\.[a-zA-Z0-9][-a-zA-Z0-9]{0,62})+\.?',
         '\d{4}-\d{1,2}-\d{1,4}(:\d{1,2}:\d{1,2})',
         '\d{1,2}月\d{1,2}日',
         '\d{1,4}年\d{1,2}月(\d{1,2}日)?',
         '[0-9.]+', ]
    sentence = re.sub('|'.join(r), '', sentence)
    new_sentence = ''
    counts = {}  #
    stop_words = get_stop_words('static/model/stopwords.txt')
    words = jieba.lcut(sentence)  # 分词
    # 去除停用词
    index = 0
    for word in words:
        if word not in stop_words:
            if len(word) == 1:
                continue
            else:
                counts[index] = word
                index += 1
    # 格式转换dict >> list
    items = list(counts.items())
    # 输出统计
    for i in range(len(items)):
        index, word = items[i]
        new_sentence += word
    return new_sentence


