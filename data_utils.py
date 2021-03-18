# coding = utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import json
import tarfile
import configparser
import pickle

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import math, os
config = configparser.RawConfigParser()
config.read('config')

#Author: Chenglin
# Special vocabulary symbols - we always put them at the start.
# 设定特殊字符，用于句子补长，起止标识
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _UNK, _EOS, _GO]

PAD_ID = 0
UNK_ID = 1
EOS_ID = 2
GO_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(r"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")
_DIGIT_RE2 = re.compile(r"\d")


#将读入的句子切分
def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    # print(words[0])
    # print(type(words[0]))
    # print([w for w in words if w])
    return [w for w in words if w]

#根据读入数据构建词典
#按照限定的词典大小和词出现频率，对超出范围词汇进行裁剪
#计算覆盖率，overlap，防止覆盖率过低
def create_vocabulary(vocabulary_path, data, max_vocabulary_size,
                                            tokenizer=None, normalize_digits=True):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data" % (vocabulary_path))
        print(len(data))
        vocab = {}
        counter = 0
        num = 0
        for line in data:
            counter += 1
            if counter % 100000 == 0:
                print("    processing line %d" % counter)
                print(line)
            # line = tf.compat.as_bytes(line)
            tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
            for w in tokens:
                num += 1
                word = _DIGIT_RE2.sub(r"0", w) if normalize_digits else w
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        pickle.dump(vocab, open('word' + '2idf', 'wb'))
        print('词典准备完了=============长度：', len(vocab))
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        # print(vocab_list)
        print('词典准备完了=============长度：', len(vocab_list))
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

            overlap = .0
            for key in vocab_list[len(_START_VOCAB):]:
                overlap += vocab[key]
            print(num)
            print(overlap)
            print(type(vocab_list[5]))
            print("overlap %f" % (overlap / num))
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + "\n")

#初始化词典
def initialize_vocabulary(vocabulary_path):
    #判断词典存在与否
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip().decode('utf8') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

#将自然语言序列转换正词典id序列
def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, normalize_digits=True):

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE2.sub(r"0", w), UNK_ID) for w in words]

#词向id转换
def data_to_token_ids(data, post_vocabulary_path, response_vocabulary_path,
                                            tokenizer=None, normalize_digits=True):
    print("Tokenizing data")
    post_vocab, _ = initialize_vocabulary(post_vocabulary_path)
    response_vocab, _ = initialize_vocabulary(response_vocabulary_path)
    counter = 0
    for pair in data:
        pair[0][0] = sentence_to_token_ids(pair[0][0], post_vocab, tokenizer, normalize_digits)
        pair[1][0] = sentence_to_token_ids(pair[1][0], response_vocab, tokenizer, normalize_digits)

        # for response in pair[1]:
        #     counter += 1
        #     if counter % 100000 == 0:
        #         print("    tokenizing pair %d" % counter)
        #     response[0] = sentence_to_token_ids(response[0], response_vocab, tokenizer, normalize_digits)

#整体处理raw data，构建词典，处理序列转化id
def prepare_data(data_dir, post_vocabulary_size, response_vocabulary_size, tokenizer=None):

    # Get data to the specified directory.
    train_path = os.path.join(data_dir, config.get('data', 'raw_train_file'))
    dev_path = os.path.join(data_dir, config.get('data', 'raw_dev_file'))

    tokenids_train_path = os.path.join(data_dir, config.get('data', 'train_file'))
    tokenids_dev_path = os.path.join(data_dir, config.get('data', 'dev_file'))
    tokenids_test_path = os.path.join(data_dir, config.get('data', 'test_file'))

    response_vocab_path = os.path.join(data_dir, config.get('data', 'response_vocab_file') % (response_vocabulary_size))
    post_vocab_path = os.path.join(data_dir, config.get('data', 'post_vocab_file') % (post_vocabulary_size))

    if not gfile.Exists(tokenids_train_path) or not gfile.Exists(tokenids_dev_path):

        train = json.load(open(train_path,'r'))
        dev = json.load(open(dev_path,'r'))
        # Create vocabularies of the appropriate sizes.
        print('==========创建词典：')
        create_vocabulary(response_vocab_path, [x[0][0] for x in train], response_vocabulary_size, tokenizer)
        create_vocabulary(post_vocab_path, [x[1][0] for x in train], post_vocabulary_size, tokenizer)

        # Create token ids for the training data.
        data_to_token_ids(train, post_vocab_path, response_vocab_path, tokenizer)

        # Create token ids for the development data.
        data_to_token_ids(dev, post_vocab_path, response_vocab_path, tokenizer)

        # Write data
        with open(tokenids_train_path, 'w') as output:
            output.write(json.dumps(train, ensure_ascii=False))
        with open(tokenids_dev_path, 'w') as output:
            output.write(json.dumps(dev, ensure_ascii=False))

    return (tokenids_train_path, tokenids_dev_path, tokenids_test_path, post_vocab_path, response_vocab_path)    
#加载词向量
def load_word_vector(fname):
    dic = {}
    with open(fname) as f:
        data = f.readlines()
        for line in data:
            s = line.strip()
            word = s[:s.find(' ')]
            vector = s[s.find(' ')+1:]
            dic[word] = vector
    return dic
#从词向量文件中load词向量
def load_vocab(fname):
    vocab = []
    with open(fname) as f:
        data = f.readlines()
        for d in data:
            vocab.append(d[:-1])
    return vocab
#为未登录词创建随机embedding
def random_init(dim):
    return 2 * math.sqrt(3) * (np.random.rand(dim) - 0.5) / math.sqrt(dim)

def refine_wordvec(rvector, vocab, dim=200):
    wordvec = []
    count = 0
    found = 0
    for word in vocab:
        count += 1
        if word in rvector:
            found += 1
            aa = np.array(list(map(float, rvector[word].split())))
            # print('aa.shape:', aa.shape)
            wordvec.append(aa)
        else:
            bb = np.array(random_init(dim))
            wordvec.append(bb)
            # print('bb.shape:', bb.shape)
    # print('Total words: %d, Found words: %d, Overlap: %f' % (count, found, float(found)/count))
    return np.array(wordvec)
#转换成自然语言序列
def cov2seq(sequence, dictionary):
    if not sequence:
        return ''
    sentence = []
    for wordId in sequence:
        if wordId == EOS_ID:  # End of generated sentence
            break
        elif wordId != PAD_ID :
            sentence.append(dictionary[wordId])
    return sentence
#加载词向量，返回post和response词向量矩阵
def get_data(data_dir, post_vocabulary_size, response_vocabulary_size):
    import scipy.io
    path = os.path.join(data_dir, config.get('data', 'wordvec'))
    try:
        mdict = scipy.io.loadmat(path)
        wordvec_post = mdict['post']
        # print('wordvec_post:', wordvec_post)
        wordvec_response = mdict['response']
        # print('wordvec_response:', wordvec_response)
    except:
        print('loading word vector...')
        raw_vector = load_word_vector(config.get('data','raw_wordvec'))
        print('loading vocabulary...')
        vocab_post = load_vocab(os.path.join(data_dir, config.get('data', 'post_vocab_file') % (post_vocabulary_size)))
        vocab_response = load_vocab(os.path.join(data_dir, config.get('data', 'response_vocab_file') % (response_vocabulary_size)))
        print('refine word vector...')
        wordvec_post = refine_wordvec(raw_vector, vocab_post)
        wordvec_response = refine_wordvec(raw_vector, vocab_response)
        mdict = {'post': wordvec_post, 'response': wordvec_response}
        scipy.io.savemat(path, mdict=mdict)
            
    return wordvec_post, wordvec_response
#取外部词典分布矩阵，暂无作用
#TODO：如情感标签有改进，可以考虑使用这个外部记忆，参考ECM论文
def get_ememory(data_dir, response_vocabulary_size):
    dic_path = os.path.join(data_dir, config.get('data', 'ememory_vocab_file') % (response_vocabulary_size))
    vocab_response, _ = initialize_vocabulary(os.path.join(data_dir, config.get('data', 'response_vocab_file') % (response_vocabulary_size)))
    dic = json.load(open(dic_path, 'r'))
    emem = []
    for i in range(6):
        #if i == 0:
        #    emem.append(np.ones(response_vocabulary_size, dtype='float32'))
        #else:
        vec = [0] * response_vocabulary_size
        for j in dic[i]:
        #    print(j, vocab_response[j])
            if j in vocab_response:
                vec[vocab_response[j]] = 1
        emem.append(np.array(vec, dtype='float32'))
    emem = np.array(emem, dtype='float32')
    return emem
    

