#coding=utf-8
# Copyright 2020 JiangChenglin. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Main script. See README.md for more information/主文件，各模块分配详见README文件

Use python 3/请使用PYTHON 3版本
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import nltk
import os
import random
import sys
import time
import json
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from six.moves import xrange    # pylint: disable=redefined-builtin
import tensorflow as tf
import pickle
import data_utils
import seq2seq_model
import configparser

#配置参数文件接口，统一配置，在此修改
#训练，测试参数输入
config = configparser.RawConfigParser()
config.read('config')

sess_config = tf.ConfigProto() 
sess_config.gpu_options.allow_growth = True

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.98,
                                                    "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                                                   "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("epoch", 80, "num of whole training turn")
tf.app.flags.DEFINE_integer("batch_size", 256,
                                                        "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("load_model", 0, "which model to load.")
tf.app.flags.DEFINE_integer("beam_size", 20, "Size of beam.")
tf.app.flags.DEFINE_integer("embedding_size", 200, "Size of word embedding.")
tf.app.flags.DEFINE_integer("emotion_size", 200, "Size of emotion embedding.")
tf.app.flags.DEFINE_integer("imemory_size", 256, "Size of imemory.")
tf.app.flags.DEFINE_integer("category", 6, "category of emotions.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("totaldata", 1007286, "the whole dataset size 1 epoch.")
tf.app.flags.DEFINE_integer("post_vocab_size", 40000, "post vocabulary size.")
tf.app.flags.DEFINE_integer("response_vocab_size", 40000, "response vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory.")
tf.app.flags.DEFINE_string("test_dir", "train", "Training directory.")
tf.app.flags.DEFINE_string("pretrain_dir", "pretrain", "Pretraining directory.")
tf.app.flags.DEFINE_integer("pretrain", -1, "pretrain model number")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                                                        "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1000,
                                                        "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("use_emb", False,
                                                        "use embedding model")
tf.app.flags.DEFINE_boolean("use_imemory", False,
                                                        "use imemory model")
tf.app.flags.DEFINE_boolean("use_ememory", False,
                                                        "use ememory model")
tf.app.flags.DEFINE_boolean("decode", False,
                                                        "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("human_evaluation", False,
                                                        "Set to True for interactive decoding.")

tf.app.flags.DEFINE_boolean("metrics", False,
                            "Set to True for make evaluations.")
tf.app.flags.DEFINE_boolean("beam_search", False, "beam search")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_boolean("use_ppx_acc", False,
                            "use metric1")
tf.app.flags.DEFINE_boolean("use_bleu", False,
                            "use BLEU")
tf.app.flags.DEFINE_boolean("use_fg", False,
                            "use fg acc")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(12, 12), (16, 16), (20, 20), (30, 30)]

#读取并分配数据，符合模型训练feed的数据结构
#返回：结构化的训练数据
def read_data(path, max_size=None):
    data_set = [[] for _ in _buckets]
    data = json.load(open(path,'r'))
  #  print(data)
    counter = 0
    size_max = 0
    for pair in data:
        post = pair[0]
        responses = pair[1]
        source_ids = [int(x) for x in post[0]]
        target_ids = [int(x) for x in responses[0]]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(source_ids) < source_size and len(target_ids) < target_size:
                data_set[bucket_id].append([source_ids, target_ids, int(post[1]), int(responses[1])])
                break

    return data_set
#可用于优化数据的函数
def refine_data(data):
    new_data = []
    for d in data:
        b = []
        for e in range(6):
            b.append([x for x in d if x[-1] == e])
        new_data.append(b)
    return new_data
#构建模型框架
#负责初始化用于训练的模型，模型的保存，以及测试时已有模型的复载。
def create_model(session, forward_only, beam_search):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    #加载预训练好的词向量文件
    vec_post, vec_response = data_utils.get_data(FLAGS.data_dir, FLAGS.post_vocab_size, FLAGS.response_vocab_size)
    print('============-===============', vec_post)
    print(len(vec_post[1]))
    model = seq2seq_model.Seq2SeqModel(
            FLAGS.post_vocab_size,
            FLAGS.response_vocab_size,
            _buckets,
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            FLAGS.learning_rate,
            FLAGS.learning_rate_decay_factor,
            wordEmbedding=vec_post,
            embedding_size=FLAGS.embedding_size,
            forward_only=forward_only,
            beam_search=beam_search,
            beam_size=FLAGS.beam_size,
            category=FLAGS.category,
            use_emb=FLAGS.use_emb,
            use_imemory=FLAGS.use_imemory,
            use_ememory=FLAGS.use_ememory,
            emotion_size=FLAGS.emotion_size,
            imemory_size=FLAGS.imemory_size,
            dtype=dtype)
    see_variable = True
    if see_variable == True:
        for i in tf.all_variables():
            print(i.name, i.get_shape())
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    pre_ckpt = tf.train.get_checkpoint_state(FLAGS.pretrain_dir)
    #判断是否已经存在模型文件
    if ckpt: #and tf.gfile.Exists(ckpt.model_checkpoint_path+".index"):
        if FLAGS.load_model == 0:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            path = ckpt.model_checkpoint_path[:ckpt.model_checkpoint_path.find('-')+1]+str(FLAGS.load_model)
            print("Reading model parameters from %s" % path)
            model.saver.restore(session, path)
    else:
      #初始化，从新训练
        if pre_ckpt:
            session.run(tf.initialize_variables(model.initial_var))
            if FLAGS.pretrain > -1:
                path = pre_ckpt.model_checkpoint_path[:pre_ckpt.model_checkpoint_path.find('-')+1]+str(FLAGS.pretrain)
                print("Reading pretrain model parameters from %s" % path)
                model.pretrain_saver.restore(session, path)
            else:
                print("Reading pretrain model parameters from %s" % pre_ckpt.model_checkpoint_path)
                model.pretrain_saver.restore(session, pre_ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())
            # vec_post, vec_response = data_utils.get_data(FLAGS.data_dir, FLAGS.post_vocab_size, FLAGS.response_vocab_size)
            # print('vec_post:', vec_post.shape)
            # print('vec_res:', vec_response)
            # initvec_post = tf.constant(vec_post, dtype=dtype, name='init_wordvector_post')
            #定位decoder词向量初始化，用预训练的词向量替换
            initvec_response = tf.constant(vec_response, dtype=dtype, name='init_wordvector_response')
            # embedding_post = [x for x in tf.trainable_variables() if x.name == 'embedding_attention_seq2seq/rnn/embedding_wrapper/embedding:0'][0]
            embedding_response = [x for x in tf.trainable_variables() if x.name == 'embedding_attention_seq2seq/embedding_attention_decoder/embedding:0'][0]
            print(type(embedding_response))
            print(embedding_response)
             # session.run(tf.assign(embedding_post, initvec_post))
             # session.run(tf.assign(embedding_response, initvec_response))
            # session.run(embedding_post.assign(initvec_post))
            session.run(embedding_response.assign(initvec_response))
#
    return model

#开始训练/Start Training
def train():
    # print(FLAGS.__flags)
    # Prepare data.
    print("Preparing data in %s" % FLAGS.data_dir)
    train_path, dev_path, test_path, _, _ = data_utils.prepare_data(
            FLAGS.data_dir, FLAGS.post_vocab_size, FLAGS.response_vocab_size)

    with tf.Session(config=sess_config) as sess:
        # 构建模型/create model
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False, False)

        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)."
                     % FLAGS.max_train_data_size)
        _, id2word = data_utils.initialize_vocabulary('/mnt/data/jiangchenglin/fc-master/data/vocab40000.response')
        dev_set = read_data(dev_path)
        dev_set = refine_data(dev_set)
        train_set = read_data(train_path, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        print([len(x) for x in dev_set])
        # for x in dev_set:
        #     print(x)
        print([len(x) for x in train_set])
        # for x in train_set:
        #     print(x)
        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                                                     for i in xrange(len(train_bucket_sizes))]
        print(train_buckets_scale)
        
        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        in_epoch_steps = FLAGS.totaldata / FLAGS.batch_size
        previous_losses = []

        word2count = pickle.load(open('/mnt/data/jiangchenglin/fc-master/word2idf', 'rb'))
        word2count["_PAD"] = 10000000
        word2count["_GO"] = 10000000
        word2count["_EOS"] = 10000000
        word2count["_UNK"] =100

        try:  # 触发用户终止训练异常，保存当前模型文件
            for e in range(FLAGS.epoch):
                print("enter the traing, epoch:",(e+1))
                for i in range(int(in_epoch_steps)):
                    # Choose a bucket according to data distribution. We pick a random number
                    # in [0, 1] and use the corresponding interval in train_buckets_scale.
                    random_number_01 = np.random.random_sample()
                    bucket_id = min([i for i in xrange(len(train_buckets_scale))
                                     if train_buckets_scale[i] > random_number_01])

                    # Get a batch and make a step./一个一个batch迭代训练，完成一个epoch
                    print("Get a batch and make a step")

                    start_time = time.time()
                    encoder_inputs, decoder_inputs, target_weights, target_weights1, decoder_emotions = model.get_batch(
                        train_set, bucket_id, id2word, word2count)
                    _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, target_weights1, decoder_emotions, bucket_id, False, False)
                    step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                    loss += step_loss / FLAGS.steps_per_checkpoint
                    current_step += 1

                    # Once in a while, we save checkpoint, print statistics, and run evals.
                    if current_step % FLAGS.steps_per_checkpoint == 0:
                        # Print statistics for the previous epoch.
                        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                        print("global step %d (%.2f epoch) learning rate %.4f step-time %.2f perplexity "
                              "%.2f" % (model.global_step.eval(), model.global_step.eval() / float(in_epoch_steps),
                                        model.learning_rate.eval(), step_time, perplexity))
                        # Decrease learning rate if no improvement was seen over last 3 times.
                        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                            sess.run(model.learning_rate_decay_op)
                        previous_losses.append(loss)
                        # Save checkpoint and zero timer and loss.
                        if current_step % (FLAGS.steps_per_checkpoint * 10) == 0 or current_step % 34000 == 0:
                            checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                        step_time, loss = 0.0, 0
                        # dev set evaluation
                        total_loss = .0
                        total_len = .0
                        for bucket_id in xrange(len(_buckets)):
                            if len(dev_set[bucket_id]) == 0:
                                print("    eval: empty bucket %d" % (bucket_id))
                                continue
                            bucket_loss = .0
                            bucket_len = .0
                            for e in range(6):
                                len_data = len(dev_set[bucket_id][e])
                                for batch in xrange(0, len_data, FLAGS.batch_size):
                                    step = min(FLAGS.batch_size, len_data - batch)
                                    model.batch_size = step
                                    encoder_inputs, decoder_inputs, target_weights, target_weights1, decoder_emotions = model.get_batch_data(
                                        dev_set[bucket_id][e][batch:batch + step], bucket_id, id2word, word2count)
                                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                                 target_weights, decoder_emotions, bucket_id, True,
                                                                 False)
                                    bucket_loss += eval_loss * step
                                bucket_len += len_data
                            total_loss += bucket_loss
                            total_len += bucket_len
                            bucket_loss = float(bucket_loss / bucket_len)
                            bucket_ppx = math.exp(bucket_loss) if bucket_loss < 300 else float(
                                "inf")
                            print("    dev_set eval: bucket %d perplexity %.2f" % (bucket_id, bucket_ppx))
                        total_loss = float(total_loss / total_len)
                        total_ppx = math.exp(total_loss) if total_loss < 300 else float(
                            "inf")
                        print("    dev_set eval: bucket avg perplexity %.2f" % (total_ppx))
                        sys.stdout.flush()
                        model.batch_size = FLAGS.batch_size
        except (KeyboardInterrupt, SystemExit):  # If the user press Ctrl+C while testing progress
            print('Interruption detected, exiting the program...')
#用于交互实测，体验对话效果。
def decode():
    try:
        from wordseg_python import Global
    except:
        Global = None

    def split(sent):
        sent = sent.decode('utf-8', 'ignore').encode('gbk', 'ignore')
        if Global == None:
            return sent.decode("gbk").split(' ')
        tuples = [(word.decode("gbk"), pos) for word, pos in Global.GetTokenPos(sent)]
        return [each[0] for each in tuples]

    with tf.Session(config=sess_config) as sess:
        with tf.device("/cpu:0"):
            # Create model and load parameters.
            model = create_model(sess, True, FLAGS.beam_search)
            model.batch_size = 1  # We decode one sentence at a time.
            beam_search = FLAGS.beam_search
            beam_size = FLAGS.beam_size
            num_output = 5

            # 加载词典.
            post_vocab_path = os.path.join(FLAGS.data_dir,
                                           config.get('data', 'post_vocab_file') % (FLAGS.post_vocab_size))
            response_vocab_path = os.path.join(FLAGS.data_dir,
                                               config.get('data', 'response_vocab_file') % (FLAGS.response_vocab_size))
            post_vocab, _ = data_utils.initialize_vocabulary(post_vocab_path)
            _, rev_response_vocab = data_utils.initialize_vocabulary(response_vocab_path)

            # Decode from standard input.
            sys.stdout.write("用户： ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            while sentence:
                print(sentence)
                sentence = " ".join(sentence)
                # Get token-ids for the input sentence.
                token_ids = data_utils.sentence_to_token_ids(sentence, post_vocab)
                print(token_ids)
                int2emotion = ['null', 'like', 'sad', 'disgust', 'angry', 'happy']
                bucket_id = min([b for b in xrange(len(_buckets))
                                 if _buckets[b][0] > len(token_ids)])
                # Get a 1-element batch to feed the sentence to the model.
                decoder_emotion = 0
                encoder_inputs, decoder_inputs, target_weights,target_weights1, decoder_emotions = model.get_batch_data(
                    [[token_ids, [], 0, decoder_emotion]], bucket_id, id2word, word2count)
                # Get output logits for the sentence.
                results, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                                       target_weights, decoder_emotions, bucket_id, True, beam_search)
                if beam_search:
                    result = results[0]
                    symbol = results[1]
                    parent = results[2]
                    result = results[0]
                    symbol = results[1]
                    parent = results[2]
                    res = []
                    nounk = []
                    for i, (prb, _, prt) in enumerate(result):
                        if len(prb) == 0: continue
                        for j in xrange(len(prb)):
                            p = prt[j]
                            s = -1
                            output = []
                            for step in xrange(i - 1, -1, -1):
                                s = symbol[step][p]
                                p = parent[step][p]
                                output.append(s)
                            output.reverse()
                            if data_utils.UNK_ID in output:
                                res.append([prb[j][0],
                                            " ".join([tf.compat.as_str(rev_response_vocab[int(x)]) for x in output])])
                            else:
                                nounk.append([prb[j][0],
                                              " ".join([tf.compat.as_str(rev_response_vocab[int(x)]) for x in output])])
                    res.sort(key=lambda x: x[0], reverse=True)
                    nounk.sort(key=lambda x: x[0], reverse=True)
                    if len(nounk) < beam_size:
                        res = nounk + res[:(num_output - len(nounk))]
                    else:
                        res = nounk
                    for i in res[:num_output]:
                        print(1)
                        
                #在预测的时候，使用greedy去top1回复进行输出
                else:

                    # This is a greedy decoder - outputs are just argmaxes of output_logits.
                    outputs = [int(np.argmax(np.split(logit, [2, FLAGS.response_vocab_size], axis=1)[1], axis=1) + 2)
                               for logit in output_logits]
                    # If there is an EOS symbol in outputs, cut them at that point.
                    if data_utils.EOS_ID in outputs:
                        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                    # Print out response sentence corresponding to outputs.
                    print('E先生：' + ':' + "".join(
                        [tf.compat.as_str(rev_response_vocab[output]) for output in outputs]))
                print("用户：  ", end="")
                sys.stdout.flush()
                sentence = sys.stdin.readline()
    
#以下为评估指标测试样例
#测试指标perplexity,bleu，以及accuraccy（闲聊中暂不考虑）
def evaluation():
    with tf.Session(config=sess_config) as sess:
        model = create_model(sess, False, FLAGS.beam_search)
        # model.evaluation = FLAGS.metrics
        print("Preparing data in %s" % FLAGS.data_dir)
        # _, dev_path, _, _, _ = data_utils.prepare_data(
        #     FLAGS.data_dir, FLAGS.post_vocab_size, FLAGS.response_vocab_size)
        #
        # dev_set = read_data(dev_path)
        # data_set = [[] for _ in _buckets]
        # print('===长度')
        # print(len(dev_set[0]))
        # print(len(dev_set[1]))
        # print(len(dev_set[2]))
        # print(len(dev_set[3]))
        # k = []
        # for i in range(1000):
        #     j = random.randint(1, 11965)
        #     k.append(j)
        #     if i == 0:
        #
        #         for bucket_id in range(len(_buckets)):
        #             data_set[bucket_id].append(dev_set[bucket_id][j])
        #     else:
        #         if j == k[-1]:
        #
        #             while j == k[-1]:
        #                 j = random.randint(1, 11965)
        #             for bucket_id in range(len(_buckets)):
        #                 data_set[bucket_id].append(dev_set[bucket_id][j])
        #         else:
        #             for bucket_id in range(len(_buckets)):
        #                 data_set[bucket_id].append(dev_set[bucket_id][j])
        #
        # dev_set = refine_data(data_set)
        # with open('/home/minelab/jiangchenglin/eqa/mutualAutoeqa/data/test_data', 'w') as output:
        #     output.write(json.dumps(dev_set, ensure_ascii=False))
        test_path = os.path.join(FLAGS.data_dir, config.get('data', 'test_data'))
        dev_set = json.load(open(test_path, 'r'))

        #选择进入模式：
        PPT = FLAGS.use_ppx_acc
        # model.PPT = PPT
        BLEU = FLAGS.use_bleu

        fg_acc = FLAGS.use_fg
        # print('======处理数据')
        # print(dev_set[0][0])
        # print(len(dev_set[0][0]))
        # print(len(dev_set[0][1]))
        # print(len(dev_set[0][5]))

        if PPT:
            total_loss = .0
            total_len = .0

            for bucket_id in xrange(len(_buckets)):
                if len(dev_set[bucket_id]) == 0:
                    print("    eval: empty bucket %d" % (bucket_id))
                    continue
                bucket_loss = .0
                bucket_len = .0

                for e in range(6):
                    len_data = len(dev_set[bucket_id][e])
                    for batch in xrange(0, len_data, FLAGS.batch_size):
                        step = min(FLAGS.batch_size, len_data - batch)
                        model.batch_size = step
                        encoder_inputs, decoder_inputs, target_weights, target_weights1, decoder_emotions = model.get_batch_data(
                            dev_set[bucket_id][e][batch:batch + step], bucket_id, id2word, word2count)
                        _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                     target_weights, decoder_emotions, bucket_id, True,
                                                     False)
                        bucket_loss += eval_loss * step

                    bucket_len += len_data
                total_loss += bucket_loss

                total_len += bucket_len

                bucket_loss = float(bucket_loss / bucket_len)
                bucket_ppx = math.exp(bucket_loss) if bucket_loss < 300 else float(
                    "inf")
                print(
                    "    test_set eval: bucket %d perplexity %.2f" % (bucket_id, bucket_ppx,))
            total_loss = float(total_loss / total_len)

            total_ppx = math.exp(total_loss) if total_loss < 300 else float(
                "inf")
            print("    test_set eval: bucket avg perplexity %.2f" % (total_ppx))
            # sys.stdout.flush()
        if BLEU:
            total_bleu = .0
            total_len = .0
            print('===计算bleu模式')
            # model = create_model(sess, len(word2id), len(word2id), True, beam_search=False, beam_size=1)
            # FLAGS.batch_size = 1
            print('Start testing --bleu (press Ctrl+C to save and exit)...')
            for bucket_id in xrange(len(_buckets)):
                if len(dev_set[bucket_id]) == 0:
                    print("    eval: empty bucket %d" % (bucket_id))
                    continue
                bucket_bleu = .0
                bucket_len = .0

                for e in range(6):
                    len_data = len(dev_set[bucket_id][e])
                    for batch in xrange(0, len_data, FLAGS.batch_size):
                        step = min(FLAGS.batch_size, len_data - batch)
                        model.batch_size = step
                        encoder_inputs, decoder_inputs, target_weights, decoder_emotions, refer = model.get_batch_data1(
                            dev_set[bucket_id][e][batch:batch + step], bucket_id)
                        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                                     target_weights, decoder_emotions, bucket_id, True,
                                                     False)


                        # print(step)
                        k = len(output_logits)
                        # print(output_logits)
                        for i in range(step):
                            logits = []

                            for j in range(k):

                                logits.append(output_logits[j][i])
                            #     print('---0-0-0-0-000---')
                            # print(len(logits))
                            # print(len(logits[1]))
                            # print(type(logits[1]))
                            # print(logits[1])
                            outputs = [
                                int(np.argmax(np.split([logit], [2, FLAGS.response_vocab_size], axis=1)[1], axis=1) + 2)
                                for logit in logits]
                            # If there is an EOS symbol in outputs, cut them at that point.
                            if data_utils.EOS_ID in outputs:
                                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                            #############
                            response_vocab_path = os.path.join(FLAGS.data_dir,
                                                               config.get('data', 'response_vocab_file') % (
                                                                   FLAGS.response_vocab_size))
                            #############
                            _, rev_response_vocab = data_utils.initialize_vocabulary(response_vocab_path)
                            candidate = data_utils.cov2seq(outputs, rev_response_vocab)
                            ref = data_utils.cov2seq(refer[i], rev_response_vocab)
                            refer1 = [ref]
                            #############

                            # print(candidate)
                            # print(refer1)
                            # print('=====')
                            bleu_2 = sentence_bleu(refer1, candidate, weights=(0.5, 0.5, 0, 0))
                            # print(bleu_2)
                            #############

                            bucket_bleu += bleu_2



                    bucket_len += len_data
                total_bleu += bucket_bleu

                total_len += bucket_len

                bucket_bleu = float(bucket_bleu / bucket_len)

                print(
                    "    test_set eval: bucket %d bleu %.2f" % (bucket_id, bucket_bleu))
            total_bleu= float(total_bleu / total_len)


            print("    test_set eval: bucket avg bleu %.2f" % (total_bleu))




#主函数
def main(_):
    if FLAGS.decode:
        decode()
    if FLAGS.metrics:
        evaluation()
    # if FLAGS.human_evaluation:
    #     generation()
    else:
        train()

if __name__ == "__main__":
    tf.app.run()
