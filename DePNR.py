import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input, TimeDistributed, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import pandas as pd
from itertools import groupby
from tensorflow import concat
import tensorflow as tf
import numpy as np
from CRF_TF2 import CRF
from nltk import pos_tag
from transformers import BertTokenizer,BertConfig,XLNetTokenizer,XLNetConfig,DebertaConfig,DebertaTokenizer
from transformers import TFBertModel,TFXLNetModel,TFDebertaModel
from transformers import logging
import string
from tensorflow.keras.initializers import RandomUniform
import itertools
from sklearn.preprocessing import StandardScaler
from math import pow,floor
from tensorflow.keras.callbacks import LearningRateScheduler,ReduceLROnPlateau


def file_read(input_url):
    data_read=pd.read_csv(input_url)
    words=list(data_read['word'])
    lalbels=list(data_read['annotation'])
    pos=list(data_read['pos_tag'])
    return words,lalbels,pos

#标签与id的相互转化
def idx_switch(sentence,label_tag,label_id):
    idx=sentence.copy()
    for i in range(len(sentence)):
        if sentence[i] in label_tag:
           idx[i]=label_id[label_tag.index(sentence[i])]
        else:
            idx[i]=len(label_tag)
    return idx

#分割句子，将list转化为以句子为单元的嵌套list
def sentence_segmention(words_sequences,label_sequences,pos_sequences):
    Index_fullstop = [i for i, x in enumerate(words_sequences) if x == '###']
    # print(Index_fullstop)

    words_sentences_nest=[]

    labels_sentences_nest=[]

    pos_sentences_nest=[]

    # char_words=[]
    char_sentences_nest=[]

    # 单独插入第一句话
    first_sentence_word = words_sequences[:Index_fullstop[0] + 1]
    words_sentences_nest.append(first_sentence_word)
    # print("第一句话word",len(words_sequences[:Index_fullstop[0]+1]),words_sequences[:Index_fullstop[0]+1])

    first_sentence_label = label_sequences[:Index_fullstop[0] + 1]
    labels_sentences_nest.append(first_sentence_label)
    # print("第一句话label", len(label_sequences[:Index_fullstop[0] + 1]),label_sequences[:Index_fullstop[0] + 1])

    first_sentence_pos = pos_sequences[:Index_fullstop[0] + 1]
    pos_sentences_nest.append(first_sentence_pos)
    # print("第一句话pos", len(pos_sequences[:Index_fullstop[0] + 1]),pos_sequences[:Index_fullstop[0] + 1])

    first_sentence_word_copy = first_sentence_word.copy()
    for i in range(len(first_sentence_word)):
        first_sentence_word_char = list(first_sentence_word[i])
        first_sentence_word_copy[i] = first_sentence_word_char

    char_sentences_nest.append(first_sentence_word_copy)
    # print("第一句话char", len(first_sentence_word_copy),first_sentence_word_copy)

    for j in range(len(Index_fullstop)):
        if j % 100 == 0:
            print(j)

        if j != len(Index_fullstop) - 1:
            # 单词（word）
            word_sentence = words_sequences[Index_fullstop[j] + 1:Index_fullstop[j + 1] + 1]
            words_sentences_nest.append(word_sentence)
            # print('word_sentence:',len(word_sentence),word_sentence)

            # 词性（part of speech）
            pos_sentence = pos_sequences[Index_fullstop[j] + 1:Index_fullstop[j + 1] + 1]
            pos_sentences_nest.append(pos_sentence)
            # print('pos_sentence:', len(pos_sentence), pos_sentence)

            # 标签（label）
            label_sentence = label_sequences[Index_fullstop[j] + 1:Index_fullstop[j + 1] + 1]
            labels_sentences_nest.append(label_sentence)
            # print('label_sentence:', len(label_sentence), label_sentence)

            # 字符（char）
            word_sentence_copy = word_sentence.copy()
            for k in range(len(word_sentence)):
                word_char = list(str(word_sentence[k]))
                word_sentence_copy[k] = word_char
            char_sentences_nest.append(word_sentence_copy)
            # print('char_sentence',len(word_sentence_copy),word_sentence_copy)

    return words_sentences_nest, labels_sentences_nest, pos_sentences_nest, char_sentences_nest



def create_inputs(words_sentences_nest,pos_sentences_nest,char_sentences_nest,labels_sentences_nest):

    maxlenth_sentence=256
    maxlenth_word=64

    input_idx=[]
    token_type_idx=[]
    attention_mask_idx=[]
    pos_idx=[]
    label_idx=[]
    char_idx=[]
    sentences_lenth=[]
    #调用BERT的分词类，用于token转换
    tokenizer = DebertaTokenizer.from_pretrained('kamalkraj/deberta-base')

    # 定义词性与其id的对应list
    pos_tag=["ADP","DET","NOUN","ADJ","PUNCT","VERB","AUX","PART","ADV","PRON","PROPN","NUM","CCONJ","X","SYM","SCONJ","INTJ"]
    pos_tag_id=list(range(len(pos_tag)))
    # print("pos_tag_id:",pos_tag_id)

    #定义标签与其id对应的list
    label_tag=['O','B-geo','I-geo']
    label_tag_id=list(range(len(label_tag)))
    # print(label_tag_id)

    #定义字符与其id的对应list
    char_tag=list(string.printable)
    char_tag_id=list(range(len(char_tag)))
    # print("char_tag_id:",char_tag_id)



    for i in range(len(words_sentences_nest)):
        if i%100==0:
            print(i)
        sentence_lenth=len(words_sentences_nest[i])
        sentences_lenth.append(sentence_lenth)
        # print("句子长度：",i,sentence_lenth)
        #单词索引转换，不使用BERT的分词器
        input_id=tokenizer.convert_tokens_to_ids(words_sentences_nest[i])
        input_id_padding=input_id+[0]*(maxlenth_sentence-len(input_id))
        input_idx.append(input_id_padding)
        # print("input_id:", len(input_id_padding), input_id_padding)

        #token类型，单个句子作为输入，全都是0，两个句子输入的话，一个句子为0，一个句子为1，这里是单个句子，全都设置为0
        token_type_id=[0] * len(input_id_padding)
        token_type_idx.append(token_type_id)
        # print("token_type:",len(token_type_id),token_type_id)

        #attention标记，1为计算attention，0为padding部分不计算
        attention_mask_id = [1] * len(input_id)+ ([0] * (maxlenth_sentence-len(input_id)))
        attention_mask_idx.append(attention_mask_id)
        # print("attention_mask:",len(attention_mask_id),attention_mask_id)

        #pos_idx
        pos_id=idx_switch(pos_sentences_nest[i],pos_tag,pos_tag_id)+([0] * (maxlenth_sentence-len(pos_sentences_nest[i])))
        pos_idx.append(pos_id)
        # print("pos_id:",len(pos_id),pos_id)

        # label_idx
        # print("labels_sentence_nest_i:", i, len(labels_sentences_nest[i]), labels_sentences_nest[i])
        # print("label转换：", i, len(idx_switch(labels_sentences_nest[i], label_tag, label_tag_id)),idx_switch(labels_sentences_nest[i], label_tag, label_tag_id))

        label_id = idx_switch(labels_sentences_nest[i], label_tag, label_tag_id) + ([0] * (maxlenth_sentence - len(labels_sentences_nest[i])))
        label_idx.append(label_id)
        # print("label_id:", len(label_id), label_id)


        #char_idx（记得在句子级别重新padding一次）
        char_sentences_id=char_sentences_nest[i].copy()
        for j in range(len(char_sentences_nest[i])):
            if j%100==0:
                print(j)
            char_id=idx_switch(char_sentences_nest[i][j],char_tag,char_tag_id)+([0] * (maxlenth_word-len(char_sentences_nest[i][j])))
            char_sentences_id[j]=char_id

        #padding
        char_id_padding=char_sentences_id+[[0] * maxlenth_word]*(maxlenth_sentence-len(char_sentences_id))

        # print("char_id:",len(char_id_padding),char_id_padding)
        char_idx.append(char_id_padding)



    # print("分割：----------------------------------------------------------------------------")
    # print("分割：----------------------------------------------------------------------------")
    # print("分割：----------------------------------------------------------------------------")
    # print("input_idx:",len(input_idx),input_idx)
    # print("token_type_idx:", len(token_type_idx), token_type_idx)
    # print("attention_mask_idx:", len(attention_mask_idx), attention_mask_idx)
    # print("pos_idx:", len(pos_idx), pos_idx)
    # print("char_idx:", len(char_idx), char_idx)
    # print("label_idx:", len(label_idx), label_idx)
    # print("每个句子的长度为：",sentences_lenth)
    # wp_sentence=pd.DataFrame({"句子长度":sentences_lenth})
    # wp_sentence.to_excel("句子长度.xlsx",index=False)
    return input_idx,token_type_idx,attention_mask_idx,pos_idx,char_idx,label_idx,sentences_lenth

def create_model():


    HIDDEN_SIZE = 64
    MAX_LEN = 256
    CLASS_NUMS = 3
    POS_SIZE=18
    word_maxlen=64
    char_maxlen=101

    # 采用绝对位置embedding时，relative_attention设置为False, position_bias_input设置为True
    configuration = DebertaConfig.from_pretrained('kamalkraj/deberta-base',output_attentions=True, output_hidden_states=True,use_cache=True, return_dict=True)
    encoding = TFDebertaModel.from_pretrained('kamalkraj/deberta-base',config=configuration)
    print(encoding.config)

    # hub调用BERT需要将数据类型设置为tf.int32
    input_ids = Input(shape=(MAX_LEN,), dtype='int32', name="input_ids")
    token_type_ids = Input(shape=(MAX_LEN,), dtype='int32', name="token_type_ids")
    attention_mask = Input(shape=(MAX_LEN,), dtype='int32', name="attention_mask")

    # BERT[2]为所有的隐藏层（Transformer的encoder部分）
    Bert = encoding(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    # Bert = encoding(input_ids=input_ids, attention_mask=attention_mask)
    print(len(Bert))

    # 最后一个隐藏层序列，所有句子的单词组成
    Bert_Last_Hidden = Bert[0]
    print('最后一层:', Bert_Last_Hidden)

    # 池化层是将[CLS]标记对应的表示取出来，并做一定的变换，作为整个序列的表示并返回，以及原封不动地返回所有的标记表示，其中，激活函数默认是tanh。
    Hidden_layer = Bert[1]
    print('Hidden_layer:', Hidden_layer)
    print('Hidden_layer_lenth:', len(Hidden_layer))
    print("...................................................")
    # # 所有的隐藏层
    # Hidden_layer = Bert[2]
    # for i in range(len(Hidden_layer)):
    #     if i == 0:
    #         print("Embedding层:", Hidden_layer[i])
    #     else:
    #         print("第", i, "个隐藏层(Transformer的encoder部分):", Hidden_layer[i])
    print("...................................................")
    print("后四个隐藏层")
    Hidden_layer_12 = Hidden_layer[12]
    Hidden_layer_11 = Hidden_layer[11]
    Hidden_layer_10 = Hidden_layer[10]
    Hidden_layer_9 = Hidden_layer[9]

    print(Hidden_layer_12)
    print(Hidden_layer_11)
    print(Hidden_layer_10)
    print(Hidden_layer_9)

    # 拼接最后四个隐藏层
    Bert_Last_Four_Hidden = concat([Hidden_layer_9, Hidden_layer_10, Hidden_layer_11, Hidden_layer_12], axis=-1)
    print(Bert_Last_Four_Hidden)

    """""
    input_dim:字典长度，即输入数据最大下标+1,注意，如果字典从1开始编码，输入维度需要设置为字典长度加1  
    output_dim:代表全连接嵌入的维度 
    """
    #pos_Embedding
    input_pos = Input(shape=(None,), dtype='int32', name="input_pos")
    pos_embedding = Embedding(input_dim=POS_SIZE, output_dim=POS_SIZE, dtype='float32', name='POS_Embedding')(input_pos)

    #char_embedding
    input_char = Input(shape=(None, word_maxlen), dtype='float32')
    # input_dim词汇表数量，最大索引值+1
    char_embedding = TimeDistributed(Embedding(input_dim=char_maxlen, output_dim=64,embeddings_initializer=
                                     RandomUniform(minval=-0.5, maxval=0.5)))(input_char)

    char_lstm = TimeDistributed(Bidirectional(LSTM(25, return_sequences=False, return_state=False)),name='char_LSTM')(char_embedding)

    #将Bert_Embedding和POS_Embedding、char_Embedding做拼接
    #Concatenated_Embedding = concat([Bert_Last_Four_Hidden, char_lstm,pos_embedding], axis=-1, name='Concatenated_Embedding')
    Concatenated_Embedding = concat([Bert_Last_Four_Hidden, char_lstm, pos_embedding], axis=-1,name='Concatenated_Embedding')


    # Bi-LSTM层
    Bilstm_layer = Bidirectional(LSTM(HIDDEN_SIZE,return_sequences=True,dropout=0.50))(Concatenated_Embedding)

    #self-Attention层
    #Attention_Layer=Attention(name='self-attention')([Bilstm_layer,Bilstm_layer])

    Full_connection_layer = TimeDistributed(Dense(3))(Bilstm_layer)

    # CRF层
    crf = CRF(CLASS_NUMS, name='crf_layer')
    outputs = crf(Full_connection_layer)

    # 创建模型
    model = Model(inputs=[input_ids, token_type_ids, attention_mask, input_char,input_pos], outputs=outputs)


    model.summary()
    model.compile(loss=crf.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=2.5e-5), metrics=crf.accuracy)
    return model
# def step_decay(epoch):
#     init_lrate=0.1
#     drop=0.5
#     epochs_drop=10
#     lrate=init_lrate*pow(drop,floor(1+epoch)/epochs_drop)
#     return lrate
#设置回调函数，当训练精度达到0.9999时，停止
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('loss')<0.1):
            self.model.stop_training=True

if __name__ == '__main__':

    callback=myCallback()
    #解决显存溢出问题
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #
    # if len(physical_devices) > 0:
    #     for k in range(len(physical_devices)):
    #         tf.config.experimental.set_memory_growth(physical_devices[k], True)
    #         tf.config.experimental
    #         print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    # else:
    #     print("Not enough GPU hardware devices available")

    # Data_NUM=5000
    #input_url_train="G:\\Project\\Topic Phrases Extraction\\topic_phrase_data\\final result\\train_data\\paper_"+str(Data_NUM)+"+wiki_3000.csv"
    input_url_train = "D:\\SCI Paper\\Geographic entities Extraction\\Data\\Train Data\\subword\\Ontonote&wiki1500_subword.csv"
    train_data=file_read(input_url_train)
    words_sequences=train_data[0]
    labels_sequences=train_data[1]
    pos_sequences=train_data[2]
    #print(words_sequences)

    #按句子处理成嵌套list
    data_sequence_nest=sentence_segmention(words_sequences,labels_sequences,pos_sequences)
    words_sentences_nest=data_sequence_nest[0]
    labels_sentences_nest=data_sequence_nest[1]
    pos_sentences_nest=data_sequence_nest[2]
    char_sentences_nest=data_sequence_nest[3]

    #将训练数据转成id，并做单词、词性、字符的padding
    train_data_id=create_inputs(words_sentences_nest,pos_sentences_nest,char_sentences_nest,labels_sentences_nest)

    #BERT模型输入
    input_idx=train_data_id[0]
    token_type_idx=train_data_id[1]
    attention_mask_idx=train_data_id[2]

    #新增输入特征
    pos_idx=train_data_id[3]
    char_idx=train_data_id[4]

    #输出并转one-hot
    # print("lable_idx为",train_data_id[5])
    print("train_lenth:",len(train_data_id[5]),train_data_id[5])
    # test1 = train_data_id[5][0]
    # test2 = train_data_id[5][1][0:300]
    # test = [test1, test2]
    label_idx=to_categorical(train_data_id[5],3)

    # print("lable_idx为(one-hot)：",label_idx)
    #创建模型
    model=create_model()
    #设置学习率衰减
    # lrate=ReduceLROnPlateau(monitor='loss',patience=1,mode='auto',factor=0.1)
    early_stopping=tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0,patience=3,mode='auto',baseline=None,restore_best_weights=False,verbose=1)
    #模型训练
    model.fit([np.array(input_idx), np.array(token_type_idx), np.array(attention_mask_idx),np.array(char_idx),np.array(pos_idx)], label_idx, epochs=100, verbose=1, batch_size=16,callbacks=[callback])


    print("模型训练完成。。。。。。。。。。。。。。。。。。。。。。。")

    #将测试数据处理成与训练数据相同的格式
    input_url_test="D:\\SCI Paper\\Geographic entities Extraction\\Data\\Train Data\\subword\\Test_Data_Literature_subword_fixed.xlsx"
    test_data_read=pd.read_excel(input_url_test)
    words_sequences_test = list(test_data_read['word'])#左闭右开，不包括135，0到134[0:135]
    labels_sequences_test = list(test_data_read['annotation'])
    pos_sequences_test = list(test_data_read['pos_tag'])

    #按句子处理成嵌套list
    data_sequence_nest_test=sentence_segmention(words_sequences_test,labels_sequences_test,pos_sequences_test)
    words_sentences_nest_test=data_sequence_nest_test[0]
    labels_sentences_nest_test=data_sequence_nest_test[1]
    pos_sentences_nest_test=data_sequence_nest_test[2]
    char_sentences_nest_test=data_sequence_nest_test[3]

    # 将训练数据转成id，并做单词、词性、字符的padding
    test_data_id = create_inputs(words_sentences_nest_test, pos_sentences_nest_test, char_sentences_nest_test, labels_sentences_nest_test)

    # BERT模型输入
    input_idx_test = test_data_id[0]
    token_type_idx_test = test_data_id[1]
    attention_mask_idx_test = test_data_id[2]

    #新增输入特征
    pos_idx_test = test_data_id[3]
    char_idx_test = test_data_id[4]


    #模型预测
    predict_lables_one_hot=model.predict([np.array(input_idx_test),np.array(token_type_idx_test),np.array(attention_mask_idx_test), np.array(char_idx_test),np.array(pos_idx_test)],verbose=1)


    #print(predict_lables_one_hot)

    #one-hot转list,并剔除padding过的0
    sentence_lenth_test=test_data_id[6]
    predict_result_idx=[]
    predict_result_labels=[]
    for i in range(len(predict_lables_one_hot)):
        predict_label_id=np.argmax(predict_lables_one_hot[i],axis=1)
        predict_id=list(predict_label_id[0:sentence_lenth_test[i]])
        predict_id_copy = predict_id.copy()
        for j in range(len(predict_id)):
            if predict_id[j]==0:
                predict_id_copy[j]="O"
            elif predict_id[j]==1:
                predict_id_copy[j] = "B-geo"
            else:
                predict_id_copy[j] = "I-geo"
        predict_result_labels.append(predict_id_copy)
        predict_result_idx.append(predict_id)

    # print("predict_result_idx:", predict_result_idx)
    # print("predict_result_labels:",predict_result_labels)
    print("预测完成。。。。。。。。。。。。。。。。。。。。。。。。。。")

    #保存预测结果
    #Output_url="G:\\Project\\Topic Phrases Extraction\\topic_phrase_data\\final result\\baseline result\\paper_"+str(Data_NUM)+"+wiki_3000_predict_result(epoch=30,lr=3e-5,absolutely).csv"
    Output_url = "D:\\SCI Paper\\Geographic entities Extraction\\Data\\Results\\Ontonote&wiki1500_subword_test_result(lr=2.5e-5_absolute3).csv"
    result_DataFrame = pd.DataFrame({"predict_results": list(itertools.chain.from_iterable(predict_result_labels))})
    result_DataFrame.to_csv(Output_url, index=False)


































