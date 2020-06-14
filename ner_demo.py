from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
import numpy as np

def load_data():
    file = open('./train.txt')
    final_data = []
    tmp_words = []
    tmp_tags = []
    for line in file:
        line = line.strip('\n')
        if (line == '-DOCSTART- -X- -X- O'):
            continue
        elif (line == '' and len(tmp_words)>0 and len(tmp_words)==len(tmp_tags)):
            final_data.append((tmp_words, tmp_tags, len(tmp_tags)))
            tmp_words = []
            tmp_tags = []
        else:
            s = line.split(' ')
            if(len(s)>1):
                tmp_words.append(s[0])
                tmp_tags.append(s[-1])
    return final_data


def get_id_dic(raw_data):
    word_freq_dic, tag_freq_dic = {}, {}
    for one_tuple in raw_data:
        words = one_tuple[0]
        tags = one_tuple[1]
        mlen = one_tuple[2]
        sample = (words, tags)
        # 统计数量
        for word, tag in zip(*sample):
            if (word in word_freq_dic.keys()):
                word_freq_dic[word] += 1
            else:
                word_freq_dic[word] = 1
            if (tag in tag_freq_dic.keys()):
                tag_freq_dic[tag] += 1
            else:
                tag_freq_dic[tag] = 1

    word_freq_list = sorted(word_freq_dic.items(), key=lambda x: x[1], reverse=True)
    tag_freq_list = sorted(tag_freq_dic.items(), key=lambda x: x[1], reverse=True)

    word_id_dic, tag_id_dic = {}, {}
    word_id_dic['UNK'] = 0

    max_word_id_num = min(16000,len(word_freq_list)+1)

    for i in range(max_word_id_num):
        word_id_dic[word_freq_list[i][0]] = i + 1

    for i in range(len(tag_freq_list)):
        tag_id_dic[tag_freq_list[i][0]] = i
    return word_id_dic, tag_id_dic,max_word_id_num


def word2num(raw_data, word_id_dic, tag_id_dic, max_words_per_sentence=80):
    all_words_id=[]
    all_tags_id=[]

    for one_tuple in raw_data:
        words = one_tuple[0]
        tags = one_tuple[1]
        mlen = one_tuple[2]
        if (mlen >= max_words_per_sentence):
            # 需要截断成最大80个字
            words = words[0:max_words_per_sentence]
            tags = tags[0:max_words_per_sentence]
        else:
            # 需要补到80个字
            words = words + ['UNK'] * (max_words_per_sentence - mlen)
            tags = tags + ['O'] * (max_words_per_sentence - mlen)
        #转成id
        for i in range(len(words)):
            word=words[i]
            if(word not in word_id_dic.keys()):
                words[i]=0
            else:
                words[i]=word_id_dic[word]
        for i in range(len(tags)):
            tag = tags[i]
            if (tag not in tag_id_dic.keys()):
                tags[i] = 0
            else:
                tags[i] = tag_id_dic[tag]
        all_words_id.append(words)
        all_tags_id.append(tags)
    return np.array(all_words_id),np.array(all_tags_id)


def get_model(num_classes,max_words_per_sentence,max_word_id_num):

    model=models.Sequential()
    model.add(layers.Embedding(input_dim=max_word_id_num+1, output_dim=256, input_length=max_words_per_sentence))
    model.add(layers.Bidirectional(layers.LSTM(128,return_sequences=True),merge_mode="concat"))
    model.add(layers.Bidirectional(layers.LSTM(128,return_sequences=True),merge_mode="concat"))
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dense(num_classes,activation='softmax'))
    opt = optimizers.Adam(0.001)
    model.compile(loss="sparse_categorical_crossentropy",optimizer=opt)
    return model


def cal_correct_rate(pre_y,true_y):
    count_0_wrong = 0
    count_1_wrong = 0
    count_2_wrong = 0
    count_others = 0
    for i in range(len(pre_y)):
        pre_y_one = pre_y[i]
        true_y_one = true_y[i]
        count = 0
        for j in range(len(pre_y_one)):
            if (pre_y_one[j] != true_y_one[j]):
                count += 1
        if (count == 0):
            count_0_wrong += 1
        elif (count == 1):
            count_1_wrong += 1
        elif (count == 2):
            count_2_wrong += 1
        else:
            count_others += 1

    print('完全正确的比例{:.2%}'.format(count_0_wrong / len(pre_y)))
    print('错1个的比例{:.2%}'.format(count_1_wrong / len(pre_y)))
    print('错2个的比例{:.2%}'.format(count_2_wrong / len(pre_y)))
    print('错3个以上的比例{:.2%}'.format(count_others / len(pre_y)))


if __name__ == '__main__':
    raw_data = load_data()
    # 制作word和tag的dic,dic的id是0开始的int,出现频率高的排在前面
    word_id_dic, tag_id_dic,max_word_id_num = get_id_dic(raw_data)

    # 将每一句话转成2个80维的向量(即最长80个字),第一个是出现句子的 word 的id(train_x),第二个是对应的ner的tag(命名实体)的id(train_y)
    max_words_per_sentence=80
    all_words_id,all_tags_id = word2num(raw_data, word_id_dic, tag_id_dic, max_words_per_sentence)

    # 之后把(train_x) (train_y) 用深度学习的方法训练后, 以后input一个句子, 就可以返回 对应的ner
    train_x,test_x,train_y,test_y=train_test_split(all_words_id,all_tags_id,test_size=0.1)

    model=get_model(num_classes=len(tag_id_dic),max_words_per_sentence=max_words_per_sentence,max_word_id_num=max_word_id_num)
    model.fit(x=train_x, y=train_y, epochs=10, batch_size=200, verbose=1, validation_split=0.1)
    # model.save("model.h5")
    # model.load_weights("model.h5")

    #用 test_x测试准确率
    pre_y = model.predict(test_x)
    pre_y = np.argmax(pre_y, axis=-1)


    cal_correct_rate(pre_y,test_y)

    print('打印前10个数据')
    for i in range(10):
        print('预测值:',pre_y[i])
        print('实际值:',test_y[i])
