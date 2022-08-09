import pandas as pd
from transformers import *
import json
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from preprocess_functions import make_input

import re

def make_ner_model(taskname,
                   model_upper_name,
                   ptmodel_name,
                   max_len,
                   batch_size):

    # max_len = 169
    # batch_size = 4
    # model_upper_name = 'bert' or 'electra' ..
    # ptmodel_name = 'bert-base-multilingual-cased' or 'monologg/koelectra-base-v3-discriminator'..

    # 데이터 읽기
    if taskname == 'maincat':
        train_data, test_data = make_input.read_and_makedf(
            train_path = './../../Traindata/NER_dataset/NERdataset/NER_maincat_train_5455.txt',
            test_path = './../../Traindata/NER_dataset/NERdataset/NER_maincat_test_606.txt')
    elif taskname == 'subcat':
        train_data, test_data = make_input.read_and_makedf(
            train_path='./../../Traindata/NER_dataset/NERdataset/NER_subcat_train_5455.txt',
            test_path='./../../Traindata/NER_dataset/NERdataset/NER_subcat_test_606.txt')


    # 원본 데이터를 데이터프레임 형태로 만들기
    splited_without_morpheme = make_input.split_tagged_and_others(train_data)
    train_df = make_input.make_BIO_df(splited_without_morpheme)
    splited_without_morpheme = make_input.split_tagged_and_others(test_data)
    test_df = make_input.make_BIO_df(splited_without_morpheme)

    #label 목록 저장
    ner_begin_label, ner_labels = make_input.make_ner_labels(train_df, 'label')


    with open('%s_%s_ner_labels.json'%(taskname,re.sub('\/','',ptmodel_name)), 'w') as outfile:
      json.dump(ner_labels, outfile)


    #['CLS']와 ['SEP']토큰을 추가한 문장 만들고, label은 정수 인코딩으로 변환
    train_sentences, train_targets = make_input.make_cls_sep_sentences(ner_labels, train_df)
    test_sentences, test_targets = make_input.make_cls_sep_sentences(ner_labels, test_df)


    #사전학습된 버트 토크나이저 불러옴
    if model_upper_name=='bert':
        tokenizer = BertTokenizer.from_pretrained(ptmodel_name)
    #코일렉트라 토크나이저
    if model_upper_name=='electra':
        tokenizer = ElectraTokenizer.from_pretrained(ptmodel_name)

    # sentence에 대해서는 버트토크나이저 적용, label도 늘어난 토큰에 맞추어 변환해줌
    train_tokenized_texts_and_labels = [make_input.tokenize_and_preserve_labels(sent, labs, tokenizer)
                                  for sent, labs in zip(train_sentences, train_targets)]
    test_tokenized_texts_and_labels = [make_input.tokenize_and_preserve_labels(sent, labs, tokenizer)
                                  for sent, labs in zip(test_sentences, test_targets)]

    # token과 label을 분리
    train_tokenized_texts = [token_label_pair[0] for token_label_pair in train_tokenized_texts_and_labels]
    train_labels = [token_label_pair[1] for token_label_pair in train_tokenized_texts_and_labels]
    test_tokenized_texts = [token_label_pair[0] for token_label_pair in test_tokenized_texts_and_labels]
    test_labels = [token_label_pair[1] for token_label_pair in test_tokenized_texts_and_labels]

    #Sequence의 최대 길이와 배치사이즈
    max_len = max_len
    bs = batch_size

    #padding작업을 통해 input_ids와 tags를 만들어 줌r
    train_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in train_tokenized_texts],
                              maxlen=max_len, dtype = "int", value=tokenizer.convert_tokens_to_ids("[PAD]"),
                              truncating="post", padding="post")
    train_tags = pad_sequences([lab for lab in train_labels], maxlen=max_len, value=ner_labels.index('[PAD]'), padding='post',
                         dtype='int', truncating='post')
    test_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in test_tokenized_texts],
                              maxlen=max_len, dtype = "int", value=tokenizer.convert_tokens_to_ids("[PAD]"),
                              truncating="post", padding="post")
    test_tags = pad_sequences([lab for lab in test_labels], maxlen=max_len, value=ner_labels.index('[PAD]'), padding='post',
                         dtype='int', truncating='post')

    #attention mask도 만들어 줌
    train_attention_masks = np.array([[int(i != tokenizer.convert_tokens_to_ids("[PAD]")) for i in ii] for ii in train_input_ids])
    test_attention_masks = np.array(
        [[int(i != tokenizer.convert_tokens_to_ids("[PAD]")) for i in ii] for ii in test_input_ids])

# #train, test 분리
# tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
#                                                             random_state=2018, test_size=0.1)
# tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
#                                              random_state=2018, test_size=0.1)


    "**********************************************모델 만들기***********************************************************"
    SEQ_LEN = max_len
    if model_upper_name == 'bert':
        model = TFBertModel.from_pretrained(ptmodel_name, from_pt=True, num_labels=len(ner_labels),
                                            output_attentions=False,
                                            output_hidden_states=False)
    if model_upper_name == 'electra':
        model = TFElectraModel.from_pretrained(ptmodel_name, from_pt=True)

    token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')  # 토큰 인풋
    mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')  # 마스크 인풋

    bert_outputs = model([token_inputs, mask_inputs])
    bert_outputs = bert_outputs[0]  # shape : (Batch_size, max_len, 30(개체의 총 개수))
    nr = tf.keras.layers.Dense(len(ner_labels), activation='softmax')(bert_outputs)  # shape : (Batch_size, max_len, 30)

    nr_model = tf.keras.Model([token_inputs, mask_inputs], nr)

    nr_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00002),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                     metrics=['sparse_categorical_accuracy'])
    nr_model.summary()

# 모델 저장을 위해 callback 생성
# cp_callback = ModelCheckpoint('BERTbase_best_weights.h5',
#                               verbose=1,
#                               save_best_only=True,
#                               save_weights_only=True)

    cp_callback = ModelCheckpoint('%s_%s_best_weights.h5'%(taskname,re.sub('\/','',ptmodel_name)),
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=True)

    # 데이터 학습 진행
    print("GPU를 사용한 학습")
    with tf.device("/device:GPU:0"):
        nr_model.fit([train_input_ids, train_attention_masks], train_tags, validation_split = 0.2, epochs=5,
                     shuffle=False, batch_size=bs, callbacks=[cp_callback])



    #학습검증
    print('\n# Evaluate on test data')
    results = nr_model.evaluate([test_input_ids, test_attention_masks], test_tags, batch_size=128)
    print('test loss, test acc : ', results)

    # bi달린 태그에 대한 리포트
    y_predicted = nr_model.predict([test_input_ids, test_attention_masks])
    f_label = ner_labels


    val_tags_l = [ner_labels[x] for x in np.ravel(test_tags).astype(int).tolist()]
    y_predicted_l = [ner_labels[x] for x in np.ravel(np.argmax(y_predicted, axis=2)).astype(int).tolist()]

    f_label.remove("[PAD]")
    f_label.remove("O")
    #print(classification_report(val_tags_l, y_predicted_l, labels=f_label))
    pd.DataFrame([(results[0], results[1])], columns=('loss', 'acc')).to_csv('%s_%s_bio_evaluation.csv'%(taskname,re.sub('\/','',ptmodel_name)), index=False)
    with open('%s_%s_bio_classification_report.txt'%(taskname,re.sub('\/','',ptmodel_name)), 'w') as text_file:
        print(classification_report(val_tags_l, y_predicted_l, labels=f_label), file=text_file)

    # bi하나로 합친 태그에 대한 리포트
    from seqeval.metrics import classification_report as clr

    val_tags_l_bi_reverse = []
    y_predicted_l_bi_reverse = []
    for i in val_tags_l :
        if '-b' in i or '-i' in i:
            val_tags_l_bi_reverse.append(i.split('-')[1]+'-'+i.split('-')[0])
        elif 'PAD' in i:
            val_tags_l_bi_reverse.append('O')
        else:
            val_tags_l_bi_reverse.append(i)
    for i in y_predicted_l :
        if '-b' in i or '-i' in i:
            y_predicted_l_bi_reverse.append(i.split('-')[1]+'-'+i.split('-')[0])
        elif 'PAD' in i:
            y_predicted_l_bi_reverse.append('O')
        else:
            y_predicted_l_bi_reverse.append(i)

    with open('%s_%s_classification_report.txt' % (taskname,re.sub('\/', '', ptmodel_name)), 'w') as text_file:
        print(clr([val_tags_l_bi_reverse], [y_predicted_l_bi_reverse]), file=text_file)

    # #bi하나로 합친 태그에 대한 리포트
    # val_tags_l = [re.sub('-[bi]', '', ner_labels[x]) for x in np.ravel(test_tags).astype(int).tolist()]
    # y_predicted_l = [re.sub('-[bi]', '', ner_labels[x]) for x in
    #                  np.ravel(np.argmax(y_predicted, axis=2)).astype(int).tolist()]
    # f_label = [re.sub('-[bi]', '', x) for x in f_label]
    # f_label = list(set(f_label))
    # print(classification_report(val_tags_l, y_predicted_l, labels=f_label))
    # pd.DataFrame([(results[0], results[1])], columns=('loss', 'acc')).to_csv('%s_%s_evaluation.csv'%(taskname,re.sub('\/','',ptmodel_name)), index=False)
    # with open('%s_%s_classification_report.txt'%(taskname,re.sub('\/','',ptmodel_name)), 'w') as text_file:
    #     print(classification_report(val_tags_l, y_predicted_l, labels=f_label), file=text_file)

if __name__ == '__main__':

    # make_ner_model(taskname = 'maincat',
    #                model_upper_name = 'bert',
    #                ptmodel_name = 'bert-base-multilingual-cased',
    #                max_len = 169,
    #                batch_size = 2)
    #
    # make_ner_model(taskname = 'subcat',
    #                model_upper_name = 'bert',
    #                ptmodel_name = 'bert-base-multilingual-cased',
    #                max_len = 169,
    #                batch_size = 2)
    #
    # make_ner_model(taskname = 'maincat',
    #                model_upper_name = 'electra',
    #                ptmodel_name = 'monologg/koelectra-base-v3-discriminator',
    #                max_len = 169,
    #                batch_size = 2)

    # make_ner_model(taskname = 'subcat',
    #                model_upper_name = 'electra',
    #                ptmodel_name = 'monologg/koelectra-base-v3-discriminator',
    #                max_len = 169,
    #                batch_size = 2),
    make_ner_model(taskname='maincat',
                   model_upper_name='bert',
                   ptmodel_name='monologg/kobert',
                   max_len=169,
                   batch_size=2),
    make_ner_model(taskname='subcat',
                   model_upper_name='bert',
                   ptmodel_name='monologg/kobert',
                   max_len=169,
                   batch_size=2),
    make_ner_model(taskname='maincat',
                   model_upper_name='bert',
                   ptmodel_name='skt/kobert-base-v1',
                   max_len=169,
                   batch_size=2),
    make_ner_model(taskname='subcat',
                   model_upper_name='bert',
                   ptmodel_name='skt/kobert-base-v1',
                   max_len=16)