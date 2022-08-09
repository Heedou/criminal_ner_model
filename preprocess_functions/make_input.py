import re
import pandas as pd

def read_and_makedf(train_path, test_path):
    #train_path = './../../Traindata/NER_dataset/NERdataset/NER_maincat_train_5455.txt'
    #test_path = './../../Traindata/NER_dataset/NERdataset/NER_maincat_test_606.txt'

    "**********************************************input 만들기***********************************************************"
    # 데이터 읽어들임
    with open(train_path, encoding='utf-8') as f:
        train_data = f.readlines()

    with open(test_path, encoding='utf-8') as f:
        test_data = f.readlines()
    print(len(train_data))
    print(len(test_data))

    return train_data, test_data


def split_tagged_and_others(texts_list):
    """
    :param texts_list: ['<피의자 : tag>는 몇 월 몇 일 경~',...]
    :return: [['피의자 : tag', '', ''], [], ....]
    """
    # 삭제할 문자
    not_use_char = ['①②③④⑤⑥⑦⑧【】()○■ㆍ[]+=～▒▷♥♡*▣◎◈@△▽♤▼♣⊙▲★▦☆']
    splited_without_morpheme = []
    for i in range(len(texts_list)):
        text = texts_list[i]
        text = re.sub('[\①\②\③\④\⑤\⑥\⑦\⑧\【\】\○\■\ㆍ\[\]\+\=\～\▒\▷\♥\♡\*\▣\◎\◈\@\△\▽\♤\▼\♣\⊙\▲\★\▦\☆]', ' ', text)
        try:
            text = text.split('<')
            splited = []
            for t in text:
                f_sp = t.split('>')
                splited += f_sp
        except:
            splited = text

        for token in splited:
            if token == '' or token == ' ' or token == '  ':
                splited.remove(token)
        #print(splited)
        splited_without_morpheme.append(splited)
    return splited_without_morpheme

def make_BIO_df(splited_without_morpheme):
    """
    :param splited_without_morpheme: [['피의자 : tag', '', ''], [], ....]
    :return: 데이터프레임형태
                sentence 컬럼 : '피의자 어쩌구 저쩌구'
                label 컬럼 : 'ps-o o o'
    """
    result = []
    for token_bio in splited_without_morpheme:
        tokens = ''
        tags = ''
        for token in token_bio:
            try:
                if token[0] == ' ':
                    token = token[1:]
            except:
                pass
            try:
                if token[-1] == ' ':
                    token = token[:-1]
            except:
                pass

            token = re.sub('  ', '', token)

            bio_search = re.findall(': [a-z]{2,3}', token)
            if len(bio_search) > 0:
                try:
                    split_what = re.findall('( : )([a-z].*?)', token)
                    plus_what = split_what[0][1]
                    split_what = split_what[0][0] + split_what[0][1]
                    token = token.split(split_what)
                    token[1] = plus_what + token[1]
                    if ' ' not in token[0]:
                        tokens += '%s ' % token[0]
                        tags += '%s-b ' % token[1]
                    elif ' ' in token[0]:
                        token_token = token[0].split(' ')
                        for t_t in range(len(token_token)):
                            if t_t != len(token_token) - 1:
                                tokens += '%s ' % token_token[t_t]
                                tags += '%s-b ' % token[1]
                            if t_t == len(token_token) - 1:
                                tokens += '%s ' % token_token[t_t]
                                tags += '%s-i ' % token[1]
                except:
                    pass
            else:
                len_o_token = len(token.split(' '))
                tokens += '%s ' % token
                tags += 'O ' * (len_o_token)
        if len(tokens.split(' ')) == len(tags.split(' ')):
            # print(tokens, tags)
            result.append((tokens, tags))
        elif len(tokens.split(' ')) != len(tags.split(' ')):
            print('tokenization failed')

    cleaned_df = pd.DataFrame(result, columns=('sentence', 'label'))
    return cleaned_df

def make_ner_labels(df, label_column):
    # Label 불러오기
    labels = [i.split(' ') for i in df[label_column]]
    ner_labels = []
    for label in labels:
        ner_labels += label

    ner_labels = list(set(ner_labels))
    ner_labels.remove('')
    ner_labels.append('[PAD]')

    ner_begin_label = [ner_labels.index(begin_label) for begin_label in ner_labels if "b" in begin_label]

    return ner_begin_label, ner_labels

def make_cls_sep_sentences(ner_labels, df):
    sentences = []
    targets = []
    for i in range(len(df)):
      sentence = []
      target = []
      sentence.append('[CLS]')
      target.append(ner_labels.index('O'))
      texts = df['sentence'][i].split(' ')
      labels = df['label'][i].split(' ')
      for t, g in zip(texts, labels):
        if t != '':
          sentence.append(t)
          target.append(ner_labels.index(g))
      sentence.append('[SEP]')
      target.append(ner_labels.index('O'))
      sentences.append(sentence)
      targets.append(target)

    return sentences, targets

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
  tokenized_sentence = []
  labels = []

  for word, label in zip(sentence, text_labels):

    tokenized_word = tokenizer.tokenize(word)
    n_subwords = len(tokenized_word)

    tokenized_sentence.extend(tokenized_word)
    labels.extend([label] * n_subwords)

  return tokenized_sentence, labels