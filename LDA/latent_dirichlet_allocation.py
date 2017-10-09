# !/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import re
import jieba
from gensim import corpora, models

reload(sys)
sys.setdefaultencoding('utf-8')

def format_data(data_filename, new_data_filename):
  data_list = []
  with open(data_filename, 'r') as data_file:
    for line in data_file:
      if not isinstance(line, unicode):
        line = line.decode('utf-8')
      line = ' '.join(line.strip().split())
      line = re.sub(ur'(\u200b)|(\s+)|(\\n)|(http:\\/\\/t\.cn\\/[0-9a-zA-Z \.]*)', ' ', line)
      data_list.append(line)
  with open(new_data_filename, 'w') as data_file:
    for data in data_list:
      data_file.write(data.encode('utf-8') + '\n')

def lda_model():
  stop_words = []
  with open('stopwords.txt', 'r') as stopword_file:
    for line in stopword_file:
      word = line.strip()
      if line:
        stop_words.append(word)
  train_set = []
  with open('sina_finance.txt') as data_file:
    for line in data_file:
      segment_words = jieba.cut(line.strip())
      train_set.append([word.strip() for word in segment_words if word.strip() and word.strip() not in stop_words])
  # with open('weibo_movie.txt') as data_file:
  #   for line in data_file:
  #     segment_words = jieba.cut(line.strip())
  #     train_set.append([word.strip() for word in segment_words if word.strip() and word.strip() not in stop_words])

  dictionary = corpora.Dictionary(train_set)
  dictionary.save('lda.dict')
  # corpus: [[(token_id, token_count), (token_id, token_count)], [(token_id, token_count), (token_id, token_count)]]
  corpus = [dictionary.doc2bow(text) for text in train_set]
  lda = models.LdaModel(corpus, id2word=dictionary, num_topics=2, alpha=1, iterations=500)
  lda.save('lda.pkl')
  print 'LDA Topics:'
  for topic in lda.print_topics(2):
    print (', '.join([str(word) for word in topic])).decode('string_escape')
  print '---------------------------------------'
  for index, score in sorted(lda[corpus[0]], key=lambda value: value[1], reverse=True):
    print 'Score: {}, Topic: {}'.format(score, lda.print_topic(index))
  print '---------------------------------------'


  dictionary = corpora.Dictionary.load('lda.dict')
  lda = models.LdaModel.load('lda.pkl')
  query = '去看 吴京 电影 战狼'
  # query = 'use machine learning to classification tagging'
  query_bow = dictionary.doc2bow(jieba.cut(query))
  for index, score in sorted(lda[query_bow], key=lambda value: value[1], reverse=True):
    print 'Score: {}, Topic: {}'.format(score, lda.print_topic(index))
  print 'end'


def main():
  lda_model()

if __name__ == '__main__':
  main()