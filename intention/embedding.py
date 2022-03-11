import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import logging
from model.Bert.bert_train import create_logger
from config import data_path, emb_log_path
import jieba


class Embedding(object):
    def __init__(self, sent_path):
        self.train_data = []
        self.data_path = sent_path
        self.use_model = None
        self.tfidf_model = None
        self.tfidf = None
        self.w2v_model = None
        self.fasttext_model = None
        self.prepare_data()

    def prepare_data(self):
        logging.info('prepare the embedding data')
        with open(self.data_path, 'r') as f:
            self.train_data.append(f.readline().split('@@')[0])
        logging.info('the embedding data has been prepared')

    def train(self, model_name, save_model=True):
        logging.info('train embedding')
        self.use_model = model_name
        if self.use_model == 'tfidf':
            self.tfidf_model = TfidfVectorizer(max_df=0.4,
                                               min_df=0.001,
                                               ngram_range=(1, 2))
            self.tfidf = self.tfidf_model.fit(self.train_data)
        elif self.use_model == 'w2v':
            self.w2v_model = gensim.models.Word2Vec(min_count=2,
                                                    window=5,
                                                    sample=6e-5,
                                                    alpha=0.03,
                                                    min_alpha=0.0007,
                                                    negative=15,
                                                    workers=4,
                                                    max_vocab_size=50000)
            self.w2v_model.build_vocab(self.train_data)
            self.w2v_model.train(self.train_data,
                                 total_examples=self.w2v_model.corpus_count,
                                 epochs=15,
                                 report_delay=1)
        elif self.use_model == 'fasttext':
            self.fasttext_model = gensim.models.FastText(self.train_data,
                                                         window=3,  # 移动窗口
                                                         alpha=0.03,
                                                         min_count=2,  # 对字典进行截断, 小于该数的则会被切掉,增大该值可以减少词表个数
                                                         max_n=3,
                                                         word_ngrams=2,
                                                         max_vocab_size=50000)
            self.fasttext_model.train(self.train_data,
                                      total_examples=self.w2v_model.corpus_count,
                                      epochs=30)
        logging.info('finish training')
        if save_model:
            self.save_model()
            logging.info('embedding_model has been saved')

    def save_model(self):
        if self.use_model == 'tfidf':
            logging.info('save tfidf embedding_model')
            joblib.dump(self.tfidf, 'embedding_model/tfidf')
        elif self.use_model == 'w2v':
            logging.info('save word2vec embedding_model')
            self.w2v_model.wv.save_word2vec_format('embedding_model/wv2.bin')
        elif self.use_model == 'fasttext':
            logging.info('save fasttext embedding_model')
            self.fasttext_model.wv.save_word2vec_format('embedding_model/fasttext.bin')

    def load_model(self):
        if self.use_model == 'tfidf':
            joblib.load(self.tfidf, 'embedding_model/tfidf')
            logging.info('tfidf embedding_model has been loaded')
        elif self.use_model == 'w2v':
            self.w2v_model.wv.load_word2vec_format('embedding_model/wv2.bin')
            logging.info('w2v embedding_model has been loaded')
        elif self.use_model == 'fasttext':
            self.fasttext_model.wv.load_word2vec_format('embedding_model/fasttext.bin')
            logging.info('fasttext embedding_model has been loaded')

    def get_similarity(self, text):
        sentence = " ".join(jieba.cut(text))
        if self.use_model == 'tfidf':
            joblib.load(self.tfidf, 'embedding_model/tfidf')
            logging.info('tfidf embedding_model has been loaded')
        elif self.use_model == 'w2v':
            self.w2v_model.wv.load_word2vec_format('embedding_model/wv2.bin')
            logging.info('w2v embedding_model has been loaded')
            self.w2v_model
        elif self.use_model == 'fasttext':
            self.fasttext_model.wv.load_word2vec_format('embedding_model/fasttext.bin')
            logging.info('fasttext embedding_model has been loaded')


if __name__ == "__main__":
    emb_logger = create_logger(emb_log_path)
    emb = Embedding(data_path)
    emb.train("w2v")
    emb.load_model()
    emb.get_similarity("你叫什么名字")