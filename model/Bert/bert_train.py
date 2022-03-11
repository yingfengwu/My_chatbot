import torch
from tqdm import tqdm
import os
import logging
import sys

from config import batch_size, lr, data_path, weibo_dev_data_path, weibo_test_data_path, weibo_train_data_path, \
    root_path, bert_chinese_model_path, log_path, max_length, max_grad_norm, gradient_accumulation
from bert_seq2seq import Seq2SeqModel
from bert_model import BertConfig
import time
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer, load_chinese_base_vocab
import matplotlib.pyplot as plt
from sklearn import metrics

sys.path.append('..')

global logger


def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def read_corpus(data_path):
    lines = open(data_path, encoding='utf-8').read().strip().split('\n')

    # 数据处理，处理成一对对的样本
    pairs = [[s for s in l.split("@@")] for l in lines]

    sents_src = []
    sents_tgt = []
    for query, answer in pairs:
        sents_src.append(query)
        sents_tgt.append(answer)
    return sents_src, sents_tgt


# 自定义dataset
class SelfDataset(Dataset):
    """
    针对数据集，定义一个相关的取数据的方式
    """
    def __init__(self, path):
        # 一般init函数是加载所有数据
        super(SelfDataset, self).__init__()
        # 读原始数据
        self.sents_src, self.sents_tgt = read_corpus(path)
        self.word2idx = load_chinese_base_vocab()
        self.idx2word = {k: v for v, k in self.word2idx.items()}
        self.tokenizer = Tokenizer(self.word2idx)

    def __getitem__(self, i):
        # 得到单个数据

        src = self.sents_src[i] if len(self.sents_src[i]) < max_length else self.sents_src[i][:max_length]
        tgt = self.sents_tgt[i] if len(self.sents_tgt[i]) < max_length else self.sents_tgt[i][:max_length]

        token_ids, token_type_ids = self.tokenizer.encode(src, tgt)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
        }
        return output

    def __len__(self):

        return len(self.sents_src)


def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """
    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        注意 token type id 右侧pad是添加1而不是0，1表示属于句子B
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded


class Trainer:
    def __init__(self):
        self.pretrain_model_path = bert_chinese_model_path
        self.batch_size = batch_size
        self.lr = lr
        logger.info('加载字典')
        self.word2idx = load_chinese_base_vocab()
        # 判断是否有可用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info('using device:{}'.format(self.device))
        # 定义模型超参数
        bertconfig = BertConfig(vocab_size=len(self.word2idx))
        logger.info('初始化BERT模型')
        self.bert_model = Seq2SeqModel(config=bertconfig)
        logger.info('加载预训练的模型～')
        self.load_model(self.bert_model, self.pretrain_model_path)
        logger.info('将模型发送到计算设备(GPU或CPU)')
        self.bert_model.to(self.device)
        logger.info(' 声明需要优化的参数')
        self.optim_parameters = list(self.bert_model.parameters())
        self.init_optimizer(lr=self.lr)
        # 声明自定义的数据加载器

        logger.info('加载训练数据')
        train = SelfDataset(data_path)
        # dev = SelfDataset(weibo_dev_data_path)
        # test = SelfDataset(weibo_test_data_path)

        # logger.info('加载测试数据')
        # dev = SelfDataset(os.path.join(root_path, 'data/chatdata_all.txt'))
        self.trainloader = DataLoader(train, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        # self.devloader = DataLoader(dev, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        # self.testloader = DataLoader(test, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

        self.plot_loss = []
        self.plot_epoch = []

    def init_optimizer(self, lr):
        # 用指定的学习率初始化优化器
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)

    def load_model(self, model, pretrain_model_path):

        checkpoint = torch.load(pretrain_model_path)
        # 模型刚开始训练的时候, 需要载入预训练的BERT

        checkpoint = {k[5:]: v for k, v in checkpoint.items()
                      if k[:4] == "bert" and "pooler" not in k}
        model.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        logger.info("{} loaded!".format(pretrain_model_path))

    def train(self, epoch):
        # 一个epoch的训练
        logger.info('starting training')
        self.bert_model.train()
        self.iteration(epoch, dataloader=self.trainloader)
        logger.info('training finished')

    def iteration(self, epoch, dataloader):
        total_loss = 0
        start_time = time.time() ## 得到当前时间
        for batch_idx, data in enumerate(tqdm(dataloader, position=0, leave=True)):
            # torch.cuda.empty_cache()
            token_ids, token_type_ids, target_ids = data
            token_ids = token_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            # 因为传入了target标签，因此会计算loss并且返回
            enc_layers, logits, loss, attention_layers = self.bert_model(token_ids,
                                                token_type_ids,
                                                labels=target_ids
                                                )
            # loss = loss / gradient_accumulation
            loss.backward()
            # 更新参数
            self.optimizer.step()
            # 清空梯度信息
            self.optimizer.zero_grad()
            # if (batch_idx + 1) % gradient_accumulation == 0:  # 每多少次输出在训练集和校验集上的效果
            #     # 为计算当前epoch的平均loss
            total_loss += loss.item()
            # 当梯度太多需要进行梯度裁剪，当梯度超过一个阈值，除以最大范数
            torch.nn.utils.clip_grad_norm_(self.bert_model.parameters(), max_grad_norm)

        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        logger.info(f"epoch is {epoch}. loss is {total_loss/batch_idx:06}. spend time is {spend_time}")
        self.plot_loss.append(total_loss/batch_idx)
        self.plot_epoch.append(epoch)
        if (epoch + 1) % 500 == 0:
            # 保存模型
            self.save_state_dict(self.bert_model, epoch+1)

    def evaluate(self):
        logger.info("start evaluating embedding_model")
        self.bert_model.eval()
        logger.info('starting evaluating')
        batch_idx = 0
        loss_total = 0
        predict_all = []
        labels_all = []
        with torch.no_grad():
            for token_ids, token_type_ids, target_ids in tqdm(self.devloader,position=0, leave=True):
                batch_idx += 1
                token_ids = token_ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                enc_layers, logits, loss = self.bert_model(token_ids,
                                                token_type_ids,
                                                labels=target_ids
                                                )
                # loss, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids, device=device)
                loss_total += loss
                predict_all.append(logits)
                labels_all.append(target_ids)
                logger.info("evaluate batch {} ,loss {}".format(batch_idx, loss))
        logger.info("finishing evaluating")
        acc = metrics.accuracy_score(labels_all, predict_all)

        return acc, loss_total/batch_idx

    def save_state_dict(self, model, epoch, file_path="saved_model/bert.model"):
        """存储当前模型参数"""
        save_path = os.path.join(root_path, file_path + ".epoch.{}".format(str(epoch)))
        torch.save(model.state_dict(), save_path)
        logger.info("{} saved!".format(save_path))


if __name__ == "__main__":
    logger = create_logger(log_path)
    trainer = Trainer()
    train_epoches = 1000

    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)

    # plot the loss
    fig, ax = plt.subplots()
    ax.plot(trainer.plot_epoch, trainer.plot_loss)
    ax.grid()
    fig.savefig("_loss.png")
    plt.show()
