import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext import transforms as T
from torchtext.vocab import GloVe

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


# 读入acimbd库
def read_acimbd(is_train, path='./aclImdb_v1/aclImdb'):
    review_list = []
    label_list = []

    # 定义一个标记器去分割英语单词
    tokenizer = get_tokenizer('basic_english')

    # 定义路径
    if is_train:
        valid_file = os.path.join(path, 'train')
    else:
        valid_file = os.path.join(path, 'test')

    valid_file_pos = os.path.join(valid_file, 'pos')
    valid_file_neg = os.path.join(valid_file, 'neg')

    # 遍历所有的positive文件，用tokenizer进行分割
    for filename in os.listdir(valid_file_pos):
        with open(os.path.join(valid_file_pos, filename), 'r', encoding='utf-8') as file_content:
            review_list.append(tokenizer(file_content.read()))
            label_list.append(1)

    # 遍历所有的negative文件
    for filename in os.listdir(valid_file_neg):
        with open(os.path.join(valid_file_neg, filename), 'r', encoding='utf-8') as file_content:
            review_list.append(tokenizer(file_content.read()))
            label_list.append(0)

    return review_list, label_list


# 将review_list与label_list经过vocab的index转化后用TensorDataset打包
def build_dataset(review_list, label_list, _vocab, max_len=512):
    # 建立一个词表转化,vocab里存储词汇与它的唯一标签，利用VocabTransform将词汇转化为对应的数字
    # 利用Truncate将所有的句子的最长长度限制在了max_len
    # 利用ToTensor将所有的句子按照此时最长的句子进行填充为张量，填充的内容为'<pad>'对应的数字
    # 利用PadTransform将ToTensor里所有的句子长度均填充为max_len
    seq_to_tensor = T.Sequential(
        T.VocabTransform(vocab=_vocab),
        T.Truncate(max_seq_len=max_len),
        T.ToTensor(padding_value=_vocab['<pad>']),
        T.PadTransform(max_length=max_len, pad_value=_vocab['<pad>'])
    )
    dataset = TensorDataset(seq_to_tensor(review_list), torch.tensor(label_list))
    return dataset


# 输入控制词的最小出现频率，设置字典
def load_acimdb(min_freq=3):
    review_train_list, label_train_list = read_acimbd(is_train=True)
    review_test_list, label_test_list = read_acimbd(is_train=False)
    _vocab = build_vocab_from_iterator(review_train_list, min_freq=min_freq, specials=['<pad>', '<unk>'])
    # 设置未登录词的索引
    _vocab.set_default_index(_vocab['<unk>'])
    dataset_train = build_dataset(review_train_list, label_train_list, _vocab=_vocab)
    dataset_test = build_dataset(review_test_list, label_test_list, _vocab=_vocab)
    return dataset_train, dataset_test, _vocab


# 定义了一个普通的rnn网络
class MyRNN(nn.Module):
    def __init__(self, _vocab, embed_size=300, hidden_size=512, num_layers=2, dropout=0.1, use_glove=False,
                 bidirectional=False, use_xavier=True):
        super(MyRNN, self).__init__()
        self.bidirectional = bidirectional
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.fc_single_direction = nn.Linear(hidden_size, 2)
        self.fc_bidirectional = nn.Linear(2*hidden_size, 2)

        # 初始化参数，将rnn层进行xavier参数化，全连接层进行N(0,1)初始化
        if use_xavier:
            for parameter in self.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
                elif parameter.dim() == 1:
                    nn.init.normal_(parameter)

        # 使用预训练的词向量
        if use_glove:
            # 利用torchtext.vocab.vector下载Glove库
            glove = GloVe(name="6B", dim=embed_size)
            self.embedding = nn.Embedding.from_pretrained(glove.get_vecs_by_tokens(_vocab.get_itos()),
                                                          padding_idx=_vocab['<pad>'])
        else:
            # 利用N(0,1)的embedding层
            self.embedding = nn.Embedding(len(_vocab), embed_size, padding_idx=_vocab['<pad>'])

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)
        _, h_n = self.rnn(x)
        if self.bidirectional:
            output = self.fc_bidirectional(torch.cat((h_n[-1], h_n[-2]), dim=-1))
        else:
            output = self.fc_single_direction(h_n[-1])
        return output


# 定义了一个LSTM网络
class LSTM(nn.Module):
    def __init__(self, _vocab, embed_size=300, hidden_size=512, num_layers=2, dropout=0.1, use_glove=False,
                 bidirectional=False, use_xavier=True):
        super(LSTM, self).__init__()
        self.bidirectional = bidirectional
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.fc_single_direction = nn.Linear(hidden_size, 2)
        self.fc_bidirectional = nn.Linear(2 * hidden_size, 2)

        # 初始化参数，将rnn层进行xavier参数化，全连接层进行N(0,1)初始化
        if use_xavier:
            for parameter in self.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
                elif parameter.dim() == 1:
                    nn.init.normal_(parameter)

        # 使用预训练的词向量
        if use_glove:
            glove = GloVe(name="6B", dim=embed_size)
            # vocab.get_itos得到原先的所有词汇，再从glove中取得这个词汇的向量表示
            self.embedding = nn.Embedding.from_pretrained(glove.get_vecs_by_tokens(_vocab.get_itos()),
                                                          padding_idx=_vocab['<pad>'])
        else:
            # 利用N(0,1)的embedding层
            self.embedding = nn.Embedding(len(_vocab), embed_size, padding_idx=_vocab['<pad>'])

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)
        _, (h_n, _) = self.rnn(x)
        if self.bidirectional:
            output = self.fc_bidirectional(torch.cat((h_n[-1], h_n[-2]), dim=-1))
        else:
            output = self.fc_single_direction(h_n[-1])
        return output


# 设置随机数种子
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 设置模型保存路径
def set_model_save_path(learning_rate, epoch_num, hidden_size, use_LSTM, bidirectional, use_glove):
    if use_LSTM:
        path = './lstm_result/lstm_' + 'rate' + str(learning_rate) + '_epoch' + str(epoch_num) + '_hidden' + \
               str(hidden_size)
    else:
        path = './rnn_result/rnn_' + 'rate' + str(learning_rate) + '_epoch' + str(epoch_num) + '_hidden' + \
               str(hidden_size)
    path += '_direct' + (str(2) if bidirectional else str(1))
    path += '_glove' + (str(1) if use_glove else str(0)) + '.pth'
    return path


# 设置图片保存路径
def set_pic_save_path(learning_rate, epoch_num, hidden_size, use_LSTM, bidirectional, use_glove):
    if use_LSTM:
        path = './draw_result/lstm_' + 'rate' + str(learning_rate) + '_epoch' + str(epoch_num) + '_hidden' + \
               str(hidden_size)
    else:
        path = './draw_result/rnn_' + 'rate' + str(learning_rate) + '_epoch' + str(epoch_num) + '_hidden' + \
               str(hidden_size)
    path += '_direct' + (str(2) if bidirectional else str(1))
    path += '_glove' + (str(1) if use_glove else str(0)) + '.png'
    return path


init_seeds(0)
# 设置跑的平台是CPU还是GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 128
learning_rate = 0.001
epoch_num = 20

dataset_train, dataset_test, _vocab = load_acimdb(min_freq=2)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

use_glove = True
use_xavier = True
bidirectional = False
use_LSTM = True
embed_size = 300
hidden_size = 256
if use_LSTM:
    model = LSTM(_vocab, use_glove=use_glove, bidirectional=bidirectional, embed_size=embed_size,
                 hidden_size=hidden_size, use_xavier=use_xavier).to(device)
else:
    model = MyRNN(_vocab, use_glove=use_glove, bidirectional=bidirectional, embed_size=embed_size,
                  hidden_size=hidden_size, use_xavier=use_xavier).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

path_model = set_model_save_path(learning_rate, epoch_num, hidden_size, use_LSTM, bidirectional, use_glove)
path_pic = set_pic_save_path(learning_rate, epoch_num, hidden_size, use_LSTM, bidirectional, use_glove)

# 用这个变量去调控是否使用已经保存的模型
use_trained_model = False
if use_trained_model:
    model.load_state_dict(torch.load(path_model))
else:
    train_loss_per_epoch = []
    for epoch in range(epoch_num):
        print(f'epoch {epoch + 1}:')
        total_loss = 0
        batch_idx = 0
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader_train):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            predict_y = model(batch_x)
            loss = loss_func(predict_y, batch_y)
            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx+1) % 5 == 0:
                print(f"loss on batch_idx {batch_idx} is: {loss:.6f}")

        total_loss /= (batch_idx + 1)
        train_loss_per_epoch.append(total_loss.item())
        print(f"loss on train set: {total_loss:.6f}\n")

    torch.save(model.state_dict(), path_model)

    # 画图展示训练的epoch过程中的loss变化
    draw_x = list(range(1, len(train_loss_per_epoch) + 1))
    plt.plot(draw_x, train_loss_per_epoch)
    plt.title('loss change in training set')
    plt.xlabel('epoch num')
    plt.ylabel('loss')
    plt.savefig(path_pic)
    plt.show()

# 在测试集上的结果
acc = 0
for batch_x, batch_y in dataloader_test:
    with torch.no_grad():
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        predict_y = model(batch_x)
        acc += (torch.argmax(predict_y, dim=1) == batch_y).sum().item()
print(f"accuracy: {acc / len(dataset_test):.6f}")
