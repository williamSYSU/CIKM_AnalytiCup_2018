import torch
import torch.nn as nn
import unicodedata
import re
import random
import torch.optim as optim
import torch.nn.functional as F
embedding = open('./data/wiki.es.vec' , encoding='utf-8').\
        read().strip().split('\n')
word_to_embedding={}
for i in range(1,int(embedding[0].split(' ')[0])):
        tmp=embedding[i].split(' ')
        word_to_embedding[tmp[0]]=torch.tensor([float(item) for item in tmp[1:301]])
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", s)
    return s
def tensorFromSentence( sentence):
    tensors = [word_to_embedding[word] for word in sentence.split(' ')]
    tensor=tensors[0].view(1,-1)
    for i in range(1,len(tensors)):
        tensor=torch.cat([tensor,tensors[i].view(1,-1)],dim=0)
    return tensor

def tensorsFromPair(pair):
    input1_tensor = tensorFromSentence( pair[0])
    input2_tensor = tensorFromSentence(pair[2])
    input_tensor=torch.cat((input1_tensor,input2_tensor),dim=0)
    label=pair[4]
    return (input_tensor,label)
def tensorsFromPair_test(pair):
    input1_tensor = tensorFromSentence( pair[0])
    input2_tensor = tensorFromSentence( pair[1])
    input_tensor = torch.cat((input1_tensor, input2_tensor), dim=0)
    return (input_tensor)
lines = open('data/cikm_spanish_train_20180516.txt' , encoding='utf-8').\
        read().strip().split('\n')
pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
class bi_lstm(nn.Module):
    def __init__(self):
        super(bi_lstm, self).__init__()
        self.bi_lstm_context =nn.LSTM(300, 100,bidirectional=True)
        self.dense = nn.Linear(400, 2)
        self.stm=nn.Softmax(dim=0)
    def forward(self,input):
        out,(_,_)=self.bi_lstm_context(input)
        print ("out1:",out)
        a=torch.cat((out[0][0],out[-1][0]),dim=0)
        print ("a:",a)
        out=self.dense(a)
        print ("out2:", out)
        out=self.stm(out)
        print ("out3:", out)
        return out
lstm=bi_lstm()
loss_function =nn.BCELoss()
optimizer = optim.SGD(lstm.parameters(), lr=0.01)
for epoch in range(100):
    for pair in pairs:
         lstm.zero_grad()
         training_pair = [tensorsFromPair(pair)]
         # print (training_pair)
         tag_scores=lstm(training_pair[0][0])
         label=training_pair[0][1]
         if label=='1':
             label=torch.tensor([1],dtype=torch.float)
         else:
             label=torch.tensor([0],dtype=torch.float)
         loss=loss_function(tag_scores[0].view(-1),label)
         print (loss)
         loss.backward()
         optimizer.step()