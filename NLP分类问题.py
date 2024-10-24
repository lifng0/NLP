from transformers import BertModel
import numpy as np
import torch
from torch import optim 
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import os

#模型载入
tokenizer = BertTokenizer.from_pretrained('C:/bert2/bert-base-uncased-vocab.txt')
bert = BertModel.from_pretrained('C:/bert2/bert-base-uncased/bert-base-uncased/')
bert = bert.cuda()


#数据处理
def dataprocess(data_path,lebal=True,batch_size=6,least=0,max=0):
    data = pd.read_csv(data_path,sep='\t',chunksize=batch_size,header=None)
    num = 0
    for chunk in data:
        num += batch_size
        print('loading',num) if num % 2000 == 0 else None
        if num >= least and num <= max:
            chunk.columns = ['id','example1','example2','results'] if lebal == True else ['id','example1','example2']
            attentionmask = []
            tokenid = []
            inputid = []
            input = chunk.drop('results',axis=1).drop('id',axis=1).values.tolist() if lebal == True else chunk.drop('id',axis=1).values.tolist()
            for i in range(len(input)):
                processed_data = tokenizer.encode_plus(input[i][0],input[i][1],padding='max_length',max_length=140)
                inputid.append(processed_data['input_ids'])
                tokenid.append(processed_data['token_type_ids'])
                attentionmask.append(processed_data['attention_mask'])
            result = chunk['results'].values.tolist() if lebal == True else 0
            if lebal == True:    
                for i in range(len(result)):
                    if result[i] == 'contradiction':
                        result[i] = 0
                    if result[i] == 'entailment':
                        result[i] = 1
                    else:
                        result[i] = 2
                yield [torch.tensor(inputid),torch.tensor(tokenid),torch.tensor(attentionmask)],torch.tensor(result)
            else:
                yield [torch.tensor(inputid),torch.tensor(tokenid),torch.tensor(attentionmask)]
#正确率判断
def get_accurancy(output,lebal,correct):
    dim0,dim1 = output.shape
    for i in range(dim0):
        if output[i].argmax() == lebal[i]:
            correct += 1
    return correct

#模型定义
class train_model(nn.Module):
    def __init__(self,bertmodel):
        super().__init__()
        self.bert = bertmodel
        self.outputs = nn.Sequential(nn.Dropout(0.1),
                                     nn.Linear(768,350),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     nn.Linear(350,3))
                                     
                                     
    def forward(self,data):
        id = data[0].cuda()
        segment_tensor = data[1].cuda()
        attentionmask = data[2].cuda()
        output = self.bert(id,attentionmask,segment_tensor)
        output1 = self.outputs(output[1])
        return output1

#loss，优化器选择
model = train_model(bert)   
model = model.cuda()
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
optimizer = optim.SGD(model.parameters(),lr=0.0001,weight_decay=0.01)
epochs = 9
#训练函数
def train(data_path,model,optimizer,epochs,batch_size):
    model.train()
    print('训练开始')
    everyloss = []
    everyaccurancy = []
    for i in range(epochs):
        torch.cuda.empty_cache()
        model.zero_grad()
        optimizer.zero_grad()
        total = 0
        runningloss = 0
        correct = 0
        dataset = dataprocess(data_path,batch_size=batch_size,max=10000)
        for inputdata,results in dataset:
            results = results.cuda()
            output = model.forward(inputdata)
            correct = get_accurancy(output,results,correct)
            loss = loss_fn(output,results)
            runningloss += loss.item()*batch_size
            total = total + batch_size
            loss.backward()
            optimizer.step()
        everyloss.append(runningloss/total)
        everyaccurancy.append(correct/total)
        print('已经训练',i+1,'次,loss:',everyloss[-1],'accurancy:',everyaccurancy[-1])
    plt.figure(figsize=(10,5),)
    plt.plot(range(epochs),everyloss,label='loss')
    plt.plot(range(epochs),everyaccurancy,label='accurancy')
    plt.xlabel('epochs')
    plt.ylabel('loss/accurancy')
    plt.legend()
    plt.show()
    everyloss.clear()
    everyaccurancy.clear()
#测试函数
def test(data_path,model,batch_size):
    print('测试开始')
    model.eval()
    runningloss = 0
    correct = 0
    total = 0
    dataset = dataprocess(data_path,batch_size=batch_size,least=10000,max=20000)
    for inputdata,results in dataset:
        results = results.cuda()
        output = model.forward(inputdata)
        correct = get_accurancy(output,results,correct)
        loss = loss_fn(output,results)
        runningloss += loss.item()*batch_size
        total += batch_size
    print('loss:',runningloss/total,'accurancy:',correct/total)
#执行函数
def run(data_path,data_path2,model,batch_size):
    print('写入开始')
    model.eval()
    outputlist = []
    dataset = dataprocess(data_path,lebal=False,batch_size=batch_size,max=20000)
    total = 0
    for inputdata in dataset:
        total += batch_size
        output = model.forward(inputdata)
        pred_output = torch.argmax(output,dim=1)
        for i in pred_output:
            if i == 0:
                output_lebal = 'contradiction'
            if i == 1:
                output_lebal = 'entailment'
            if i == 2:
                output_lebal = 'neutral'
            outputlist.append(output_lebal)
        if total % 1000 == 0:
            print('已经写入',total)
    print(len(outputlist))
    data = {'id':range(16000,20000),'label':outputlist}
    df = pd.DataFrame(data)
    df.head()
    df.to_csv(data_path2,index=0)
    

if __name__ == '__main__'  :
    train('C:\\python\\NLP\\train.txt',model,optimizer,epochs,30)
    test('C:\\python\\NLP\\train.txt',model,30)
    run('C:\\python\\NLP\\test.txt','C:\\python\\NLP\\submission.csv',model,30)
    



