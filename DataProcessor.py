import torch.utils.data as Data
from torchvision import transforms
from tqdm import tqdm
import pickle
import os
from PIL import Image

import spacy
import numpy as np
# 加载Spacy模型
nlp = spacy.load('en_core_web_sm')
# def get_adjacency_matrix(text):
#     # 对文本进行依赖关系分析
#     doc = nlp(text)
#     # 获取文本中的词语数量
#     num_tokens = len(doc)
#     # 创建空的邻接矩阵
#     adjacency_matrix = np.zeros((num_tokens, num_tokens))
#     # 填充邻接矩阵
#     for token in doc:
#         # 获取当前词语的索引和依赖关系的头部索引
#         token_index = token.i
#         head_index = token.head.i
#         # 设置邻接矩阵中对应位置为1，表示存在依赖关系
#         adjacency_matrix[token_index][head_index] = 1
#     return adjacency_matrix

def image_process(image_path, itransform):
    image = Image.open(image_path).convert('RGB')
    image = itransform(image)
    return image

class MyDataset(Data.Dataset):
    def __init__(self,data_dir,imagefeat_dir,tokenizer,max_seq_len,img_feat_dim=2048,crop_size=224):
        self.imagefeat_dir=imagefeat_dir
        self.tokenizer=tokenizer
        self.sentiment_label_list=self.get_sentiment_labels()
        self.max_seq_len=max_seq_len
        self.examples=self.creat_examples(data_dir)
        self.number = len(self.examples)
        self.img_feat_dim = img_feat_dim
        self.crop_size = crop_size
        self.data_dir = data_dir

    def __len__(self):
        return self.number
    def __getitem__(self,index):
        line=self.examples[index]
        return self.transform(line,index)   

    def creat_examples(self,data_dir):
        with open(data_dir,"rb") as f:
            dict=pickle.load(f)
        examples=[]
        for key,value in tqdm(dict.items(),desc="CreatExample"):
            examples.append(value)
        return examples

    def get_sentiment_labels(self):
        return ["0","1","2"]

#读取并处理数据，比如打开.graph文件，去除对应样本的邻接矩阵，并用 np.pad 填充到模型需要的 max_seq_len（返回的 adj 即传入模型的邻接矩阵）。
    def transform(self,line,index):
        max_seq_len =self.max_seq_len
        value=line
        text_a = value['sentence'] 
        text_b = value['aspect']
        graph_id = index
        filename = self.data_dir.rstrip('.pkl')
        fin = open(filename + '.graph', 'rb')
        idx2graph = pickle.load(fin)
        fin.close()
        # 原始邻接矩阵为文本长度 (<= max_seq_len)
        text_adj = np.pad(idx2graph[graph_id], \
                                  ((0, max_seq_len - idx2graph[graph_id].shape[0]),
                                   (0, max_seq_len - idx2graph[graph_id].shape[0])), 'constant')
        # 将图像区域也作为图节点加入构图；默认区域数为49（与 Faster R-CNN 保持一致）
        num_image_regions = 49
        # 构造联合邻接矩阵：
        # [ text | image ]
        # text-text: 原始 dependency 矩阵
        # image-image: 自环（identity）
        # cross (text-image / image-text): 全连接（可促进跨模态信息流通）
        total_len = max_seq_len + num_image_regions
        new_adj = np.zeros((total_len, total_len), dtype='float32')
        # text-text
        new_adj[:max_seq_len, :max_seq_len] = text_adj
        # image-image 自环
        new_adj[max_seq_len:, max_seq_len:] = np.eye(num_image_regions, dtype='float32')
        # cross 模态连接由模型在运行时根据 attention 权重构建；这里先保持为 0（即不加入跨模态连边）
        # 这样可以让 model.forward 动态填充 cross-block（更灵活，也避免在 DataProcessor 里依赖模型计算）
        # new_adj[:max_seq_len, max_seq_len:] = 0.0
        # new_adj[max_seq_len:, :max_seq_len] = 0.0
        adj = new_adj
        target_ids = self.tokenizer(text_b.lower())['input_ids']  # <s>text_b</s>
        target_mask = [1] * len(target_ids)
        input_ids=self.tokenizer(text_a.lower(),text_b.lower())['input_ids']   #  <s>text_a</s></s>text_b</s>
        input_mask=[1]*len(input_ids)
        padding_id = [1]*(max_seq_len-len(input_ids)) #<pad> :1
        padding_mask=[0]*(max_seq_len-len(input_ids))
        input_ids += padding_id
        input_mask += padding_mask
        tokens=self.tokenizer.decode(input_ids)
        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len


        padding_idt = [1] * (max_seq_len - len(target_ids))  # <pad> :1
        padding_maskt = [0] * (max_seq_len - len(target_ids))
        target_ids += padding_idt
        target_mask += padding_maskt
        tokenst = self.tokenizer.decode(target_ids)
        assert len(target_ids) == max_seq_len
        assert len(target_mask) == max_seq_len

        img_id = value['iid']
        img_feat = read_pic(self.imagefeat_dir,img_id,self.crop_size)
        sentiment_label=-1
        sentiment_label_map = {label: i for i, label in enumerate(self.sentiment_label_list)}
        senti=value['sentiment']
        if senti:
            sentiment_label=sentiment_label_map[senti]

        return tokens,tokenst,input_ids,input_mask, target_ids,target_mask, sentiment_label,img_id,img_feat,adj



def read_pic(imagefeat_dir,img_id,crop_size):
    itransform = transforms.Compose([
        transforms.RandomCrop(crop_size),  # args.crop_size, by default it is set to be 224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    if 'twitter' in imagefeat_dir.lower():
        img_id2 = img_id + '.jpg'
        image_path = os.path.join(imagefeat_dir, img_id2)

    if not os.path.exists(image_path):
        print(image_path)
    try:
        img_feat = image_process(image_path, itransform)
    except:
        # count += 1
        # print('image has problem!')
        # Fallback image logic
        if 'twitter2017' in imagefeat_dir:
             image_path_fail = os.path.join(imagefeat_dir, '17_06_4705.jpg')
        else:
             # For twitter2015, we need a valid fallback image that exists in that dataset
             # Let's try to find one that exists, or just use a black image if really needed
             # But better to use a real image from the set. 
             # Assuming '1365.jpg' exists in 2015 set (it's a common ID format)
             # Or we can just list the dir and pick the first one.
             
             # Let's try to be safer.
             all_files = os.listdir(imagefeat_dir)
             if len(all_files) > 0:
                 image_path_fail = os.path.join(imagefeat_dir, all_files[0])
             else:
                 # This is bad, empty dir
                 raise Exception("Image directory is empty")
                 
        img_feat = image_process(image_path_fail, itransform)
    return img_feat



#调用spaCy 对句子做依赖分析并生成邻接矩阵
def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1
    return matrix


