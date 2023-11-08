from ast import arg
import timm
from pprint import pprint
import torch
from torchsummary import summary
import transformers
import torch.nn as nn
from collections import OrderedDict

from .vit_overlap import VisionTransformer



class ViT_Bert_Overlap_ShareHead(nn.Module):
    def __init__(self, args):
        super(ViT_Bert_Overlap_ShareHead, self).__init__()
        self.model_img = VisionTransformer(stride_size=args.stride_size)
        self.model_img.load_param('models/vit_base_patch16_384.pth')
        self.model_txt = transformers.BertModel.from_pretrained('bert-base-uncased')

        # cross-modal share head
        # self.share_head = nn.Linear(768, args.feature_size) 

        self.hidden_size = 4096
        self.share_head = nn.Sequential(*[
                        nn.Linear(768, self.hidden_size ), 
                        nn.BatchNorm1d(self.hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.hidden_size, 768), 
                        ])

    def forward(self, img, txt, mask):
        img_f4 = self.model_img(img)  
        txt_f4= self.model_txt(txt, mask)  

        img_out = self.share_head(img_f4)
        txt_out = self.share_head(txt_f4[1])

        return img_out, txt_out



class ViT_Bert_Overlap_IndependentHead(nn.Module):
    def __init__(self, args):
        super(ViT_Bert_Overlap_IndependentHead, self).__init__()
        self.model_img = VisionTransformer(stride_size=args.stride_size)
        self.model_img.load_param('models/vit_base_patch16_384.pth')
        self.model_txt = transformers.BertModel.from_pretrained('bert-base-uncased')

        # self.img_head = nn.Linear(768, args.feature_size) 
        # self.txt_head = nn.Linear(768, args.feature_size) 

        self.hidden_size = 4096
        self.img_head =  nn.Sequential(*[
                        nn.Linear(768, self.hidden_size ), 
                        nn.BatchNorm1d(self.hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.hidden_size, 768), 
                        ])
        self.txt_head = nn.Sequential(*[
                        nn.Linear(768, self.hidden_size ), 
                        nn.BatchNorm1d(self.hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.hidden_size, 768), 
                        ])
                        
    def forward(self, img, txt, mask):
        img_f4 = self.model_img(img)  
        txt_f4= self.model_txt(txt, mask)  

        img_out = self.img_head(img_f4)
        txt_out = self.txt_head(txt_f4[1])

        return img_out, txt_out


class ViT_Bert_Overlap_MutualProj(nn.Module):
    def __init__(self, args):
        super(ViT_Bert_Overlap_MutualProj, self).__init__()
        self.model_img = VisionTransformer(stride_size=args.stride_size)
        self.model_img.load_param('models/vit_base_patch16_384.pth')
        self.model_txt = transformers.BertModel.from_pretrained('bert-base-uncased')

        # cross-modal share head
        self.hidden_size = 4096
        self.img_head = nn.Sequential(*[
                        nn.Linear(768, self.hidden_size ), 
                        nn.BatchNorm1d(self.hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.hidden_size, 768), 
                        ])
        self.txt_head = nn.Sequential(*[
                        nn.Linear(768, self.hidden_size ), 
                        nn.BatchNorm1d(self.hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.hidden_size, 768), 
                        ])   

        # self.img_head = nn.Linear(768, args.feature_size) 
        # self.txt_head = nn.Linear(768, args.feature_size) 
    def forward(self, img, txt, mask):
        img_f4 = self.model_img(img)  
        txt_f4= self.model_txt(txt, mask)  

        img_proj = self.img_head(img_f4)
        txt_proj = self.txt_head(txt_f4[1])

        return img_f4, txt_f4[1], img_proj, txt_proj


if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description='command for train on CUHK-PEDES')
        parser.add_argument('--embedding_type', type=str,
                        default='BERT')
        args = parser.parse_args()
        return args
    args = parse_args()
    model = ViT_Bert_Overlap(args)
    # print(model)
    total = sum(p.numel() for p in model.model_img.parameters())
    print("Total params: %.2fM" % (total/1e6))
    total = sum(p.numel() for p in model.model_txt.parameters())
    print("Total params: %.2fM" % (total/1e6))

    img_demo = torch.zeros((2, 3, 384, 128))
    txt_demo = torch.zeros((2, 64), dtype=int)
    img_f4, txt_f4 = model(img_demo, txt_demo, txt_demo)
    print(img_f4.shape, txt_f4.shape)
