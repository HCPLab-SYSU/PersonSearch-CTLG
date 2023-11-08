import timm
from pprint import pprint
import torch
from torchsummary import summary
import transformers
import torch.nn as nn
from collections import OrderedDict

from .vit_overlap import VisionTransformer


class ViT_Bert_Overlap(nn.Module):
    def __init__(self, args):
        super(ViT_Bert_Overlap, self).__init__()
        self.model_img = VisionTransformer(stride_size=args.stride_size)
        self.model_img.load_param('models/vit_base_patch16_384.pth')
        self.model_txt = transformers.BertModel.from_pretrained('bert-base-uncased')


    def forward(self, img, txt, mask):
        img_f4 = self.model_img(img)  
        txt_f4 = self.model_txt(txt, mask)  
        return img_f4, txt_f4[1]


class ViT_Bert_Overlap_SelfSup(nn.Module):
    def __init__(self, args):
        super(ViT_Bert_Overlap_SelfSup, self).__init__()
        self.model_img = VisionTransformer(stride_size=args.stride_size)
        self.model_img.load_param('models/vit_base_patch16_384.pth')
        self.model_txt = transformers.BertModel.from_pretrained('bert-base-uncased')


    def forward(self, inputs, flag=0):
        
        if flag==0:
            # Image Self supervised
            img1, img2 = inputs
            img_f1 = self.model_img(img1)  
            img_f2 = self.model_img(img2)  
            return img_f1, img_f2
            
        elif flag==1:
            # Text Self supervised
            txt1, mask1, txt2, mask2 = inputs
            txt_f1 = self.model_txt(txt1, mask1)  
            txt_f2 = self.model_txt(txt2, mask2)  
            return txt_f1[1], txt_f2[1]
        elif flag==2:
            # Image-Text 
            img, txt, mask = inputs
            img_f4 = self.model_img(img)  
            txt_f4 = self.model_txt(txt, mask)  
            return img_f4, txt_f4[1]

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
