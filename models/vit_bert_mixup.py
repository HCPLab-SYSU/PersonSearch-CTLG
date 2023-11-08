from matplotlib.pyplot import flag
import timm
from pprint import pprint
import torch
from torchsummary import summary
import transformers
import torch.nn as nn
from collections import OrderedDict

# from vit_overlap import VisionTransformer
from .vit_overlap import VisionTransformer



class ViT_Bert_Overlap_Mixup(nn.Module):
    def __init__(self, args):
        super(ViT_Bert_Overlap_Mixup, self).__init__()
        self.model_img = VisionTransformer(stride_size=args.stride_size)
        # self.model_img = VisionTransformer(stride_size=16)
        self.model_img.load_param('models/vit_base_patch16_384.pth')
        self.model_txt = transformers.BertModel.from_pretrained('bert-base-uncased')

    def forward_mixup_part1(self, img, txt, mask):
        img_layer1_output = self.model_img.forward_mixup_part1(img)  
        # print(img_layer1_output.shape)
        txt_layer1_output= self.model_txt.forward_mixup_part1(txt, mask)  
        # print(txt_layer1_output.shape)
        return img_layer1_output, txt_layer1_output
    
    def forward_mixup_part2(self, img_layer1_output, txt_layer1_output, txt, mask):
        img_f4 =  self.model_img.forward_mixup_part2(img_layer1_output)  
        txt_f4 = self.model_txt.forward_mixup_part2(txt_layer1_output, txt, mask)  

        return img_f4, txt_f4

    def forward(self, img, txt, mask, img_layer1_output=None, txt_layer1_output=None, forward_flag=1):
        if forward_flag == 1:
            img_layer1_output, txt_layer1_output= self.forward_mixup_part1(img, txt, mask)
            return img_layer1_output, txt_layer1_output
        elif forward_flag == 2:
            img_f4, txt_f4 = self.forward_mixup_part2(img_layer1_output, txt_layer1_output, txt, mask)
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
    model = ViT_Bert_Overlap_Mixup(args)
    # print(model)
    total = sum(p.numel() for p in model.model_img.parameters())
    print("Total params: %.2fM" % (total/1e6))
    total = sum(p.numel() for p in model.model_txt.parameters())
    print("Total params: %.2fM" % (total/1e6))

    img_demo = torch.zeros((2, 3, 384, 128))
    txt_demo = torch.zeros((2, 64), dtype=int)
    img_f4, txt_f4 = model(img_demo, txt_demo, txt_demo)
    print(img_f4.shape, txt_f4.shape)
