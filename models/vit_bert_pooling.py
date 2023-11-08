from pyexpat import model
import timm
from pprint import pprint
import torch
from torchsummary import summary
import transformers
import torch.nn as nn
from collections import OrderedDict
from itertools import repeat
from .vit_pooling import *

class Bert_Linear_txt(nn.Module):
    def __init__(self, args):
        super(Bert_Linear_txt, self).__init__()
        if args.embedding_type == 'BERT':
            model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer, 'bert-base-uncased')
        self.text_embed = model_class.from_pretrained(pretrained_weights)

    def forward(self, txt, mask):
        # print(txt, mask)
        txt = self.text_embed(txt, attention_mask=mask)
        output = txt[1]
        # print(txt[0].shape, txt[1].shape)
        return output

# ================================== txt↑  image↓ ==================================

class ViT_B_patch16(nn.Module):
    def __init__(self):
        super(ViT_B_patch16, self).__init__()
        self.vit_pooling = VisionTransformerPooling()
        # NOTE load from pretrain
        self.vit_pooling.load_state_dict(torch.load('models/vit_base_patch16_384.pth')) 
        self.pos_embed = nn.Parameter(torch.zeros(1, 193, 768)) # 192+1 (1为class token) 输入分辨率为384*128，,需要修正。
        for n,p in self.vit_pooling.named_parameters():
            if 'pos' in n:
                new_p = p[:, 0:1, :]
                for i in range(24):
                    x = p[:, 9+i*24:17+i*24, :]
                    new_p = torch.cat((new_p, x), dim=1)
                self.pos_embed = nn.Parameter(new_p)
                break
        self.vit_pooling.pos_embed = self.pos_embed
    def forward(self, x):
        out_l8, out_l10, out_l12 = self.vit_pooling.forward_features(x)    # 得到的是第一个cls_token的 embedding 掌握了全局的信息。
        # print(out_l8.shape, out_l10.shape, out_l12.shape)
        return out_l8, out_l10, out_l12


# ====================== model definition ==========================
class ViT_Bert_Pooling(nn.Module):
    def __init__(self, args):
        super(ViT_Bert_Pooling, self).__init__()
        self.model_img = ViT_B_patch16()
        self.model_txt = Bert_Linear_txt(args)

    def forward(self, img, txt, mask):
        out_l8, out_l10, out_l12 = self.model_img(img)  
        txt_f4 = self.model_txt(txt, mask)  
        return out_l8, out_l10, out_l12, txt_f4


if __name__ == '__main__':

    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description='command for train on CUHK-PEDES')
        parser.add_argument('--embedding_type', type=str,
                        default='BERT')
        args = parser.parse_args()
        return args
    args = parse_args()
    model = ViT_Bert_Pooling(args)
    # print(model)

    # total = sum(p.numel() for p in model.model_img.parameters())
    # print("Total params: %.2fM" % (total/1e6))
    # total = sum(p.numel() for p in model.model_txt.parameters())
    # print("Total params: %.2fM" % (total/1e6))

    img_demo = torch.zeros((2, 3, 384, 128))
    txt_demo = torch.ones((2, 64), dtype=int)
    txt_mask = torch.zeros((2, 64), dtype=int)
    out_l8, out_l10, out_l12, txt_f4 = model(img_demo, txt_demo, txt_mask)
    # print(img_f4)

    # SAVING ORIGINAL MODEL As .pth
    # model_img = timm.create_model('vit_base_patch16_384',pretrained=True,)
    # torch.save(model_img.state_dict(), 'vit_base_patch16_384.pth')