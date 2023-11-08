# from turtle import forward
import timm
from pprint import pprint
import torch
from torchsummary import summary
import transformers
import torch.nn as nn
from collections import OrderedDict

from .vit_overlap import VisionTransformer


class VisionTransformer_MultiLayerOutput(VisionTransformer):
    def forward_multi_layer_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # global features
        # layer11 = self.blocks[:-1](x)
        layer8 = self.blocks[:-4](x)
        layer10 = self.blocks[-4:-2](layer8)
        layer12 = self.blocks[-2:](layer10)

        layer8_out = self.pre_logits(self.norm(layer8)[:, 0])
        layer10_out = self.pre_logits(self.norm(layer10)[:, 0])
        global_out = self.pre_logits(self.norm(layer12)[:, 0])
        return layer8_out, layer10_out, global_out  # 第一个输出的是class token 第二个输出的是所有patch token 
    def forward(self, x):
        o1,o2,o3 = self.forward_multi_layer_features(x)
        return o1,o2,o3

class ViT_Bert_MultiLayer(nn.Module):
    def __init__(self, args):
        super(ViT_Bert_MultiLayer, self).__init__()
        self.model_img = VisionTransformer_MultiLayerOutput(stride_size=args.stride_size)
        self.model_img.load_param('models/vit_base_patch16_384.pth')
        self.model_txt = transformers.BertModel.from_pretrained('bert-base-uncased')


    def forward(self, img, txt, mask):
        o1,o2,o3 = self.model_img(img)  
        txt_f4 = self.model_txt(txt, mask)  
        return o1,o2,o3, txt_f4[1]


if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description='command for train on CUHK-PEDES')
        parser.add_argument('--embedding_type', type=str,
                        default='BERT')
        args = parser.parse_args()
        return args
    args = parse_args()
    model = ViT_Bert_MultiLayer(args)
    # print(model)
    total = sum(p.numel() for p in model.model_img.parameters())
    print("Total params: %.2fM" % (total/1e6))
    total = sum(p.numel() for p in model.model_txt.parameters())
    print("Total params: %.2fM" % (total/1e6))

    img_demo = torch.zeros((2, 3, 384, 128))
    txt_demo = torch.zeros((2, 64), dtype=int)
    img_f4, txt_f4 = model(img_demo, txt_demo, txt_demo)
    print(img_f4.shape, txt_f4.shape)
