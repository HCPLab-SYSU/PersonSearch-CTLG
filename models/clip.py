import timm
from pprint import pprint
import torch
from torchsummary import summary
import transformers
import torch.nn as nn
from collections import OrderedDict
# import open_clip

# class CLIP(nn.Module):
#     def __init__(self, args):
#         super(CLIP, self).__init__()
#         # NOTE 显存占用会过高
#         # self.model, self.preprocess = open_clip.load("ViT-B/32")
#         self.model, _, preprocess = open_clip.create_model_and_transforms("ViT-B/32")
#     def forward(self, img, txt, mask):
#         img_f4 = self.model.encode_image(img)
#         txt_f4= self.model.encode_text(txt)
#         # print(img_f4, txt_f4)
#         return img_f4, txt_f4


if __name__ == '__main__':

    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description='command for train on CUHK-PEDES')
        parser.add_argument('--embedding_type', type=str,
                        default='BERT')
        args = parser.parse_args()
        return args
    args = parse_args()
    model = ViT_Bert(args)
    print(model)

    total = sum(p.numel() for p in model.model_img.parameters())
    print("Total params: %.2fM" % (total/1e6))
    total = sum(p.numel() for p in model.model_txt.parameters())
    print("Total params: %.2fM" % (total/1e6))

    img_demo = torch.zeros((2, 3, 384, 128))
    txt_demo = torch.zeros((2, 64), dtype=int)
    img_f4, txt_f4 = model(img_demo, txt_demo, txt_demo)
    print(img_f4.shape, txt_f4.shape)
