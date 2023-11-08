

import timm
from pprint import pprint
import torch
from torchsummary import summary
import transformers
import torch.nn as nn
from collections import OrderedDict
from torchvision import models



class Bert_Linear_txt(nn.Module):
    def __init__(self, args):
        super(Bert_Linear_txt, self).__init__()
        if args.embedding_type == 'BERT':
            model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer, 'bert-base-uncased')
        elif args.embedding_type == 'Roberta':
            model_class, tokenizer_class, pretrained_weights = (transformers.RobertaModel, transformers.RobertaTokenizer, 'roberta-base')

        self.text_embed = model_class.from_pretrained(pretrained_weights)
        self.BERT = True
    
    def forward(self, txt, mask):
        txt = self.text_embed(txt, attention_mask=mask)
        output = txt[1]
        return output

# ================================== txt↑  image↓ ===============================================
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        resnet50 = models.resnet50(pretrained=True)
        resnet50.layer4[0].downsample[0].stride = (1, 1)
        resnet50.layer4[0].conv2.stride = (1, 1)
        self.img_feature = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,  # 256 64 32
            resnet50.layer2,  # 512 32 16
            resnet50.layer3,  # 1024 16 8
            resnet50.layer4  # 2048 16 8
        )
        self.max_pool2d = nn.AdaptiveMaxPool2d((1, 1))    # force ouput dim x 1 x 1 feature
        self.fc = nn.Linear(2048, 768)

    def forward(self, x):
        x = self.img_feature(x)
        # print(x.shape)
        x = self.max_pool2d(x).squeeze(dim=-1).squeeze(dim=-1)
        # print(x.shape)
        x = self.fc(x)
        return x

class Res_Bert(nn.Module):
    def __init__(self, args):
        super(Res_Bert, self).__init__()
        self.model_img = ResNet50()
        self.model_txt = Bert_Linear_txt(args)

    def forward(self, img, txt, mask):
        img_f4 = self.model_img(img)  
        txt_f4= self.model_txt(txt, mask)  
        return img_f4, txt_f4


if __name__ == '__main__':

    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description='command for train on CUHK-PEDES')
        parser.add_argument('--embedding_type', type=str,
                        default='BERT')
        args = parser.parse_args()
        return args
    args = parse_args()
    model = Res_Bert(args)
    print(model)

    total = sum(p.numel() for p in model.model_img.parameters())
    print("Total params: %.2fM" % (total/1e6))
    total = sum(p.numel() for p in model.model_txt.parameters())
    print("Total params: %.2fM" % (total/1e6))

    img_demo = torch.zeros((2, 3, 384, 128))
    txt_demo = torch.zeros((2, 64), dtype=int)
    img_f4, txt_f4 = model(img_demo, txt_demo, txt_demo)
    print(img_f4.shape, txt_f4.shape)
