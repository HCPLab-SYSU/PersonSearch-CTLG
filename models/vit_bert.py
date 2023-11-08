import timm
from pprint import pprint
import torch
from torchsummary import summary
import transformers
import torch.nn as nn
from collections import OrderedDict


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
class ViT_B_patch16(nn.Module):
    def __init__(self):
        super(ViT_B_patch16, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_384',
                                         pretrained=True,)
                                        #  checkpoint_path='./models/pretrained_checkpoint_path/vit_base_patch16_384.npz')
        self.pos_embed = nn.Parameter(torch.zeros(1, 193, 768)) # hard code here

        # NOTE position embedding ~
        for n,p in self.vit.named_parameters():
            # print(n, p.shape, p.numel())
            if 'pos' in n:
                # 需要取 0 , [9,17) * row (range from 0 -> 24) 的值
                new_p = p[:, 0:1, :]
                for i in range(24):
                    x = p[:, 9+i*24:17+i*24, :]
                    new_p = torch.cat((new_p, x), dim=1)
                self.pos_embed = nn.Parameter(new_p)
                break
        self.vit.pos_embed = self.pos_embed

    def forward(self, x):
        x = self.vit.forward_features(x)    # 得到的是第一个cls_token的 embedding 掌握了全局的信息。
        return x

class ViT_Bert(nn.Module):
    def __init__(self, args):
        super(ViT_Bert, self).__init__()
        self.model_img = ViT_B_patch16()
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
