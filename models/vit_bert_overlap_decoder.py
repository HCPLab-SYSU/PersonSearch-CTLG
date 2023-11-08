import timm
from pprint import pprint
import torch
from torchsummary import summary
import transformers
import torch.nn as nn
from collections import OrderedDict

from .vit_overlap import VisionTransformer, Block

class ViT_Bert_Overlap_Decoder(nn.Module):
    def __init__(self, args):
        super(ViT_Bert_Overlap_Decoder, self).__init__()
        self.model_img = VisionTransformer(stride_size=args.stride_size)
        self.model_img.load_param('models/vit_base_patch16_384.pth')
        # self.model_img.load_param('models/vit_base_patch16_384.pth')
        self.model_txt = transformers.BertModel.from_pretrained('bert-base-uncased')

        decoder_depth = 6
        dpr = [x.item() for x in torch.linspace(0, 0., decoder_depth)]  # stochastic depth decay rule
        self.decoder = self.blocks = nn.Sequential(*[
            Block(
                dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True, drop=0.,
                attn_drop=0., drop_path=dpr[i], norm_layer=nn.LayerNorm, act_layer=nn.GELU)
            for i in range(decoder_depth)]) # 2层decoder
        # self.norm = nn.LayerNorm(768)

    def forward(self, img, txt, mask):
        # img_f4 = self.model_img(img)  
        txt_f4 = self.model_txt(txt, mask) 
        all_feas = self.model_img.forward_global_local_features(img)
        # print(all_feas.shape, txt_f4[1].shape, txt_f4[1].reshape(all_feas.shape[0], 1, -1).shape)
        x = torch.cat(( txt_f4[1].reshape(all_feas.shape[0], 1, -1) ,all_feas), dim=1)     # 把文本的embedding concat进去，用于重建
        # print(x.shape)
        reconstruct_img = self.decoder(x)[:,2:].reshape(img.shape)

        return all_feas[:,0], reconstruct_img, txt_f4[1]

    # def forward(self, img, txt, mask):
    #     # img_f4 = self.model_img(img)  
    #     all_feas = self.model_img.forward_global_local_features(img)
    #     reconstruct_img = self.decoder(all_feas)[:,1:].reshape(img.shape)
    #     txt_f4 = self.model_txt(txt, mask)  
    #     return all_feas[:,0], reconstruct_img, txt_f4[1]

if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description='command for train on CUHK-PEDES')
        parser.add_argument('--embedding_type', type=str,
                        default='BERT')
        parser.add_argument('--stride_size', type=int,
                        default=16)
        args = parser.parse_args()
        return args
    args = parse_args()
    model = ViT_Bert_Overlap_Decoder(args)
    # print(model)
    total = sum(p.numel() for p in model.model_img.parameters())
    print("Total params: %.2fM" % (total/1e6))
    total = sum(p.numel() for p in model.model_txt.parameters())
    print("Total params: %.2fM" % (total/1e6))

    img_demo = torch.zeros((2, 3, 384, 128))
    txt_demo = torch.zeros((2, 64), dtype=int)
    img_f4, reconstruc_img,  txt_f4 = model(img_demo, txt_demo, txt_demo)
    print(img_f4.shape, reconstruc_img.shape, txt_f4.shape)
