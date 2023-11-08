import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class CMPMLoss(nn.Module):
    def __init__(self, args):
        super(CMPMLoss, self).__init__()
        self.epsilon = args.epsilon

    def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
        batch_size = image_embeddings.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()   # distance=0 表示匹配。
        labels_mask = (labels_dist == 0)
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)    # ground truth matrix

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())

        # normalize the true matching distribution
        i2t_pred = F.softmax(image_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon))
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + self.epsilon))
        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
        return cmpm_loss
    def forward(self, imgs, txts, labels):
        loss = 0.0
        if isinstance(imgs, tuple):
            # 多个分支的loss计算
            assert len(imgs) == len(txts)
            for img_branch, txt_branch in zip(imgs, txts):
                loss += self.compute_cmpm_loss(img_branch, txt_branch, labels)
        else:
            loss = self.compute_cmpm_loss(imgs, txts, labels)
        return loss


class InfoNCELoss(nn.Module):
    def __init__(self, temperature):
        super(InfoNCELoss, self).__init__()

        self.temperature = temperature

    def compute_info_nce_loss(self, image_embeddings, text_embeddings, labels):

        image_features = F.normalize(image_embeddings, dim=-1)
        text_features = F.normalize(text_embeddings, dim=-1)

        logits_per_image =  image_features @ text_features.t()
        logits_per_text =  text_features @ image_features.t()

        ground_truth = torch.arange(len(logits_per_image), device=image_features.device)
        total_loss = (F.cross_entropy(logits_per_image/self.temperature, ground_truth) +
                         F.cross_entropy(logits_per_text/self.temperature, ground_truth))/2
        
        return total_loss


    def forward(self, image_embeddings, text_embeddings, labels):
        loss = 0.0
        loss = self.compute_info_nce_loss(image_embeddings, text_embeddings, labels)
        return loss


# class InfoNCELoss_v2(nn.Module):
#     def __init__(self, args):
#         super(InfoNCELoss_v2, self).__init__()

#         self.temperature = args.temperature

#     def compute_info_nce_loss(self, image_embeddings, text_embeddings, labels):
#         # 扔掉重复id的样本
#         batch_size = image_embeddings.shape[0]
#         labels_reshape = torch.reshape(labels, (batch_size, 1))
#         labels_dist = labels_reshape - labels_reshape.t()   # distance=0 表示匹配。
#         labels_mask = (labels_dist == 0).float()

#         image_features = F.normalize(image_embeddings, dim=-1)
#         text_features = F.normalize(text_embeddings, dim=-1)

#         logits_per_image =  image_features @ text_features.t()
#         logits_per_text =  text_features @ image_features.t()

#         ground_truth = torch.arange(len(logits_per_image), device=image_features.device)
#         total_loss = (F.cross_entropy(logits_per_image/self.temperature, ground_truth) +
#                          F.cross_entropy(logits_per_text/self.temperature, ground_truth))/2
        
#         return total_loss


#     def forward(self, image_embeddings, text_embeddings, labels):
#         loss = 0.0
#         loss = self.compute_info_nce_loss(image_embeddings, text_embeddings, labels)
#         return loss


# class ModifiedInfoNCELoss(nn.Module):
#     def __init__(self, args):
#         super(ModifiedInfoNCELoss, self).__init__()
#         self.temperature = args.temperature

#     def compute_modified_info_nce_loss(self, image_embeddings, text_embeddings, labels):

#         image_features = F.normalize(image_embeddings, dim=-1)
#         text_features = F.normalize(text_embeddings, dim=-1)

#         logits_per_image =  image_features @ text_features.t()
#         logits_per_text =  text_features @ image_features.t()

#         batch_size = image_embeddings.shape[0]
#         labels_reshape = torch.reshape(labels, (batch_size, 1))
#         labels_dist = labels_reshape - labels_reshape.t()   # distance=0 表示匹配。
#         labels_mask = (labels_dist == 0).float().to(device=image_features.device)
#         total_loss = (F.binary_cross_entropy( F.sigmoid(logits_per_image/self.temperature),  labels_mask) +
#                          F.binary_cross_entropy( F.sigmoid(logits_per_text/self.temperature),  labels_mask))/2
#         return total_loss

#     def forward(self, image_embeddings, text_embeddings, labels):
#         loss = 0.0
#         loss = self.compute_modified_info_nce_loss(image_embeddings, text_embeddings, labels)
#         return loss
