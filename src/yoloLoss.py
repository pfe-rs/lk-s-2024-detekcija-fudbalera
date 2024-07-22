import torch
from src.utils import *

class YOLOv8Loss(nn.Module):
    def __init__(self, feature_size=7, num_bboxes=2, num_classes=4, lambda_box=1.0, lambda_cls=1.0, lambda_df=1.0, phi=0.0005):
        super(YOLOv8Loss, self).__init__()
        self.S = feature_size #7
        self.B = num_bboxes #2
        self.C = num_classes #4
        self.lambda_box = lambda_box #tezinski koeficijent za BB
        self.lambda_cls = lambda_cls
        self.lambda_df = lambda_df
        self.phi = phi

    def compute_iou(self, bbox1, bbox2):
      #intersection over union
      #lt je gornji levi ugao, rb donji desni
        lt = torch.max(bbox1[:, :2].unsqueeze(1).expand_as(bbox2[:, :2].unsqueeze(0)), bbox2[:, :2].unsqueeze(0).expand_as(bbox1[:, :2].unsqueeze(1)))
        rb = torch.min(bbox1[:, 2:].unsqueeze(1).expand_as(bbox2[:, 2:].unsqueeze(0)), bbox2[:, 2:].unsqueeze(0).expand_as(bbox1[:, 2:].unsqueeze(1)))
        #sirina i visina preseka
        wh = rb - lt
        wh[wh < 0] = 0
        inter = wh[:, :, 0] * wh[:, :, 1] #presek
        #unija
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
        union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter

        iou = inter / union
        return iou

    def forward(self, pred_tensor, target_tensor):
        batch_size = pred_tensor.size(0) #8
        S, B, C = self.S, self.B, self.C
        N = 5 * B + C #3. cifra u YOLO izlazu

        coord_mask = target_tensor[:, :, :, 4] > 0 #maska preko objekata
        noobj_mask = target_tensor[:, :, :, 4] == 0 #maska gde nema objekata
        #Dodaje novu dimenziju za proširenje maski na sve kanale
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor)

        #prediktovani targetsi
        coord_pred = pred_tensor[coord_mask].view(-1, N)
        bbox_pred = coord_pred[:, :5*B].contiguous().view(-1, 5)
        class_pred = coord_pred[:, 5*B:]
        #ciljni targetsi
        coord_target = target_tensor[coord_mask].view(-1, N)
        bbox_target = coord_target[:, :5*B].contiguous().view(-1, 5)
        class_target = coord_target[:, 5*B:]

        
        noobj_pred = pred_tensor[noobj_mask].view(-1, N) #predvidjanje za celije koje nemaju objekte
        noobj_target = target_tensor[noobj_mask].view(-1, N) #ciljne vrednosti za -II-
        noobj_conf_mask = torch.zeros(noobj_pred.size()).byte().cuda()#maska za confidence predvidjen
        
        for b in range(B):
            noobj_conf_mask[:, 4 + b * 5] = 1
        noobj_pred_conf = noobj_pred[noobj_conf_mask]
        noobj_target_conf = noobj_target[noobj_conf_mask]
        loss_noobj = F.binary_cross_entropy(noobj_pred_conf, noobj_target_conf, reduction='sum')#Gubitak za ćelije koje ne sadrže objekte, koristi binarnu cross-entrop

        # BBox loss
        bbox_target_iou = torch.zeros(bbox_target.size()).cuda() #IoU za ciljne
        coord_response_mask = torch.zeros(bbox_target.size()).byte().cuda() #maskira ćelije s okvirima

        for i in range(0, bbox_target.size(0), B): #prolazi kroz sve batchove
            pred = bbox_pred[i:i+B]#predviđanja za okvire u trenutnom segmentu
            pred_xyxy = torch.zeros(pred.size()).cuda() ##tensor sa istim dimenzijama kao pred, ali sa nultim vrednostima, koordinate bb pretvorene u x1,y1,x2,y2
            pred_xyxy[:, :2] = pred[:, :2] / float(S) - 0.5 * pred[:, 2:4]
            #Pretvara predviđene koordinate okvira iz format (center_x, center_y, width, height) u format (x1, y1, x2, y2). Prvi deo (pred[:, :2] / float(S)) daje koordinate
            # centra okvira normalizovane na raspon [0, 1], dok - 0.5 * pred[:, 2:4] oduzima polovinu širine i visine da bi se dobio gornji levi ugao.
            pred_xyxy[:, 2:4] = pred[:, :2] / float(S) + 0.5 * pred[:, 2:4]

            target = bbox_target[i].view(-1, 5) #isto to za ciljne vrednosti
            target_xyxy = torch.zeros(target.size()).cuda()
            target_xyxy[:, :2] = target[:, :2] / float(S) - 0.5 * target[:, 2:4]
            target_xyxy[:, 2:4] = target[:, :2] / float(S) + 0.5 * target[:, 2:4]

            iou = self.compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4])
            max_iou, max_index = iou.max(0)
            #masku odgovora (koordinata) na 1 za okvir sa najvećim IoU. Ovo označava da je ovaj okvir odgovoran za predviđanje objekta u toj ćeliji mreže.
            coord_response_mask[i + max_index] = 1
            #okvir sa najvećim IoU
            bbox_target_iou[i + max_index, 4] = max_iou

        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)
        bbox_target_response = bbox_target[coord_response_mask].view(-1, 5)
        target_iou = bbox_target_iou[coord_response_mask].view(-1, 5)
        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]), reduction='sum')
        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')

        # cross entropy loss
        loss_cls = F.binary_cross_entropy(class_pred, class_target, reduction='sum')

        # focal loss
        loss_df = torch.zeros_like(loss_cls).cuda()
        for i in range(batch_size):
            for j in range(S):
                for k in range(S):
                    qxy = target_tensor[i, j, k, 4] # ground truth iou
                    q_pred = pred_tensor[i, j, k, 4] # predvidjen iou
                    alpha_xy = (1 - qxy) / (1 - q_pred)
                    delta_xy = 4 / (3.14 ** 2) * ((torch.atan2(bbox_target[i, j, k, 2], bbox_target[i, j, k, 3]) - torch.atan2(bbox_pred[i, j, k, 2], bbox_pred[i, j, k, 3])) ** 2)
                    loss_df += alpha_xy * (q_pred - qxy) * torch.log(q_pred) + (qxy - q_pred) * torch.log(1 - q_pred)
        
        # Total loss
        total_loss = self.lambda_box * (loss_xy + loss_wh + loss_obj) + self.lambda_cls * loss_cls + self.lambda_df * loss_df
        total_loss += self.phi * sum(p.pow(2.0).sum() for p in self.parameters()) # L2 regularization

        return total_loss
