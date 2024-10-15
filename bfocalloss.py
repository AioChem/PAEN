import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    """

    def __init__(self, alpha=0.75, gamma=2, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        """
        output  (n, c)
        target  (n, c)
        """
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        # target = target.unsqueeze(dim=1)
        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -self.alpha * pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -(1-self.alpha) * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)

        loss = pos_loss + neg_loss
        if self.reduction == 'mean':
            loss = loss.mean()

        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

if __name__ == '__main__':
    """
    对于输入来说，(5, 1)  (5, 2) (5, 3)都是被接受的
    每个列都会被单独计算损失
    """

    predictions = torch.tensor([[0.8, -0.3], [-0.2, 0.1], [0.5, -0.7], [0.8, -0.9], [-0.2, 0.1]])  # 5个样本  (5, 1)
    labels = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 0], [0, 1]])  # 对应的标签  # (5, 1)

    bloss = BinaryFocalLoss(alpha=0.75, reduction='none')
    loss = bloss(predictions, labels)
    print(loss)
    print('flossmean', loss.mean())

    from utils import outcome_valid_sigmoid
    miou, precision, recall, f1, acc, mcc, auroc = outcome_valid_sigmoid(predictions, labels, 0.5)
    print(miou)
    # criterion = nn.BCEWithLogitsLoss(reduction='none')
    # loss = criterion(predictions, labels.float())  # 注意：标签需要是浮点数
    #
    # print("BCELoss:", loss)
    # print('bcemean', loss.mean())
    # print(torch.log(torch.tensor(0.5)))
    # print(torch.log(torch.tensor(1-0.5)))
    #
    # print(0.75*(1-0.69)**2*0.3711)
    # print(0.25*(0.69)**2*1.1712)
    #
    # print(0.75 * (1 - 0.1) ** 2 * torch.log(torch.tensor(0.1)))
    # print(0.25 * (0.1) ** 2 * torch.log(torch.tensor(0.1)))




