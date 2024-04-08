import torch.nn as nn

class NBVLoss(nn.Module):
    def  __init__(self, lambda_for_1, device):
        super(NBVLoss, self).__init__()

        # self.mse = nn.MSELoss()
        self.entropy = nn.BCELoss()

        self.lambda_for_1 = lambda_for_1
        self.device = device

    def forward(self, predictions, target):
        # mask_0 = (target == 0)
        # mask_1 = ~mask_0
        # loss_where_0 = (self.entropy(predictions[mask_0], target[mask_0])).to(self.device)
        # loss_where_1 = (self.entropy(predictions[mask_1], target[mask_1])).to(self.device)

        loss_where_1 = 0
        loss_where_0 = 0
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if target[i][j] == 0:
                    loss_where_0 += self.entropy(predictions[i][j], target[i][j]).to(self.device)
                else:
                    loss_where_1 += self.entropy(predictions[i][j], target[i][j]).to(self.device)

        return self.lambda_for_1 * loss_where_1 + loss_where_0
