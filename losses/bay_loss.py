from torch.nn.modules import Module
import torch

class Bay_Loss(Module):
    def __init__(self, use_background, device):
        super(Bay_Loss, self).__init__()
        self.device = device
        self.use_bg = use_background

    def forward(self, prob_list, target_list, pre_density):
        loss = 0
        for idx, prob in enumerate(prob_list):  # iterative through each sample
            # print(idx)
            if prob is None:  # image contains no annotation points
                # print(pre_density[idx])
                pre_count = torch.sum(pre_density[idx][idx])
                target = torch.zeros((1,), dtype=torch.float32, device=self.device)
            else:
                N = len(prob)
                if self.use_bg:
                    target = torch.zeros((N,), dtype=torch.float32, device=self.device)
                    target[:-1] = target_list[idx]
                else:
                    target = target_list[idx]
                # print(len(prob_list))
                # print(target.shape)
                # print(pre_density[idx][idx].shape, prob.shape, target.shape)
                # print(idx, N)
                pre_count = torch.sum(pre_density[idx][idx].view((1, -1)) * prob, dim=1)  # flatten into vector

            loss += torch.sum(torch.abs(target - pre_count))
        loss = loss / len(prob_list)
        return loss



