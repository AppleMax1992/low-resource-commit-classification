import torch
import torch.nn.functional as F

class CumulativeFocalLoss:
    def __init__(self, gamma_start=1.0, gamma_end=3.0, alpha_start=0.25, alpha_end=0.5, total_epochs=50):
        """
        初始化动态调整的参数
        :param gamma_start: 初始 gamma 值（控制难样本权重）
        :param gamma_end: 最终 gamma 值
        :param alpha_start: 初始 alpha 值（控制类别权重）
        :param alpha_end: 最终 alpha 值
        :param total_epochs: 总训练轮次
        """
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.total_epochs = total_epochs

    def compute_gamma_alpha(self, epoch):
        """
        根据当前 epoch 计算动态 gamma 和 alpha
        :param epoch: 当前训练轮次
        :return: 动态调整后的 gamma 和 alpha
        """
        gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * (epoch / self.total_epochs)
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * (epoch / self.total_epochs)
        # print('当前epoch的',gamma, alpha)
        return gamma, alpha

    def focal_loss(self, logits, targets, epoch):
        """
        计算具有累计学习系数的 Focal Loss
        :param logits: 模型输出的 logits
        :param targets: 真实标签
        :param epoch: 当前训练轮次
        :return: 计算后的 Focal Loss
        """
        # 动态调整 gamma 和 alpha
        gamma, alpha = self.compute_gamma_alpha(epoch)

        # 计算预测概率
        probs = F.softmax(logits, dim=-1)

        # 将 targets 转为 one-hot 表示
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()

        # 获取预测正确类别的概率
        pt = (probs * targets_one_hot).sum(dim=-1)

        # 动态类别权重（根据 targets 调整 alpha）
        alpha_t = alpha * targets_one_hot + (1 - alpha) * (1 - targets_one_hot)
        alpha_t = alpha_t.sum(dim=-1)

        # 计算 Focal Loss 权重
        focal_weight = alpha_t * (1 - pt) ** gamma
        
        # 计算 Focal Loss，使用 reduction='none' 避免过早聚合
        focal_loss = focal_weight * F.cross_entropy(logits, targets, reduction='none')
        return focal_loss.mean()