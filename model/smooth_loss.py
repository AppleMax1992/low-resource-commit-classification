import torch
import torch.nn.functional as F

class CumulativeSmoothLoss:
    def __init__(self, smoothing_start=0.2, smoothing_end=0.0, total_epochs=50):
        """
        初始化动态调整的标签平滑参数。
        :param smoothing_start: 初始的平滑系数（较大，表示强平滑）。
        :param smoothing_end: 最终的平滑系数（较小，表示弱平滑）。
        :param total_epochs: 总训练轮次。
        """
        self.smoothing_start = smoothing_start
        self.smoothing_end = smoothing_end
        self.total_epochs = total_epochs

    def compute_smoothing(self, epoch):
        """
        根据当前 epoch 计算动态标签平滑系数。
        :param epoch: 当前训练轮次。
        :return: 动态调整后的平滑系数。
        """
        return self.smoothing_start + (self.smoothing_end - self.smoothing_start) * (epoch / self.total_epochs)

    def smooth_loss(self, logits, targets, epoch):
        """
        计算具有累计学习调整的 Smooth Loss。
        :param logits: 模型输出的 logits。
        :param targets: 真实标签（整数形式）。
        :param epoch: 当前训练轮次。
        :return: 计算后的 Smooth Loss。
        """
        # 动态计算平滑系数
        smoothing = self.compute_smoothing(epoch)

        # 将 targets 转为 one-hot 表示
        num_classes = logits.size(-1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # 应用标签平滑
        smooth_targets = (1 - smoothing) * targets_one_hot + smoothing / num_classes

        # 计算交叉熵损失
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        return loss.mean()