import torch
import torch.nn as nn
from transformers import AdamW, BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)
# Define base model class
class BaseModel(nn.Module):
    def __init__(self, transformer_model):
        super(BaseModel, self).__init__()
        self.transformer_model = transformer_model
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs):
        outputs = self.transformer_model(**inputs).last_hidden_state[:, 0, :]
        outputs = self.dropout(outputs)
        return outputs

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class LinearGateNetwork(nn.Module):
    def __init__(self, input_dim):
        super(LinearGateNetwork, self).__init__()
        # 两个线性层，分别计算 A 和 B 的权重
        self.projection = nn.Linear(input_dim, input_dim//2)  # 用于计算 h

    def forward(self, embedding_A, embedding_B):
        # 拼接 A 和 B
        combined_features = torch.cat((embedding_A, embedding_B), dim=1)  # [batch_size, input_dim]

        # 计算 A 和 B 的权重
        projected_h = self.projection(combined_features)  # [batch_size, 1]
        # 使用 softmax 归一化权重
        weights = F.softmax(projected_h, dim=1)  # [batch_size, 2]

        # 分别获取 A 和 B 的权重
        weight_A, weight_B = weights[:, 0].unsqueeze(1), weights[:, 1].unsqueeze(1)  # [batch_size, 1]
        return weight_A, weight_B



class MoEModel(nn.Module):
    def __init__(self, base_model1, base_model2):
        super(MoEModel, self).__init__()
        self.base_model1 = base_model1
        self.base_model2 = base_model2
        self.hidden_size1 = base_model1.transformer_model.config.hidden_size
        self.hidden_size2 = base_model2.transformer_model.config.hidden_size
        combined_input_dim = self.hidden_size1 + self.hidden_size2
        # 使用改进的 LinearGateNetwork
        self.gate_network = LinearGateNetwork(combined_input_dim)
        self.classifier = nn.Linear(combined_input_dim, 2)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, inputs_bert, inputs_codebert):
        # 获取 BERT 和 CodeBERT 的输出
        outputs_bert = self.base_model1(inputs_bert)  # [batch_size, hidden_size1]
        outputs_codebert = self.base_model2(inputs_codebert)  # [batch_size, hidden_size2]

        # 计算 A 和 B 的权重
        weight_A, weight_B = self.gate_network(outputs_bert, outputs_codebert)  # [batch_size, 1] x 2
        # print("A",weight_A)
        # print("B",weight_B)
        # 对 A 和 B 进行加权
        weighted_outputs_bert = outputs_bert * weight_A  # [batch_size, hidden_size1]
        weighted_outputs_codebert = outputs_codebert * weight_B  # [batch_size, hidden_size2]

        # 拼接加权后的 A 和 B
        combined = torch.cat((weighted_outputs_bert, weighted_outputs_codebert), dim=1)  # [batch_size, combined_input_dim]

        # 分类
        logits = self.classifier(combined)

        return logits, combined


    # 定义Focal Loss
    def focal_loss(self, logits, targets, gamma=2):
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        focal_weight = (1 - probs) ** gamma
        focal_loss = F.cross_entropy(logits, targets) * focal_weight
        return focal_loss.mean()
    
    # 定义Smooth Loss
    def smooth_loss(self, logits, targets, smoothing=0.1):
        num_classes = logits.size(-1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        targets_smoothed = targets_one_hot * (1 - smoothing) + smoothing / num_classes
        log_probs = F.log_softmax(logits, dim=-1)
        smooth_loss = -torch.sum(targets_smoothed * log_probs, dim=-1)
        return smooth_loss.mean()
    
    # 定义组合累计学习损失函数
    def cumulative_loss_fn(self, logits, targets, epoch, total_epochs, gamma=2, smoothing=0.1):
        # 计算权重
        imbalance_weight = (epoch / total_epochs)
        
        # Focal Loss 部分
        focal_component = self.focal_loss(logits, targets, gamma)
        focal_weight = (1 - imbalance_weight) ** 2
        
        # Smooth Loss 部分
        smooth_component = self.smooth_loss(logits, targets, smoothing)
        smooth_weight = 1 - focal_weight
        
        # 组合两个损失部分
        cumulative_loss = focal_weight * focal_component + smooth_weight * smooth_component
        return cumulative_loss


    
    def trainer(self, train_loader, val_loader, num_epochs=3, learning_rate=2e-5, patience=3):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        optimizer = AdamW(self.parameters(), lr=learning_rate)
        # criterion = self.cumulative_loss_fn()
        early_stopper = EarlyStopper(patience=3, min_delta=0)
        best_acc = 0.0
        patience_counter = 0

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
                for batch in train_loader:
                    inputs_bert, inputs_codebert, labels = batch
                    inputs_bert = {k: v.to(device) for k, v in inputs_bert.items()}
                    inputs_codebert = {k: v.to(device) for k, v in inputs_codebert.items()}
                    labels = labels.to(device)
    
                    optimizer.zero_grad()
                    logits, _ = self(inputs_bert, inputs_codebert)
                    # loss = criterion(logits, labels)
                    loss = self.cumulative_loss_fn(logits, labels, epoch + 1, num_epochs)
                    loss.mean().backward()
                    optimizer.step()
    
                    total_loss += loss.item()
                    # update bar
                    pbar.update(1)
                avg_loss = total_loss / len(train_loader)
                print('=============================train========================')
                pbar.set_description(f'Epoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}')

                if early_stopper.early_stop(avg_loss):
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        val_acc, val_labels, val_probabilities, val_embeddings, val_predictions = self.evaluate(val_loader)
    def evaluate(self, val_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        all_probabilities = []
        all_embeddings = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs_bert, inputs_codebert, labels = batch
                inputs_bert = {k: v.to(device) for k, v in inputs_bert.items()}
                inputs_codebert = {k: v.to(device) for k, v in inputs_codebert.items()}
                labels = labels.to(device)

                # Forward pass through the model to get logits and attention weights
                logits, embeddings = self(inputs_bert, inputs_codebert)
                # print("eval embedding",embeddings)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy()[:, 1]) # probability of the positive class
                all_embeddings.extend(embeddings.cpu().numpy())

        # Compute metrics
        accuracy = correct / total
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        # Print metrics
        print(f'Validation Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')

        # Return results
        return accuracy, all_labels, all_probabilities, all_embeddings, all_predictions


    def plot_pr_curve(self, labels, probabilities):
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        plt.plot(recall, precision, label='PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_tsne(self, embeddings, labels):
        # working code 
        # tsne = TSNE(n_components=2, random_state=0)
        # embeddings_np = np.vstack(embeddings, dtype=np.float32)
        
        # print(embeddings_np.shape)
        
        # embeddings_2d = tsne.fit_transform(embeddings_np)
        
        # plt.figure(figsize=(10, 8))
        # scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.5)
        # plt.colorbar(scatter, label='Labels')
        # plt.title('t-SNE Visualization of Embeddings')
        # plt.show()
    
        
        tsne = TSNE(n_components=2, random_state=42)
    
        print("embedding来啦",embeddings[0].shape)
        # embeddings_cpu = combined.cpu().detach()
        # embeddings_np = torch.stack(embeddings_cpu).numpy()  # 形状：(600, 8, 768)
        
        # 维度变换
        embeddings_np = np.vstack(embeddings) 
        embeddings_2d = tsne.fit_transform(embeddings_np)
        df_tsne = pd.DataFrame(embeddings_2d, columns=['TSNE1', 'TSNE2'])
        df_tsne['Class Name'] = labels # Add labels column from df_train to df_tsne
        df_tsne
        
        fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
        sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})
        sns.scatterplot(data=df_tsne, x='TSNE1', y='TSNE2', hue='Class Name', palette='hls')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.title('Scatter plot of news using t-SNE');
        plt.xlabel('TSNE1');
        plt.ylabel('TSNE2');
        plt.axis('equal')
    
        # Apply KMeans
        kmeans_model = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(embeddings_np)
        labels = kmeans_model.fit_predict(embeddings_np)
        
        df_tsne['Cluster'] = labels
        df_tsne
        fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
        sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})
        sns.scatterplot(data=df_tsne, x='TSNE1', y='TSNE2', hue='Cluster', palette='magma')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.title('Scatter plot of news using KMeans Clustering');
        plt.xlabel('TSNE1')
        plt.ylabel('TSNE2')
        plt.axis('equal')

    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix on Dataset I', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
    
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
    
        plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        plt.tight_layout()



