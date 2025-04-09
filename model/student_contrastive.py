

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


class BiLSTMStudent(nn.Module):
    def __init__(self, hidden_dim, output_dim, base_model1, base_model2):
        super(BiLSTMStudent, self).__init__()
        self.hidden_size1 = base_model1.transformer_model.config.hidden_size
        # print(self.hidden_size1)
        self.hidden_size2 = base_model2.transformer_model.config.hidden_size
        # 分别用于处理code和message信息的BiLSTM
        self.bert = base_model1
        self.codebert = base_model2
        self.lstm_code = nn.LSTM(self.hidden_size1, hidden_dim, num_layers=1, bidirectional=True
, batch_first=True)
        self.lstm_message = nn.LSTM(self.hidden_size2, hidden_dim, num_layers=1, bidirectional=True
, batch_first=True)
        
        # 将BiLSTM编码结果拼接后通过全连接层进行分类
        self.fc = nn.Linear(hidden_dim * 4, output_dim)  # hidden_dim * 2 for each LSTM (bidirectional)

    def forward(self, message_input, code_input):
        # 对code信息编码
        # print(message_input)
        # bertmodel 
        del code_input['token_type_ids']
        # roberta model
        del message_input['token_type_ids']
        bert_output = self.bert(message_input)
        codebert_output = self.codebert(code_input)


        # print('================',bert_output.last_hidden_state)
        lstm_out_code, _ = self.lstm_code(codebert_output.last_hidden_state)
        # code_feat = lstm_out_code[:, -1, :]  # 最后一个hidden state
        # 对message信息编码
        lstm_out_message, _ = self.lstm_message(bert_output.last_hidden_state)
        # message_feat = lstm_out_message[:, -1, :]  # 最后一个hidden state

        lstm_code_mean = torch.mean(lstm_out_code, dim=1)  # (batch_size, 2*lstm_hidden_size)
        lstm_message_mean = torch.mean(lstm_out_message, dim=1)  # (batch_size, 2*lstm_hidden_size)
        
        # 拼接code和message的特征
        combined_feat = torch.cat((lstm_code_mean, lstm_message_mean), dim=1)
        out = self.fc(combined_feat)
        return out


    # 定义 MSE 蒸馏损失
    def mse_distillation_loss(self, teacher_logits, student_logits):
        """
        Compute MSE loss between teacher and student logits.
        
        Args:
            teacher_logits (torch.Tensor): Logits from teacher model. Shape: (batch_size, num_classes)
            student_logits (torch.Tensor): Logits from student model. Shape: (batch_size, num_classes)
        
        Returns:
            torch.Tensor: MSE loss value.
        """
        mse_loss = F.mse_loss(teacher_logits, student_logits, reduction='mean')
        return mse_loss

    def kl_distillation_loss(self, teacher_logits, student_logits):
        """
        Compute KL Divergence loss between teacher and student logits.
    
        Args:
            teacher_logits (torch.Tensor): Logits from teacher model. Shape: (batch_size, num_classes)
            student_logits (torch.Tensor): Logits from student model. Shape: (batch_size, num_classes)
    
        Returns:
            torch.Tensor: KL Divergence loss value.
        """
        # Apply softmax to convert logits to probabilities
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
    
        # Compute KL divergence
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        return kl_loss
    def contrastive_distillation_loss(self, teacher_logits, student_logits, margin=1.0):
        """
        Compute Contrastive loss between teacher and student logits.
    
        Args:
            teacher_logits (torch.Tensor): Logits from teacher model. Shape: (batch_size, num_classes)
            student_logits (torch.Tensor): Logits from student model. Shape: (batch_size, num_classes)
            margin (float): Margin for negative pairs. Default is 1.0.
    
        Returns:
            torch.Tensor: Contrastive loss value.
        """
        # Compute pairwise distances
        positive_pairs = (teacher_logits - student_logits).norm(p=2, dim=1)
        negative_pairs = margin - positive_pairs
    
        # Compute contrastive loss
        positive_loss = positive_pairs.pow(2).mean()
        negative_loss = F.relu(negative_pairs).pow(2).mean()
        contrastive_loss = positive_loss + negative_loss
        return contrastive_loss
    
    def distill_trainer(self, teacher_model, train_loader, num_epochs=3, learning_rate=2e-5, alpha=0.5, T=2.0):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        teacher_model.to(device)
        self.to(device)
        teacher_model.eval()  # Set teacher model to evaluation mode
        early_stopper = EarlyStopper(patience=3, min_delta=0)
        optimizer = AdamW(self.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
                for batch in train_loader:
                    inputs_bert, inputs_codebert, labels = batch
                    inputs_bert = {k: v.to(device) for k, v in inputs_bert.items()}
                    inputs_codebert = {k: v.to(device) for k, v in inputs_codebert.items()}
    
                    with torch.no_grad():
                        teacher_logits, _ = teacher_model(inputs_bert, inputs_codebert)  # Teacher's output
    
                    student_logits = self(inputs_bert,inputs_codebert)  # Adjust input as needed
    
                    loss = self.contrastive_distillation_loss(teacher_logits, student_logits)
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
    
                    total_loss += loss.item()
                    pbar.update(1)
                avg_loss = total_loss / len(train_loader)
                print('=============================train========================')
                pbar.set_description(f'Epoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}')

                if early_stopper.early_stop(avg_loss):
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
                # print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")
        val_acc, val_labels, val_probabilities, val_embeddings, val_predictions = self.evaluate(train_loader)
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
                student_logits = self(inputs_bert,inputs_codebert)  # Adjust input as needed
                # print("eval embedding",embeddings)
                _, predicted = torch.max(student_logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(torch.softmax(student_logits, dim=1).cpu().numpy()[:, 1]) # probability of the positive class

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
