# model.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class LSTM_Network(nn.Module):
    """Simple LSTM Network that predicts for every timeframe."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # batch_first=True means input/output tensors are (batch, seq, feature)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_size)
        #output = self.fc(lstm_out[:, -1, :])  # Use only the last time st
        # Apply the fully connected layer to each time step
        output = self.fc(lstm_out)    # output shape: (batch_size, seq_len, output_size)
        return output

class AdditiveAttention(nn.Module):
    """Additive Attention Mechanism (Bahdanau)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, query, key):
        energy = torch.tanh(self.linear1(query + key))
        attention = self.linear2(energy).squeeze(-1)  # (batch_size, seq_len)
        return attention

class DotProductAttention(nn.Module):
    """Dot-Product Attention Mechanism."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)  # Add Wv layer

    def forward(self, query, key, value):
        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)

        attention_weights = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(Q.size(-1))
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.bmm(attention_weights, V)
        return context_vector

class LSTM_Network_with_Attention(nn.Module):
    """LSTM Network with Attention Mechanism."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2, attention_type='additive'):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention_type = attention_type
        if attention_type == 'additive':
            self.attention = AdditiveAttention(hidden_size)
        elif attention_type == 'dot':
            self.attention = DotProductAttention(hidden_size)
        else:
            raise ValueError("Invalid attention type. Choose 'additive' or 'dot'.")
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)
        
        if self.attention_type == 'additive':
           # Current AdditiveAttention returns scores: (batch_size, seq_len)
           attention_scores = self.attention(lstm_out, lstm_out) # query=lstm_out, key=lstm_out
           attention_weights = torch.softmax(attention_scores, dim=1) # (batch_size, seq_len)

           # Apply weights to lstm_out to get attended_lstm_out per time step
           # attended_lstm_out will be lstm_out weighted by attention.
           # For example, element-wise multiplication:
           attended_lstm_out = lstm_out * attention_weights.unsqueeze(-1) # (batch, seq, hidden) * (batch, seq, 1)
           # You could also explore other ways to combine, e.g., attended_lstm_out = lstm_out + (lstm_out * attention_weights.unsqueeze(-1))

        elif self.attention_type == 'dot':
           # DotProductAttention, as implemented, already returns an attended sequence:
           # Q, K, V are all lstm_out in this self-attention scenario.
           # The output of self.attention will be (batch_size, seq_len, hidden_size)
           attended_lstm_out = self.attention(lstm_out, lstm_out, lstm_out)

        else:
           raise ValueError("Invalid attention type. Choose 'additive' or 'dot'.")

        # Pass the attended sequence (retaining seq_len) to the fully connected layer
        # This applies the fc layer to each time step independently.
        output = self.fc(attended_lstm_out)    # output shape: (batch_size, seq_len, output_size)
        return output


