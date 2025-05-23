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
        self.fc = nn.Linear(hidden_size, output_size)
    
    # Updated forward to accept input_mask (even if not directly used by this simple LSTM)
    # This maintains a consistent interface if you switch between models.
    def forward(self, x, input_mask=None): # Added input_mask argument
        # x: (batch, seq_len, input_size)
        # input_mask: (batch, seq_len), 1 for real data, 0 for padding
        
        # If you wanted to use PackedSequence for efficiency with LSTMs:
        # if input_mask is not None:
        #     lengths = input_mask.sum(dim=1).cpu().to(torch.int64)
        #     x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_size)
        
        # If you used PackedSequence:
        # lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        output = self.fc(lstm_out)    # output shape: (batch_size, seq_len, output_size)
        return output

class AdditiveAttention(nn.Module):
    """Additive Attention Mechanism (Bahdanau)."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.energy_layer = nn.Linear(hidden_dim, 1)

    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch_size, seq_len, hidden_dim)
        # mask: (batch_size, seq_len) -> indicates valid key/value positions (1 for valid, 0 for pad)
        
        # Project query and key
        projected_query = self.query_layer(query)  # (batch_size, seq_len_q, hidden_dim)
        projected_key = self.key_layer(key)        # (batch_size, seq_len_k, hidden_dim)

        # Calculate energy. For self-attention seq_len_q == seq_len_k
        # We need to make query and key broadcastable for addition if their seq_len differ.
        # For self-attention on LSTM outputs, query, key, value are all lstm_out (seq_len is the same)
        # So, we can do (B, S, H) + (B, S, H) -> (B, S, H)
        # If query is decoder state (B, 1, H) and key is encoder_outputs (B, S_enc, H)
        # then projected_query.unsqueeze(2) + projected_key.unsqueeze(1) might be needed.
        # Assuming self-attention here:
        scores = self.energy_layer(torch.tanh(projected_query + projected_key)).squeeze(-1) # (batch_size, seq_len)

        if mask is not None:
            # mask is (batch_size, seq_len_k)
            # Apply mask to scores before softmax
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len)
        
        # Context vector: (batch_size, seq_len, hidden_dim)
        # (B, S, 1) * (B, S, H) -> (B, S, H)
        context_vector = attention_weights.unsqueeze(-1) * value 
        
        # If you want a single context vector summarizing the sequence (common in seq2seq decoders):
        # context_vector_summed = torch.sum(attention_weights.unsqueeze(-1) * value, dim=1) # (B, H)
        
        return context_vector, attention_weights


class DotProductAttention(nn.Module):
    """Dot-Product Attention Mechanism."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value, mask=None):
        # query: (batch_size, seq_len_q, hidden_dim)
        # key:   (batch_size, seq_len_k, hidden_dim)
        # value: (batch_size, seq_len_v, hidden_dim) (seq_len_k should be == seq_len_v)
        # mask:  (batch_size, seq_len_k) -> mask for keys/values
        
        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)

        # Scaled Dot-Product Attention
        # (B, S_q, H) @ (B, H, S_k) -> (B, S_q, S_k)
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(Q.size(-1))

        if mask is not None:
            # mask is (B, S_k). Needs to be (B, 1, S_k) to broadcast over S_q
            # This masks out padding tokens in the key/value sequences
            mask_for_scores = mask.unsqueeze(1) # (B, 1, S_k)
            attention_scores = attention_scores.masked_fill(mask_for_scores == 0, -1e9)

        attention_weights = torch.softmax(attention_scores, dim=-1) # Softmax over S_k
        
        # Context vector
        # (B, S_q, S_k) @ (B, S_v, H) -> (B, S_q, H)
        context_vector = torch.bmm(attention_weights, V)
        return context_vector, attention_weights


class LSTM_Network_with_Attention(nn.Module):
    """LSTM Network with Attention Mechanism."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, attention_type='dot', bidirectional_lstm=False):
        super().__init__()
        self.bidirectional_lstm = bidirectional_lstm
        self.lstm_hidden_size = hidden_size # Store original LSTM hidden size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional_lstm)
        
        # Adjust attention input dimension if LSTM is bidirectional
        attention_input_dim = hidden_size * 2 if bidirectional_lstm else hidden_size
        
        self.attention_type = attention_type
        if attention_type == 'additive':
            self.attention = AdditiveAttention(attention_input_dim)
        elif attention_type == 'dot':
            self.attention = DotProductAttention(attention_input_dim)
        else:
            raise ValueError("Invalid attention type. Choose 'additive' or 'dot'.")
            
        # The FC layer takes the output of the attention mechanism.
        # If attention returns a sequence (B, S, H_att), then fc input is H_att.
        # H_att is typically the same as attention_input_dim.
        self.fc = nn.Linear(attention_input_dim, output_size)
        self.attention_weights = None # To store attention weights for visualization/analysis

    # Corrected forward method to accept input_mask
    def forward(self, x, input_mask=None):
        # x: (batch, seq_len, input_size)
        # input_mask: (batch, seq_len), 1 for real data, 0 for padding
        
        # If using PackedSequence (optional, for potential speedup with variable lengths):
        # original_lengths = None
        # if input_mask is not None:
        #     original_lengths = input_mask.sum(dim=1).cpu().to(torch.int64)
        #     # Important: pack_padded_sequence requires lengths to be > 0
        #     if not all(l > 0 for l in original_lengths):
        #         # Handle cases where a whole sequence might be padding if that's possible
        #         # For now, assume valid sequences have at least one non-padded element
        #         # Or, filter out zero-length sequences before this point.
        #         pass # Potentially skip packing if zero lengths are present and unhandled
        #     else:
        #        x = nn.utils.rnn.pack_padded_sequence(x, original_lengths, batch_first=True, enforce_sorted=False)

        lstm_out, (hn, cn) = self.lstm(x)
        # lstm_out: (batch, seq_len, num_directions * hidden_size)
        # hn: (num_layers * num_directions, batch, hidden_size)
        # cn: (num_layers * num_directions, batch, hidden_size)

        # If PackedSequence was used:
        # if input_mask is not None and original_lengths is not None and all(l > 0 for l in original_lengths):
        #    lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # The input_mask (batch_size, seq_len) indicates valid time steps in the lstm_out.
        # This mask is used for the key/value sequences in self-attention.
        
        # Query, Key, Value for self-attention are all derived from lstm_out
        query_seq = lstm_out
        key_seq = lstm_out
        value_seq = lstm_out
        
        if self.attention_type == 'additive':
            attended_lstm_out, self.attention_weights = self.attention(query_seq, key_seq, value_seq, mask=input_mask)
        elif self.attention_type == 'dot':
            attended_lstm_out, self.attention_weights = self.attention(query_seq, key_seq, value_seq, mask=input_mask)
        else:
           raise ValueError("Invalid attention type. Choose 'additive' or 'dot'.")

        # Pass the attended sequence (retaining seq_len) to the fully connected layer
        # attended_lstm_out shape: (batch_size, seq_len, num_directions * hidden_size)
        output = self.fc(attended_lstm_out) # output shape: (batch_size, seq_len, output_size)
        return output
