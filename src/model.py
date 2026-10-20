# model.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for variable-length sequences"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for variable-length sequences
    
    Benefits over Sinusoidal PE:
    - Learns relative timing relationships specific to biomechanics tasks
    - Generalizes better to subjects with different movement speeds
    - Adaptive to the actual patterns in the data
    - Each position learns an embedding optimized for the task
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)
        self.max_len = max_len
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        
        Returns:
            x + positional_embedding, shape [batch_size, seq_len, embedding_dim]
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos_embed = self.embedding(positions)  # (seq_len, d_model)
        x = x + pos_embed.unsqueeze(0)  # Add to batch
        return self.dropout(x)


def _tpad(x, kernel_size, causal, dilation=1):
    """Left-pad the last (time) axis of a (batch, channels, time) tensor by (k - 1) * dilation when causal, so a
    padding-0 convolution sees frames <= t only; identity otherwise (the conv then pads symmetrically)."""
    return F.pad(x, ((kernel_size - 1) * dilation, 0)) if causal else x


class ImprovedConvFormer(nn.Module):
    """
    Fixed ConvFormer for sequence-to-sequence prediction with:
    - Proper temporal output preservation
    - Metadata integration
    - Residual connections in CNN
    - Batch normalization for stability
    - Pre-LN Transformer architecture
    causal=True: convolutions see frames <= t only (left padding) and self-attention gets a causal mask, so the
    output at frame t depends on frames <= t (see LookaheadShift in model_factory for a fixed look-ahead).
    """
    def __init__(self,
                 input_size,
                 output_size,
                 metadata_input_size=2,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=3,
                 dim_feedforward=1024,
                 dropout=0.1,
                 num_cnn_filters=128,
                 max_sequence_length=5000,
                 causal=False):
        super().__init__()
        self.causal = causal
        sym = lambda k: 0 if causal else (k - 1) // 2

        # Multi-scale CNN with residual connections (from Response 1)
        self.conv1 = nn.Conv1d(input_size, num_cnn_filters, kernel_size=3, padding=sym(3))
        self.conv2 = nn.Conv1d(num_cnn_filters, num_cnn_filters, kernel_size=5, padding=sym(5))
        self.conv3 = nn.Conv1d(num_cnn_filters, num_cnn_filters, kernel_size=7, padding=sym(7))

        # Batch normalization for training stability
        self.bn1 = nn.BatchNorm1d(num_cnn_filters)
        self.bn2 = nn.BatchNorm1d(num_cnn_filters)
        self.bn3 = nn.BatchNorm1d(num_cnn_filters)

        self.gelu = nn.GELU()
        self.dropout_cnn = nn.Dropout(dropout)

        # Project CNN features to d_model
        self.projection = nn.Linear(num_cnn_filters, d_model)
        self.projection_dropout = nn.Dropout(dropout)

        # Metadata integration (will be expanded to each timestep for stronger signal flow)
        self.metadata_projection = nn.Linear(metadata_input_size, d_model)

        # Learnable positional encoding (learns relative timing, generalizes better to variable speeds)
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout, max_sequence_length)

        # Pre-transformer layer normalization (from Response 1)
        self.pre_transformer_norm = nn.LayerNorm(d_model)

        # Transformer encoder with Pre-LN architecture (more stable)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN is more stable than Post-LN
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Output projection - CRITICAL FIX: maintains sequence dimension
        self.output_layer = nn.Linear(d_model, output_size)

        # Proper weight initialization
        self._init_weights()

        # To store attention weights
        self.attention_weights = None

    def _init_weights(self):
        """Xavier initialization for better convergence"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_attention_maps(self, x, src_key_padding_mask=None):
        """A helper to get attention maps from the encoder."""
        maps = []
        for layer in self.transformer_encoder.layers:
            # We need to get the attention map from the multi-head attention layer
            # This is a bit of a hack, as we are accessing internal components
            # For a production system, a custom TransformerEncoder might be better.
            _, attn_map = layer.self_attn(layer.norm1(x), layer.norm1(x), layer.norm1(x),
                                          key_padding_mask=src_key_padding_mask,
                                          need_weights=True,
                                          average_attn_weights=False)
            maps.append(attn_map)
        return torch.stack(maps)

    def forward(self, x, metadata, input_mask=None, return_attention=False):
        """
        Args:
            x: (batch_size, seq_len, input_size)
            metadata: (batch_size, metadata_input_size)
            input_mask: (batch_size, seq_len) - 1 for valid, 0 for padding

        Returns:
            output: (batch_size, seq_len, output_size)
        """
        batch_size, seq_len, _ = x.shape

        # === CNN Feature Extraction with Residual Connections ===
        x_t = x.transpose(1, 2)  # (batch, input_size, seq_len)

        # First conv block
        conv1_out = self.conv1(_tpad(x_t, 3, self.causal))
        conv1_out = self.bn1(conv1_out)
        conv1_out = self.gelu(conv1_out)
        conv1_out = self.dropout_cnn(conv1_out)

        # Second conv block with residual
        conv2_out = self.conv2(_tpad(conv1_out, 5, self.causal))
        conv2_out = self.bn2(conv2_out)
        conv2_out = self.gelu(conv2_out)
        conv2_out = self.dropout_cnn(conv2_out)
        conv2_out = conv2_out + conv1_out  # Residual connection

        # Third conv block with residual
        conv3_out = self.conv3(_tpad(conv2_out, 7, self.causal))
        conv3_out = self.bn3(conv3_out)
        conv3_out = self.gelu(conv3_out)
        conv3_out = self.dropout_cnn(conv3_out)
        conv3_out = conv3_out + conv2_out  # Residual connection

        # Back to (batch, seq_len, num_cnn_filters)
        x_cnn = conv3_out.transpose(1, 2)

        # === Project to d_model ===
        x_proj = self.projection(x_cnn)
        x_proj = self.projection_dropout(x_proj)

        # === Metadata Integration: Expand to all timesteps (forces metadata signal through network) ===
        if metadata is not None:
            meta_proj = self.metadata_projection(metadata)  # (batch, d_model)
            # Expand metadata to match sequence: (batch, d_model) → (batch, seq_len, d_model)
            meta_expanded = meta_proj.unsqueeze(1).expand(-1, seq_len, -1)
            # Add to CNN features at every timestep (ensures mass signal flows through)
            x_combined = x_proj + meta_expanded
        else:
            x_combined = x_proj

        # === Positional Encoding (Learnable, not Sinusoidal) ===
        x_pos = self.pos_encoder(x_combined)

        # === Pre-Transformer Normalization ===
        x_norm = self.pre_transformer_norm(x_pos)

        # === Transformer Encoder ===
        if input_mask is not None:
            # PyTorch expects True for masked (padding) positions
            src_key_padding_mask = (input_mask == 0)
        else:
            src_key_padding_mask = None

        if return_attention:
            # This is a bit of a hack to get attention maps.
            # A cleaner way is to implement a custom TransformerEncoder that returns them.
            self.attention_weights = self._get_attention_maps(x_norm, src_key_padding_mask)

        # causal: frame t attends to frames <= t only (True = not allowed)
        attn_mask = (torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
                     if self.causal else None)
        transformer_out = self.transformer_encoder(
            x_norm,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        # === Output Projection (PRESERVES SEQUENCE DIMENSION) ===
        output = self.output_layer(transformer_out)  # (batch, seq_len, output_size)

        return output


class LightConvFormer(nn.Module):
    """
    Lightweight alternative - simpler architecture that may train faster
    and generalize better with limited data
    """
    def __init__(self,
                 input_size,
                 output_size,
                 metadata_input_size=2,
                 d_model=128,
                 nhead=4,
                 num_encoder_layers=2,
                 dim_feedforward=512,
                 dropout=0.1,
                 max_sequence_length=5000):
        super().__init__()

        # Simple CNN embedding
        self.conv_embedding = nn.Sequential(
            nn.Conv1d(input_size, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Metadata projection
        self.metadata_projection = nn.Linear(metadata_input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_sequence_length)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Simple output projection
        self.output_layer = nn.Linear(d_model, output_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, metadata, input_mask=None):
        # CNN embedding
        x_emb = self.conv_embedding(x.transpose(1, 2)).transpose(1, 2)

        # Metadata injection
        if metadata is not None:
            meta_proj = self.metadata_projection(metadata)
            x_emb = x_emb + meta_proj.unsqueeze(1)

        # Positional encoding and normalization
        x_pos = self.pos_encoder(x_emb)
        x_norm = self.layer_norm(x_pos)

        # Transformer
        if input_mask is not None:
            mask = (input_mask == 0)
        else:
            mask = None

        transformer_out = self.transformer_encoder(x_norm, src_key_padding_mask=mask)

        # Output
        return self.output_layer(transformer_out)

class LSTM_Network(nn.Module):
    """Simple LSTM Network that predicts for every timeframe."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=4, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    # Updated forward to accept input_mask (even if not directly used by this simple LSTM)
    # This maintains a consistent interface if you switch between models.
    def forward(self, x, metadata=None, input_mask=None): # Added metadata argument
        # x: (batch, seq_len, input_size)
        # input_mask: (batch, seq_len), 1 for real data, 0 for padding
        
        # If you wanted to use PackedSequence for efficiency with LSTMs:
        # if input_mask is not None:
        #     lengths = input_mask.sum(dim=1).cpu().to(torch.int64)
        #     x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_size)
        
        # If you used PackedSequence:
        # lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        lstm_out = self.dropout(lstm_out)
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
    """LSTM Network with Attention Mechanism and metadata integration for initial states."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, attention_type='dot', bidirectional_lstm=False, metadata_input_size=2, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional_lstm = bidirectional_lstm
        self.lstm_directions = 2 if bidirectional_lstm else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional_lstm, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        
        # Linear layers to project metadata to initial hidden and cell states
        # The target size for each state (h_0 or c_0) for one layer is (batch_size, hidden_size)
        # So, these layers will project metadata (batch_size, metadata_input_size) to (batch_size, hidden_size * num_layers * lstm_directions)
        # which will then be reshaped.
        self.meta_to_hidden = nn.Linear(metadata_input_size, hidden_size * num_layers * self.lstm_directions)
        self.meta_to_cell = nn.Linear(metadata_input_size, hidden_size * num_layers * self.lstm_directions)

        # Adjust attention input dimension if LSTM is bidirectional
        attention_input_dim = hidden_size * self.lstm_directions
        
        self.attention_type = attention_type
        if attention_type == 'additive':
            self.attention = AdditiveAttention(attention_input_dim)
        elif attention_type == 'dot':
            self.attention = DotProductAttention(attention_input_dim)
        else:
            raise ValueError("Invalid attention type. Choose 'additive' or 'dot'.")
            
        self.fc = nn.Linear(attention_input_dim, output_size)
        self.attention_weights = None # To store attention weights

    def forward(self, x, metadata, input_mask=None):
        # x: (batch_size, seq_len, input_size)
        # metadata: (batch_size, metadata_input_size) - e.g., (batch_size, 2) for weight and height
        # input_mask: (batch_size, seq_len), 1 for real data, 0 for padding
        
        batch_size = x.size(0)
        
        # Generate initial hidden and cell states from metadata
        # Project metadata to the total size needed for all layers and directions
        h_0_proj = self.meta_to_hidden(metadata) # (batch_size, hidden_size * num_layers * lstm_directions)
        c_0_proj = self.meta_to_cell(metadata)   # (batch_size, hidden_size * num_layers * lstm_directions)

        # Reshape to (num_layers * lstm_directions, batch_size, hidden_size)
        h_0 = h_0_proj.view(batch_size, self.num_layers * self.lstm_directions, self.hidden_size).permute(1, 0, 2).contiguous()
        c_0 = c_0_proj.view(batch_size, self.num_layers * self.lstm_directions, self.hidden_size).permute(1, 0, 2).contiguous()

        # Pass data and initial states to LSTM
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        lstm_out = self.dropout(lstm_out)
        # lstm_out: (batch_size, seq_len, num_directions * hidden_size)

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

        attended_lstm_out = self.dropout(attended_lstm_out)
        output = self.fc(attended_lstm_out) # output shape: (batch_size, seq_len, output_size)
        return output

class CNN_Network(nn.Module):
    """1D CNN Network for sequence data."""
    def __init__(self, input_size, output_size, num_filters=64, kernel_size=3, dropout=0.5, causal=False):
        super().__init__()
        self.causal, self.k = causal, kernel_size
        pad = 0 if causal else (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size, padding=pad)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, padding=pad)
        self.fc = nn.Linear(num_filters, output_size)

    def forward(self, x, metadata=None, input_mask=None):
        # x shape: (batch_size, seq_len, input_size)
        # Conv1d expects (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)

        x = self.conv1(_tpad(x, self.k, self.causal))
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(_tpad(x, self.k, self.causal))
        x = self.relu(x)

        # permute back to (batch_size, seq_len, num_filters)
        x = x.permute(0, 2, 1)

        output = self.fc(x)
        return output

# ── ST-GCN (spatio-temporal graph convolution over markers) ──────────────────
# Re-implementation of the model family of Meng et al. 2026 (Front. Bioeng. Biotechnol. 14:1871692, no code
# released), which follows ST-GCN (Yan et al. 2018) with marker nodes instead of joints. Built from the paper's
# description: nodes = markers, node features = position, velocity and acceleration, graph convolution over an
# anatomical marker graph, temporal convolution per node. Meng regress a fixed 101-frame stance waveform (strides
# 2 at blocks 4 and 7, global average pooling -> FC to 3 x 101); here the head is per-frame on whole trials.

# Body segments as sets of marker short names. Every pair of markers present in the same segment is an edge;
# markers shared by two segments (shoulders, elbows, knees, ankles, pelvis) join them along the kinematic chain.
# Names cover the Chapter 1 / combined 20-marker set and the 40-marker wind-turbine set.
_PELVIS = {"LASI", "RASI", "LPSI", "RPSI"}
MARKER_SEGMENTS = [
    {"LAH", "RAH", "LPH", "RPH", "C7"},                                   # head and neck
    {"LSHO", "RSHO", "C7", "MChest"} | _PELVIS,                          # torso (shoulders to pelvis)
    {"LSHO", "LUPA", "LELBL", "LELBM"},                                  # left upper arm
    {"LELBL", "LELBM", "LFRM", "LWRL", "LWRM"},                          # left forearm
    {"RSHO", "RUPA", "RELBL", "RELBM"},                                  # right upper arm
    {"RELBL", "RELBM", "RFRM", "RWRL", "RWRM"},                          # right forearm
    _PELVIS | {"LTHI", "LKNE", "LKNEL", "LKNEM"},                        # left thigh (hip joint via pelvis)
    {"LKNE", "LKNEL", "LKNEM", "LTIB", "LANKL", "LANKM"},                # left shank
    {"LANKL", "LANKM", "LHEE", "LTOE"},                                  # left foot
    _PELVIS | {"RTHI", "RKNE", "RKNEL", "RKNEM"},                        # right thigh
    {"RKNE", "RKNEL", "RKNEM", "RTIB", "RANKL", "RANKM"},                # right shank
    {"RANKL", "RANKM", "RHEE", "RTOE"},                                  # right foot
]



class CNN_LSTM_Network(nn.Module):
    """CNN-LSTM Network for sequence data."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, num_filters=64, kernel_size=3, dropout=0.5,
                 causal=False):
        super().__init__()
        self.causal, self.k = causal, kernel_size   # the LSTM is forward-only, so causal only changes the conv
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size,
                               padding=0 if causal else (kernel_size - 1) // 2)
        self.relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(dropout)

        self.lstm = nn.LSTM(num_filters, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout_lstm = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, metadata=None, input_mask=None):
        # x shape: (batch_size, seq_len, input_size)
        # Conv1d expects (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)

        cnn_out = self.conv1(_tpad(x, self.k, self.causal))
        cnn_out = self.relu(cnn_out)
        cnn_out = self.dropout_cnn(cnn_out)

        # permute back to (batch_size, seq_len, num_filters) for LSTM
        cnn_out = cnn_out.permute(0, 2, 1)

        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = self.dropout_lstm(lstm_out)

        output = self.fc(lstm_out)
        return output