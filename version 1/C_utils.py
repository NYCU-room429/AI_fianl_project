# model.py
import torch
import torch.nn as nn
import config # To access N_MELS for dummy pass if needed

class InstrumentCRNN(nn.Module):
    def __init__(self, num_classes, n_mels=config.N_MELS, 
                 num_spec_frames=config.NUM_SPECTROGRAM_FRAMES, 
                 cnn_filters=config.MODEL_CNN_FILTERS, 
                 rnn_units=config.MODEL_RNN_UNITS,
                 attention_heads=config.MODEL_ATTENTION_HEADS, 
                 dropout_rate=config.MODEL_DROPOUT_RATE):
        super(InstrumentCRNN, self).__init__()
        
        self.bn0 = nn.BatchNorm2d(1) 

        cnn_layers = []
        in_channels = 1 
        current_freq_dim = n_mels

        for i, out_filters in enumerate(cnn_filters):
            cnn_layers.append(nn.Conv2d(in_channels, out_filters, kernel_size=3, padding=1, bias=False))
            cnn_layers.append(nn.BatchNorm2d(out_filters))
            cnn_layers.append(nn.ELU(inplace=True))
            cnn_layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, bias=False))
            cnn_layers.append(nn.BatchNorm2d(out_filters))
            cnn_layers.append(nn.ELU(inplace=True))

            freq_pool = 2 if current_freq_dim >= 4 else 1 
            time_pool = 2 
            cnn_layers.append(nn.MaxPool2d(kernel_size=(freq_pool, time_pool)))
            current_freq_dim //= freq_pool
            
            cnn_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_filters
        
        self.cnn = nn.Sequential(*cnn_layers)

        with torch.no_grad():
            dummy_input = torch.randn(1, 1, n_mels, num_spec_frames) 
            cnn_out_dummy = self.cnn(self.bn0(dummy_input)) 
            self.cnn_output_channels = cnn_out_dummy.shape[1]
            self.cnn_output_freq_dim = cnn_out_dummy.shape[2]
            # self.cnn_output_time_dim = cnn_out_dummy.shape[3] # Not directly used after this

        self.rnn_input_features = self.cnn_output_channels * self.cnn_output_freq_dim
        self.adaptive_pool_rnn_time = nn.AdaptiveAvgPool1d(num_spec_frames)

        self.gru = nn.GRU(input_size=self.rnn_input_features,
                          hidden_size=rnn_units,
                          num_layers=2, batch_first=True, bidirectional=True,
                          dropout=dropout_rate if 2 > 1 else 0)

        self.attention = nn.MultiheadAttention(embed_dim=rnn_units * 2,
                                               num_heads=attention_heads,
                                               dropout=dropout_rate, batch_first=True)
        self.attn_norm = nn.LayerNorm(rnn_units * 2)
        self.fc_out = nn.Linear(rnn_units * 2, num_classes)

    def forward(self, x):
        x = self.bn0(x)
        x = self.cnn(x) 
        b, c, f, t_cnn = x.shape
        x = x.view(b, c * f, t_cnn) 
        x = self.adaptive_pool_rnn_time(x)
        x = x.permute(0, 2, 1)
        rnn_out, _ = self.gru(x)
        attn_out, _ = self.attention(rnn_out, rnn_out, rnn_out)
        x = self.attn_norm(rnn_out + attn_out) 
        logits = self.fc_out(x)
        return logits