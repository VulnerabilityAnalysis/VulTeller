import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def max_pooling(data, input_lens):
    B, N, T, H = data.size()
    idx = torch.arange(T, device=data.device).unsqueeze(0).expand(B, N, -1)
    idx = idx > input_lens.unsqueeze(2)
    idx = idx.unsqueeze(3).expand(-1, -1, -1, H)*float('-inf')
    ret = (data.float() + idx.float()).max(2)[0]
    return ret


class PathEncoder(nn.Module):

    def __init__(self, input_dim, emb_dim, enc_hid_dim):
        super(PathEncoder, self).__init__()
        self.hidden_dim = enc_hid_dim
        self.n_node = None
        self.extra = True

        self.node_embedding = nn.Embedding(input_dim, emb_dim)
        self.path_encoder = nn.GRU(self.hidden_dim, self.hidden_dim,
                                   bidirectional=True, batch_first=True)
        self.transit = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.rank_proj = nn.Linear(self.hidden_dim * 2 + 1, self.hidden_dim * 2)
        self.h_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.c_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def forward(self, src, src_len=None):
        nodes, paths = src
        self.n_node = nodes.size(1)

        node_embeddings = self.node_embedding(nodes)  # [B, N, T, H]
        node_embeddings = torch.max(node_embeddings, dim=2)[0]  # [B, N, H]
        # node_embeddings = max_pooling(node_embeddings, src_len[1])  # [B, N, H]

        all_v = []
        base_v, last_state = self.path_encoder(node_embeddings)
        for idx, (x, p) in enumerate(zip(node_embeddings, paths)):
            nx = torch.cat([x, torch.zeros((1, x.size(-1)), device=x.device)], dim=0)  # [N, H]
            nx, _ = self.path_encoder(nx[p])  # [P,L,H]
            if self.extra:
                nx = torch.cat([nx, 1 / (torch.arange(p.size(0), device=x.device).
                                         unsqueeze(1).
                                         unsqueeze(1).
                                         expand(-1, p.size(1), 1) + 1)], dim=2)
                nx = self.rank_proj(nx)

            v = torch.zeros(self.n_node, self.hidden_dim * 2, device=x.device)
            for i in range(self.n_node):
                if torch.any(p == i):
                    v[i] = torch.max(nx[p == i], dim=0)[0]
                else:
                    v[i] = self.transit(x[i])
            all_v.append(v)
        all_v = base_v + torch.stack(all_v)  # [B, N, H]
        if type(last_state) is tuple:
            last_hidden, last_cell = last_state
            init_decoder_hidden = torch.cat((last_hidden[0], last_hidden[1]), 1)

            init_decoder_cell = torch.cat((last_cell[0], last_cell[1]), 1)
            last_state = (self.h_proj(init_decoder_hidden), self.c_proj(init_decoder_cell))
        else:

            last_state = self.h_proj(torch.max(all_v, 1)[0])

        return all_v, last_state


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention = [batch size, src len]

        attention = attention.masked_fill(~mask, -float('inf'))

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.attention = Attention(enc_hid_dim, dec_hid_dim)

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn_type = 'GRU'
        self.rnn = getattr(nn, self.rnn_type + 'Cell')((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, dec_state, encoder_outputs, mask):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        # mask = [batch size, src len]
        if self.rnn_type == 'LSTM':
            dec_hidden, _ = dec_state
        else:
            dec_hidden = dec_state
        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        a = self.attention(dec_hidden, encoder_outputs, mask)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        dec_state = self.rnn(rnn_input.squeeze(0), dec_state)

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        # assert (output == hidden).all()
        if self.rnn_type == 'LSTM':
            output, _ = dec_state
        else:
            output = dec_state
        embedded = embedded.squeeze(0)
        # output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, dec_state, a.squeeze(1)


class MTModel(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 device):
        super().__init__()
        self.encoder = PathEncoder(input_dim, embedding_dim, hidden_dim)
        self.tagger = nn.Linear(hidden_dim * 2, 1)
        self.decoder = Decoder(output_dim, embedding_dim, hidden_dim, hidden_dim, 0)
        self.device = device

    def sequence_mask(self, lengths):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (torch.arange(0, max_len, device=lengths.device)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))

    def decode(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # src_len = [batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, dec_init_state = self.encoder(src, src_len)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        mask = self.sequence_mask(src_len[0])

        # mask = [batch size, src len]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state, all encoder hidden states
            #  and mask
            # receive output tensor (predictions) and new hidden state
            output, dec_init_state, _ = self.decoder(input, dec_init_state, encoder_outputs, mask)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs

    def forward(self, src, src_len, choice, trg=None, teacher_force=0.5):
        # src = [batch size, src len]
        # src_len = [batch size]
        # trg = [batch size, trg len]

        if choice == 'loc':
            encoder_outputs, hidden = self.encoder(src, src_len)
            return torch.sigmoid(self.tagger(encoder_outputs)), None
        elif choice == 'desc':

            return None, self.decode(src, src_len, trg, teacher_force)
        else:
            encoder_outputs, hidden = self.encoder(src, src_len)
            return torch.sigmoid(self.tagger(encoder_outputs)), self.decode(src, src_len, trg, teacher_force)
