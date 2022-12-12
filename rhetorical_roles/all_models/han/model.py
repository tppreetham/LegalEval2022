import torch.nn as nn
import torch

class wordEncoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, embedding_dim):
    super(wordEncoder, self).__init__()
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size

    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.gru = nn.GRU(embedding_dim, hidden_size, bidirectional = True)

  def forward(self, word, h0):
    word = self.embedding(word).unsqueeze(0).unsqueeze(1)
    out, h0 = self.gru(word, h0)
    return out, h0

class sentEncoder(nn.Module):
  def __init__(self, hidden_size):
    super(sentEncoder, self).__init__()
    self.hidden_size = hidden_size
    self.gru = nn.GRU(hidden_size, hidden_size, bidirectional = True)

  def forward(self, sentence, h0):
    sentence = sentence.unsqueeze(0).unsqueeze(1)
    out, h0 = self.gru(sentence)
    return out, h0

class HANModel(nn.Module):
  def __init__(self, wordEncoder, sentEncoder, num_classes, device):
    super(HANModel, self).__init__()
    self.wordEncoder = wordEncoder
    self.sentEncoder = sentEncoder
    self.device = device
    self.softmax = nn.Softmax(dim=1)
    # word-level attention
    self.word_attention = nn.Linear(self.wordEncoder.hidden_size*2, self.wordEncoder.hidden_size*2)
    self.u_w = nn.Linear(self.wordEncoder.hidden_size*2, 1, bias = False)

    # sentence-level attention
    self.sent_attention = nn.Linear(self.sentEncoder.hidden_size * 2, self.sentEncoder.hidden_size*2)
    self.u_s = nn.Linear(self.sentEncoder.hidden_size*2, 1, bias = False)

    # final layer
    self.dense_out = nn.Linear(self.sentEncoder.hidden_size*2, num_classes)
    self.log_softmax = nn.LogSoftmax()

  def forward(self, document):
    word_attention_weights = []
    sentenc_out = torch.zeros((document.size(0), 2, self.sentEncoder.hidden_size)).to(self.device)
    # iterate on sentences
    h0_sent = torch.zeros(2, 1, self.sentEncoder.hidden_size, dtype = float).to(self.device)
    for i in range(document.size(0)):
      sent = document[i]
      wordenc_out = torch.zeros((sent.size(0), 2, self.wordEncoder.hidden_size)).to(self.device)
      h0_word = torch.zeros(2, 1, self.wordEncoder.hidden_size, dtype = float).to(self.device)
      # iterate on words
      for j in range(sent.size(0)):
        _, h0_word = self.wordEncoder(sent[j], h0_word)
        wordenc_out[j] = h0_word.squeeze()
      wordenc_out = wordenc_out.view(wordenc_out.size(0), -1)
      u_word = torch.tanh(self.word_attention(wordenc_out))
      word_weights = self.softmax(self.u_w(u_word))
      word_attention_weights.append(word_weights)
      sent_summ_vector = (u_word * word_weights).sum(axis=0)

      _, h0_sent = self.sentEncoder(sent_summ_vector, h0_sent)
      sentenc_out[i] = h0_sent.squeeze()
    sentenc_out = sentenc_out.view(sentenc_out.size(0), -1)
    u_sent = torch.tanh(self.sent_attention(sentenc_out))
    sent_weights = self.softmax(self.u_s(u_sent))
    doc_summ_vector = (u_sent * sent_weights).sum(axis=0)
    out = self.dense_out(doc_summ_vector)
    return word_attention_weights, sent_weights, self.log_softmax(out)
