import torch
import torch.nn as nn
import torch.nn.functional as F


class HAP(nn.Module):
	def __init__(self, hidden_dim):
		super(HAP, self).__init__()

		self.W1 = nn.Linear(hidden_dim, hidden_dim)
		self.W2 = nn.Linear(hidden_dim, hidden_dim)
		self.v = nn.Linear(hidden_dim, 1)


	def forward(self, E, di, mask=None):
		# E = {e1, e2, ..., ej} size: bs * n * hidden_dim
		# di size: bs * 1 * hidden_dim
		# mask size: bs * n

		# Di size bs * n * hidden_dim
		# Di = di.expand(-1, E.size(1), -1)
		Di = di.repeat(1, E.size(1), 1)

		# Ui size bs * n
		Ui = self.v(torch.tanh(self.W1(E) + self.W2(Di))).squeeze(2)
		ai = F.softmax(Ui, dim=1)

		# mask has the same size as Ui: bs * n
		if mask is not None:
			# Ui[mask] = float('-inf')
			ai[mask] = float('-inf')
		
		return Ui, ai


class OAE(nn.Module):
	def __init__(self, input_dim, hidden_dim, dropout):
		super(OAE, self).__init__()

		self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout)

	def forward(self, seq):
		# seq size: bs * T * input_dim

		output, (hn, cn) = self.lstm(seq)
		return output, (hn, cn)


class SRD(nn.Module):
	def __init__(self, input_dim, hidden_dim, dropout, tuple_len):
		super(SRD, self).__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.register_buffer('inp0', torch.zeros(1, input_dim))
		self.register_buffer('mask', torch.zeros(1, dtype=torch.uint8))

		self.attn = HAP(hidden_dim)
		self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout)

		self.conv1d = nn.Conv1d(tuple_len, 1, 1)

	def forward(self, inps, enc_output): # , h0, c0):
		# inps size bs * T * input_dim
		# enc_output size bs * T * hidden_dim 
		probs = []
		ptrs  = []

		# first initialize size bs * hidden_dim
		# mask size bs * n
		# Ui for un-normalized scores
		# bs_n size bs * 1
		inp0 = self.conv1d(enc_output)
		mask = self.mask.repeat(enc_output.size(0), enc_output.size(1))
		bs_n = torch.tensor(list(range(enc_output.size(0)))).unsqueeze(1)
		
		di, (h, c) = self.lstm(inp0)
		Ui, ai = self.attn(enc_output, di)

		_, ptr = torch.max(ai, dim=1, keepdim=True)
		# bs * n
		probs.append(Ui)
		ptrs.append(ptr)
		# bs * 1
		mask[bs_n, ptr] = True

		# ind = ptr.unsqueeze(2).expand(-1, -1, self.input_dim)
		ind = ptr.unsqueeze(2).repeat(1, 1, self.input_dim)
		# bs * 1 * input_dim
		inp = torch.gather(inps, dim=1, index=ind)
		# inp size bs * 1 * input_dim

		for i in range(enc_output.size(1)-1):

			di, (h, c) = self.lstm(inp, (h, c))

			Ui, ai = self.attn(enc_output, di, mask)
			'''
			need a mask to mask the indices already chosen
			'''
			prob, ptr = torch.max(ai, dim=1, keepdim=True)
			probs.append(Ui)
			ptrs.append(ptr)
			mask[bs_n, ptr] = True
			ind = ptr.unsqueeze(2).expand(-1, -1, self.input_dim)
			inp = torch.gather(inps, dim=1, index=ind)


		return torch.stack(probs, dim=1), torch.cat(ptrs, dim=1)
		# bs * T * n, bs * T * 1


class TaskNet(nn.Module):
	def __init__(self, input_dim, hidden_dim, dropout, tuple_len=3):
		super(TaskNet, self).__init__()

		self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
		self.OAE = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
		self.SRD = SRD(input_dim, hidden_dim, dropout, tuple_len=tuple_len)

		self.init_rnn()


	def forward(self, seq):
		# seq is encoder output
		# seq shape(bs, l, dim)
		
		trf_seq = seq.permute(1,0,2)
		# trf_seq shape(l, bs, dim)
		trf_output = self.OAE(trf_seq)
		# trf_output shape(l, bs, dim)
		# enc_output, (hn, cn) = self.encoder(seq)

		# probs, ptrs = self.decoder(enc_output, enc_output, hn, cn)
		trf_output = trf_output.permute(1,0,2)
		probs, ptrs = self.SRD(seq, trf_output)

		return probs, ptrs


	def init_rnn(self):
		def init_each(param_data, init_method):
			for idx in range(4):
				mul = param.size(0)//4
				init_method(param.data[idx*mul:(idx+1)*mul])

		for m in self.modules():
			if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
				for name, param in m.named_parameters():
					if 'weight_ih' in name:
						init_each(param.data, torch.nn.init.xavier_uniform_)
					elif 'weight_hh' in name:
						init_each(param.data, torch.nn.init.orthogonal_)
					elif 'bias' in name:
						param.data.fill_(0)












