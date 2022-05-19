import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multinomial import Multinomial
import numpy as np

from encoders import *


# parts of the code has been
# adapted from: https://github.com/stangelid/qt

class SequenceQuantizerSoftEMA(nn.Module):
    def __init__(self,
                 codebook_size,
                 d_model,
                 l1_cost=1000,
                 entropy_cost=0.00005,
                 num_samples=10,
                 temp=1.0,
                 epsilon=1e-5,
                 padding_idx=None):
        super(SequenceQuantizerSoftEMA, self).__init__()

        self.d_model = d_model
        self.codebook_size = codebook_size
        self.padding_idx = padding_idx

        self.codebook = nn.Parameter(torch.FloatTensor(self.codebook_size,
                                                       self.d_model),
                                     requires_grad=True)
        torch.nn.init.xavier_uniform_(self.codebook)

        self.l1_cost = l1_cost
        self.entropy_cost = entropy_cost
        self.num_samples = num_samples
        self.temp = temp

        self._epsilon = epsilon

    def entropy(self, tensor):
        return torch.mean(
            torch.sum(-1 *
                      torch.matmul(F.log_softmax(tensor, dim=1), tensor.t()),
                      dim=1))

    def forward(self, inputs, l1_cost=None, entropy_cost=None, temp=None):

        if l1_cost is None:
            l1_cost = self.l1_cost

        if entropy_cost is None:
            entropy_cost = self.entropy_cost

        if temp is None:
            temp = self.temp

        # inputs can be:
        # 2-dimensional [B x E]         (already flattened)
        # 3-dimensional [B x T x E]     (e.g., batch of sentences)
        # 4-dimensional [B x S x T x E] (e.g., batch of documents)
        input_shape = inputs.size()

        # Flatten input
        flat_input = inputs.reshape(-1, self.d_model)

        # Calculate distances
        norm_C = self.codebook / self.codebook.norm(2, dim=1)[:, None]
        flat_input = flat_input / flat_input.norm(2, dim=1)[:, None]

        # (input size x codebook size)
        distances = F.softmax(torch.matmul(flat_input, norm_C.t()), dim=1)

        # Soft-quantize and unflatten
        reconstruction = torch.matmul(distances, norm_C).view(input_shape)

        l1_loss = nn.L1Loss()
        loss = l1_cost * l1_loss(distances, torch.zeros_like(
            distances)) + entropy_cost * self.entropy(distances)

        return reconstruction, loss

    def cluster(self, inputs):

        # inputs can be:
        # 2-dimensional [B x E]         (already flattened)
        # 3-dimensional [B x T x E]     (e.g., batch of sentences)
        # 4-dimensional [B x S x T x E] (e.g., batch of documents)
        input_shape = inputs.size()
        input_dims = inputs.dim()

        # Flatten input
        flat_input = inputs.reshape(-1, self.d_model)

        flat_input = flat_input / flat_input.norm(2, dim=1)[:, None]
        codebook = self.codebook / self.codebook.norm(2, dim=1)[:, None]

        # Calculate distances
        distances = F.softmax(torch.matmul(flat_input, codebook.t()).reshape(
            -1, self.output_nheads, codebook.shape[0]),
                              dim=2)

        reconstruction = torch.matmul(distances, codebook).view(input_shape)

        # Encoding
        encoding_indices = torch.argmax(distances,
                                        dim=1).reshape(-1, self.output_nheads)
        return reconstruction, encoding_indices, distances

    def set_codebook(self, new_codebook):
        self.codebook.copy_(new_codebook)


class TransformerDocumentQuantizerSoftEMA(SequenceQuantizerSoftEMA):
    def __init__(self,
                 codebook_size=64,
                 d_model=200,
                 temp=1.0,
                 num_samples=10,
                 l1_cost=1000,
                 entropy_cost=0.00005,
                 ema_decay=0.99,
                 epsilon=1e-5,
                 nlayers=3,
                 internal_nheads=4,
                 output_nheads=4,
                 d_ff=512,
                 dropout=0.1):
        assert d_model % output_nheads == 0, 'Number of output heads must divide d_model'
        super(TransformerDocumentQuantizerSoftEMA,
              self).__init__(codebook_size,
                             d_model,
                             l1_cost=l1_cost,
                             entropy_cost=entropy_cost,
                             temp=temp,
                             num_samples=num_samples,
                             epsilon=epsilon)
        self.nlayers = nlayers
        self.internal_nheads = internal_nheads
        self.output_nheads = output_nheads
        self.d_ff = d_ff
        self.dropout = dropout

        self.encoder = TransformerDocumentEncoder(
            d_model=d_model,
            sentence_nlayers=nlayers,
            sentence_internal_nheads=internal_nheads,
            sentence_output_nheads=output_nheads,
            sentence_d_ff=d_ff,
            dropout=dropout)

    def forward(self,
                inputs,
                padding_mask=None,
                quantize=True,
                residual_coeff=0.0,
                l1_cost=None,
                entropy_cost=None,
                temp=None):
        assert inputs.dim() == 4, \
                'Inputs must have 4 dimensions: [B x S x T x E]'

        out = self.encoder(inputs, padding_mask)
        if quantize:
            if residual_coeff > 0.0:
                reconstruction, loss = \
                        super(TransformerDocumentQuantizerSoftEMA, self).forward(out,
                                l1_cost=l1_cost, entropy_cost=entropy_cost, temp=temp)

                reconstruction = residual_coeff * out + (
                    1 - residual_coeff) * reconstruction
                return reconstruction, loss
            else:
                return super(TransformerDocumentQuantizerSoftEMA,
                             self).forward(out,
                                           l1_cost=l1_cost,
                                           entropy_cost=entropy_cost,
                                           temp=temp)
        return out, 0.0

    def cluster(self, inputs, padding_mask):
        assert inputs.dim() == 4, \
                'Inputs must have 4 dimensions: [B x S x T x E]'

        out = self.encoder(inputs, padding_mask)
        reconstruction, clusters, distances = \
                super(TransformerDocumentQuantizerSoftEMA, self).cluster(out)

        return out, reconstruction, clusters, distances
