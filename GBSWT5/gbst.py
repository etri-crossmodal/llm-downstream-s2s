"""
    Gradient-based Subword Tokenization(GBST) Layer implementation.

    based on lucidrains/charformer-pytorch implementation,
    which distributed under MIT License.

    original code location:
    https://github.com/lucidrains/charformer-pytorch/charformer_pytorch.py

    copyright (c) 2023~, ETRI LIRS. Jong-hun Shin.
"""
import math
import functools
import torch
import torch.nn.functional as F

from torch import einsum, nn
from einops.layers.torch import Rearrange
from einops import rearrange, repeat


# Block definition
_BLOCKS = (
    (1, 0), (2, 0), (3, 0), (4, 0),
    (6, 0), (9, 0),
    #(12, 0), (12, 3), (12, 6), (12, 9)
)

def pad_to_multiple(in_tensor, multiple, *, seq_dim, dim=-1, value=0.0):
    seqlen = in_tensor.shape[seq_dim]
    padded_len = math.ceil(seqlen / multiple) * multiple
    if seqlen == padded_len:
        return in_tensor
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(in_tensor, (*pad_offset, 0, padded_len - seqlen), value=value)


def masked_mean(in_tensor, mask, dim=-1):
    len_diff = len(in_tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * len_diff))]
    in_tensor.masked_fill_(~(mask.bool()), 0.)

    total_elems = mask.sum(dim=dim)
    mean = in_tensor.sum(dim=dim) / total_elems.clamp(min=1.)
    mean.masked_fill_((total_elems == 0), 0.)
    return mean.float()


class Depthwise1dConv(nn.Module):
    def __init__(self, in_dim, out_dim, krnl_size):
        super().__init__()
        self.convol = nn.Conv1d(in_dim, out_dim, krnl_size, groups=in_dim)
        self.proj = nn.Conv1d(out_dim, out_dim, 1)

    def forward(self, in_tensor):
        in_tensor = self.convol(in_tensor)
        return self.proj(in_tensor)


class Padding(nn.Module):
    def __init__(self, padding, value=0):
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, in_tensor):
        return F.pad(in_tensor, self.padding, value=self.value)


class GBSWT(nn.Module):
    """ Gradient-based Sub-Word Tokenizer implementation. """
    def __init__(self, embed_tokens,
                 max_block_size=None,
                 blocks=_BLOCKS,
                 downsample_factor=1,
                 score_consensus_attn=True,):
        super().__init__()
        num_tokens, dim = embed_tokens.weight.shape

        assert (max_block_size is not None) ^ (blocks is not None), \
                'max_block_size or blocks must be given.'
        if blocks is None:
            self.blocks = tuple(map(lambda elem: (elem, 0), range(1, max_block_size+1)))
        else:
            if not isinstance(blocks, tuple):
                raise ValueError('blocks must be assigned as a tuple')
            self.blocks = tuple(map(lambda elem: elem if isinstance(elem, tuple) else (elem, 0), blocks))
            if not all([(offset < block_size) for block_size, offset in self.blocks]):
                raise ValueError('Offset must be smaller than given block size.')
            max_block_size = max(list(map(lambda x: x[0], self.blocks)))

        assert downsample_factor <= max_block_size, \
            'downsample factor must be less than the max_block_size.'

        self.downsample_factor = downsample_factor
        self.score_consensus_attn = score_consensus_attn
        print(f"** GBSWT: Subword Block Combinations: {self.blocks}")
        print(f"** GBSWT: Downsampling factor: {self.downsample_factor}")

        def lcm(*num):
            return int(functools.reduce(lambda x, y: int((x * y) / math.gcd(x, y)), num, 1))

        self.block_pad_multiple = lcm(*[block_size for block_size, _ in self.blocks])
        #print(f"block_pad_multiple: {self.block_pad_multiple}")

        # layer definition
        self.embeds = embed_tokens
        self.positional_convol = nn.Sequential(
            Padding((0, 0, 0, max_block_size-1)),
            Rearrange('b s d -> b d s'),
            Depthwise1dConv(dim, dim, krnl_size=max_block_size),
            Rearrange('b d s -> b s d'))
        self.cand_scoring = nn.Sequential(
            nn.Linear(dim, 1),
            Rearrange('... () -> ...'))

    def get_blocks(self):
        """ return GBST candidate blocking list. """
        return self.blocks

    @torch.cuda.amp.autocast()
    def forward(self, in_tensor, attention_mask=None):
        b, s = in_tensor.shape
        mask = attention_mask
        #print(f"mask: {mask}")
        block_multi, ds_factor = self.block_pad_multiple, self.downsample_factor
        # get divisible length to pad
        m = math.ceil(s / ds_factor) * ds_factor

        in_tensor = self.embeds(in_tensor)
        in_tensor = self.positional_convol(in_tensor)
        in_tensor = pad_to_multiple(in_tensor, block_multi, seq_dim=1, dim=-2)

        if mask is not None:
            mask = pad_to_multiple(mask, block_multi, seq_dim=1, dim=-1, value=False)

        block_reprs, block_masks = [], []

        # 이제 입력 시퀀스를 cloning해서 후보를 세팅
        for block_size, offset in self.blocks:
            block_in = in_tensor.clone()
            if mask is not None:
                block_mask = mask.clone()
            need_padding = offset > 0

            if need_padding:
                loff, roff = (block_size - offset), offset
                #print(f"loff: {loff}, roff: {roff}")
                block_in = F.pad(block_in, (0, 0, loff, roff), value=0.0)
                if mask is not None:
                    block_mask = F.pad(block_mask, (0, 0, loff, roff), value=False)

            blks = rearrange(block_in, 'b (s m) d -> b s m d', m=block_size)
            if mask is not None:
                mask_blks = rearrange(block_mask, 'b (s m) -> b s m', m=block_size)
                blk_repr = masked_mean(blks, mask_blks, dim=-2)
            else:
                blk_repr = blks.mean(dim=-2)

            blk_repr = repeat(blk_repr, 'b s d -> b (s m) d', m=block_size)

            if need_padding:
                blk_repr = blk_repr[:, loff:-roff]

            block_reprs.append(blk_repr)

            if mask is not None:
                mask_blks = torch.any(mask_blks, dim=-1)
                mask_blks = repeat(mask_blks, 'b s -> b (s m)', m=block_size)
                if need_padding:
                    mask_blks = mask_blks[:, loff:-roff]
                block_masks.append(mask_blks)

        # stack them all
        block_reprs = torch.stack(block_reprs, dim=2)
        scores = self.cand_scoring(block_reprs)

        if mask is not None:
            block_masks = torch.stack(block_masks, dim=2)
            max_neg_val = -torch.finfo(scores.dtype).max
            scores = scores.masked_fill(~block_masks, max_neg_val)

        scores = scores.softmax(dim=2)

        # cheap consensus attention, as equation (5) in paper.
        if self.score_consensus_attn:
            score_sim = einsum('b i d, b j d -> b i j', scores, scores)

            if mask is not None:
                cross_mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
                max_neg_val = -torch.finfo(score_sim.dtype).max
                score_sim = score_sim.masked_fill((~(cross_mask.bool())), max_neg_val)

            score_attn = score_sim.softmax(dim=-1)
            scores = einsum('b i j, b j m -> b i m', score_attn, scores)

        scores = rearrange(scores, 'b n m -> b n m ()')
        in_tensor = (block_reprs * scores).sum(dim=2)

        in_tensor = in_tensor[:, :m]
        if mask is not None:
            mask = mask[:, :m]

        # downsample with mean pooling
        in_tensor = rearrange(in_tensor, 'b (n m) d -> b n m d', m=ds_factor)
        if mask is not None:
            mask = rearrange(mask, 'b (n m) -> b n m', m=ds_factor)
            in_tensor = masked_mean(in_tensor, mask, dim=2)
            mask = torch.any(mask, dim=-1)
        else:
            in_tensor = in_tensor.mean(dim=-2)

        # tuple을 반환하기 때문에, forward()에서 [0]을 취해 바꿔줘야 한다
        return in_tensor, mask
