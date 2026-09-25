from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
@dataclass
class LMOutput:
    logits: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None
class ByteTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    OFFSET = 3
    def __init__(self) -> None:
        self.vocab_size = self.OFFSET + 256
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [b + self.OFFSET for b in text.encode("utf-8", errors="replace")]
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids
    def decode(self, ids: Sequence[int]) -> str:
        payload = bytes(
            i - self.OFFSET
            for i in ids
            if i >= self.OFFSET and i - self.OFFSET < 256
        )
        return payload.decode("utf-8", errors="replace")
    def __call__(self, text: str, add_special_tokens: bool = False):
        ids = self.encode(text)
        return type("Encoding", (), {"input_ids": ids})()
class _Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x
class TinyCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layer: int = 2,
        n_head: int = 4,
        ctx: int = 512,
        dropout: float = 0.0,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.ctx = ctx
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(ctx, d_model)
        self.blocks = nn.ModuleList(
            [_Block(d_model, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((length, length), float("-inf"), device=device), diagonal=1
        )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> LMOutput:
        batch, length = input_ids.shape
        length = min(length, self.ctx)
        input_ids = input_ids[:, :length]
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = self._causal_mask(length, input_ids.device)
        if attention_mask is not None:
            pad = (attention_mask[:, :length] == 0)
            mask = mask.unsqueeze(0).expand(batch, length, length).clone()
            mask = mask.masked_fill(pad[:, None, :], float("-inf"))
        else:
            mask = mask.unsqueeze(0).expand(batch, length, length)
        mask = (
            mask.unsqueeze(1)
            .expand(batch, self.n_head, length, length)
            .reshape(batch * self.n_head, length, length)
        )
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        hidden = x if output_hidden_states else None
        logits = self.lm_head(x)
        return LMOutput(logits=logits, hidden_states=hidden)
    @torch.no_grad()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.6,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
        pad_token_id: int = 0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        eos_token_id = self.eos_token_id if eos_token_id is None else eos_token_id
        generated = input_ids
        for _ in range(max_new_tokens):
            window = generated[:, -self.ctx:]
            logits = self.forward(window).logits[:, -1, :]
            next_token = self._sample(logits, temperature, top_p, generator)
            generated = torch.cat([generated, next_token], dim=-1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        return generated
    @torch.no_grad()
    def generate_batch_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        eos_token_id: Optional[int] = None,
        pad_token_id: int = 0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        eos_token_id = self.eos_token_id if eos_token_id is None else eos_token_id
        generated = input_ids
        masks = attention_mask
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            window = generated[:, -self.ctx:]
            logits = self.forward(window, masks[:, -window.shape[1]:]).logits[:, -1, :]
            next_token = self._sample(logits, temperature, top_p, generator)
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, pad_token_id),
                    next_token,
                )
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
            generated = torch.cat([generated, next_token], dim=-1)
            masks = torch.cat([masks, torch.ones_like(next_token)], dim=-1)
            if eos_token_id is not None and bool(finished.all()):
                break
        del pad_token_id
        return generated
    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        return _sample_top_p(logits / temperature, top_p, generator)
    def sequence_logprob(
        self,
        context_ids: torch.Tensor,
        continuation_ids: torch.Tensor,
    ) -> torch.Tensor:
        full = torch.cat([context_ids, continuation_ids], dim=-1)
        full = full[:, -self.ctx:]
        logits = self.forward(full).logits
        log_probs = F.log_softmax(logits, dim=-1)
        targets = full[:, 1:]
        gathered = log_probs[:, :-1, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        n_context = full.shape[1] - continuation_ids.shape[1]
        mask = torch.zeros_like(gathered)
        mask[:, max(0, n_context - 1):] = 1.0
        return (gathered * mask).sum(dim=-1)
    @property
    def eos_token_id(self) -> int:
        return ByteTokenizer.eos_token_id
def _sample_top_p(
    logits: torch.Tensor,
    top_p: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    mask = cumulative - torch.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    probs = torch.softmax(sorted_logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1, generator=generator)
    return sorted_indices.gather(-1, sampled)
def build_tiny_lm(
    vocab_size: int,
    d_model: int = 128,
    n_layer: int = 2,
    n_head: int = 4,
    ctx: int = 512,
) -> TinyCausalLM:
    return TinyCausalLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
        ctx=ctx,
        pad_token_id=ByteTokenizer.pad_token_id,
    )
