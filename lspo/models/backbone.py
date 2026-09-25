from __future__ import annotations
import copy
from typing import List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
from ..utils.device import autocast, best_attention_implementation, resolve_dtype
from .tiny_lm import ByteTokenizer, LMOutput, build_tiny_lm
PROMPT_TEMPLATE = 'Question:\n{question}\n\nShow your step-by-step reasoning, then give the final answer on its own line as "Answer: <answer>".'
REVISE_TEMPLATE = 'Question:\n{question}\n\nCurrent answer:\n{answer}\n\nRewrite the answer, fixing the specific step that is wrong while keeping every other part of the reasoning unchanged. Output the full corrected answer and end with "Answer: <answer>".'
SWITCH_TEMPLATE = 'Question:\n{question}\n\nAnswer the question from scratch using a different line of reasoning than before. End with "Answer: <answer>".'
REFLECT_TEMPLATE = 'Question:\n{question}\n\nPrevious attempt:\n{answer}\n\nCritique the previous attempt: identify the first incorrect step and explain why it is wrong. Then write a corrected answer ending with "Answer: <answer>".'

def format_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)

class BackboneLM(nn.Module):
    d_model: int = 0
    pad_token_id: int = 0
    eos_token_id: Optional[int] = None
    compute_dtype: Optional[torch.dtype] = None
    amp_enabled: bool = True

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward_ids(self, input_ids, attention_mask=None, output_hidden_states: bool=False) -> LMOutput:
        raise NotImplementedError

    def context(self):
        device = next(self.parameters()).device
        return autocast(device, self.compute_dtype, enabled=self.amp_enabled)

    def pooled_tensors(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, pooling: str='mean_last') -> torch.Tensor:
        with self.context():
            out = self.forward_ids(input_ids, attention_mask, output_hidden_states=True)
        hidden = out.hidden_states
        if attention_mask is None:
            return hidden.mean(dim=1) if pooling != 'last' else hidden[:, -1, :]
        mask = attention_mask[:, :hidden.shape[1]].unsqueeze(-1).to(hidden.dtype)
        if pooling == 'last':
            lengths = mask.squeeze(-1).sum(dim=-1).clamp(min=1).long() - 1
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def pooled(self, prompt: str, answer: str, pooling: str='mean_last') -> torch.Tensor:
        ids, mask = self.encode(prompt + '\n' + answer)
        return self.pooled_tensors(ids, mask, pooling=pooling)[0]

    def encode_batch(self, texts: Sequence[str], padding_side: str='right', max_length: Optional[int]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encode(text) for text in texts]
        lengths = [int(ids.shape[1]) for ids, _ in encoded]
        width = max(lengths) if lengths else 1
        if max_length is not None:
            width = min(width, max_length)
        pad_id = int(self.pad_token_id)
        input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(encoded), width), dtype=torch.long)
        for row, (ids, mask) in enumerate(encoded):
            length = min(int(ids.shape[1]), width)
            if padding_side == 'left':
                input_ids[row, width - length:] = ids[0, :length]
                attention[row, width - length:] = mask[0, :length]
            else:
                input_ids[row, :length] = ids[0, :length]
                attention[row, :length] = mask[0, :length]
        return (input_ids, attention)

    def pooled_texts_batch(self, prompts: Sequence[str], answers: Sequence[str], pooling: str='mean_last') -> torch.Tensor:
        texts = [f'{prompt}\n{answer}' for prompt, answer in zip(prompts, answers)]
        input_ids, attention = self.encode_batch(texts)
        input_ids = input_ids.to(self._device())
        attention = attention.to(self._device())
        return self.pooled_tensors(input_ids, attention, pooling=pooling)

    def _device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def generate_ids(self, prompt_ids: torch.Tensor, max_new_tokens: int, temperature: float, top_p: float) -> torch.Tensor:
        raise NotImplementedError

    def generate_ids_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int, temperature: float, top_p: float) -> torch.Tensor:
        raise NotImplementedError

    def generate(self, prompt: str, max_new_tokens: int, temperature: float=0.6, top_p: float=0.95) -> str:
        ids, _ = self.encode(prompt)
        ids = ids.to(self._device())
        generated = self.generate_ids(ids, max_new_tokens, temperature, top_p)
        return self.decode_ids(generated[0, ids.shape[1]:])

    def generate_batch(self, prompts: Sequence[str], max_new_tokens: int, temperature: float=0.6, top_p: float=0.95) -> List[str]:
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.generate(prompts[0], max_new_tokens, temperature, top_p)]
        input_ids, attention = self.encode_batch(list(prompts), padding_side='left')
        input_ids = input_ids.to(self._device())
        attention = attention.to(self._device())
        prompt_length = input_ids.shape[1]
        with torch.no_grad():
            generated = self.generate_ids_batch(input_ids, attention, max_new_tokens, temperature, top_p)
        outputs: List[str] = []
        for row in range(generated.shape[0]):
            text = self.decode_ids(generated[row, prompt_length:])
            outputs.append(text)
        return outputs

    def decode_ids(self, ids: torch.Tensor) -> str:
        raise NotImplementedError

    def logprob_answer(self, prompt: str, answer: str) -> torch.Tensor:
        prompt_ids, _ = self.encode(prompt)
        prompt_ids = prompt_ids.to(self._device())
        return self.logprob_answer_ids(prompt_ids, answer)

    def logprob_answer_ids(self, prompt_ids: torch.Tensor, answer: str) -> torch.Tensor:
        answer_ids, _ = self.encode(answer)
        answer_ids = answer_ids.to(prompt_ids.device)
        return self.sequence_logprob(prompt_ids, answer_ids)

    def sequence_logprob(self, context_ids: torch.Tensor, continuation_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def token_count(self, text: str) -> int:
        ids, _ = self.encode(text)
        return int(ids.shape[1])

class TinyBackbone(BackboneLM):

    def __init__(self, d_model: int=128, n_layer: int=2, n_head: int=4, ctx: int=512, dtype: str='auto', device: str='auto'):
        super().__init__()
        from ..utils.device import resolve_device
        self.tokenizer = ByteTokenizer()
        self.model = build_tiny_lm(self.tokenizer.vocab_size, d_model, n_layer, n_head, ctx)
        self.d_model = d_model
        self.ctx = ctx
        self.pad_token_id = ByteTokenizer.pad_token_id
        self.eos_token_id = ByteTokenizer.eos_token_id
        resolved = resolve_device(device)
        self.compute_dtype = resolve_dtype(dtype, resolved)
        if resolved.type == 'cpu':
            self.compute_dtype = torch.float32
        self.model.to(device=resolved, dtype=torch.float32)
        self.amp_enabled = resolved.type != 'cpu'

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.tokenizer.encode(text)[:self.ctx]
        tensor = torch.tensor([ids], dtype=torch.long)
        mask = torch.ones_like(tensor)
        return (tensor, mask)

    def forward_ids(self, input_ids, attention_mask=None, output_hidden_states: bool=True) -> LMOutput:
        return self.model(input_ids, attention_mask, output_hidden_states=True)

    def generate_ids(self, prompt_ids, max_new_tokens, temperature, top_p):
        with torch.no_grad():
            return self.model.generate_ids(prompt_ids, max_new_tokens, temperature, top_p, eos_token_id=None, pad_token_id=self.pad_token_id)

    def generate_ids_batch(self, input_ids, attention_mask, max_new_tokens, temperature, top_p):
        return self.model.generate_batch_ids(input_ids, attention_mask, max_new_tokens, temperature, top_p, eos_token_id=None, pad_token_id=self.pad_token_id)

    def decode_ids(self, ids: torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids)

    def sequence_logprob(self, context_ids, continuation_ids) -> torch.Tensor:
        return self.model.sequence_logprob(context_ids, continuation_ids)

class HFBackbone(BackboneLM):

    def __init__(self, repo_id: str, tokenizer_id: Optional[str]=None, dtype: str='auto', device: str='auto', trust_remote_code: bool=False, attn_implementation: str='auto', use_cache: bool=True):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers is required for HF backbones; install it or set model.backbone='tiny' for the dependency-free path.") from exc
        from ..utils.device import resolve_device
        resolved = resolve_device(device)
        torch_dtype = resolve_dtype(dtype, resolved)
        attention = best_attention_implementation(resolved, attn_implementation)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or repo_id, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        load_kwargs = {'torch_dtype': torch_dtype, 'trust_remote_code': trust_remote_code, 'attn_implementation': attention}
        if resolved.type == 'cuda' and device == 'auto':
            load_kwargs['device_map'] = 'auto'
        self.model = AutoModelForCausalLM.from_pretrained(repo_id, **load_kwargs)
        if 'device_map' not in load_kwargs and resolved.type != 'cpu':
            self.model.to(resolved)
        self.model.config.use_cache = use_cache
        self.d_model = self.model.config.hidden_size
        self.ctx = getattr(self.model.config, 'max_position_embeddings', 4096)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.compute_dtype = torch_dtype
        self.amp_enabled = resolved.type == 'cuda'
        self.attn_implementation = attention

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.ctx)
        return (encoded['input_ids'], encoded['attention_mask'])

    def encode_batch(self, texts: Sequence[str], padding_side: str='right', max_length: Optional[int]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        previous = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        try:
            encoded = self.tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True, max_length=max_length or self.ctx)
        finally:
            self.tokenizer.padding_side = previous
        return (encoded['input_ids'], encoded['attention_mask'])

    def forward_ids(self, input_ids, attention_mask=None, output_hidden_states: bool=True) -> LMOutput:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return LMOutput(logits=out.logits, hidden_states=out.hidden_states[-1])

    def generate_ids(self, prompt_ids, max_new_tokens, temperature, top_p):
        with torch.no_grad():
            return self.model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-05), top_p=top_p, pad_token_id=self.tokenizer.pad_token_id)

    def generate_ids_batch(self, input_ids, attention_mask, max_new_tokens, temperature, top_p):
        with torch.no_grad():
            return self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-05), top_p=top_p, pad_token_id=self.tokenizer.pad_token_id)

    def decode_ids(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def sequence_logprob(self, context_ids, continuation_ids) -> torch.Tensor:
        import torch.nn.functional as F
        full = torch.cat([context_ids, continuation_ids.to(context_ids.device)], dim=-1)
        full = full[:, -self.ctx:]
        with self.context():
            logits = self.forward_ids(full).logits
        log_probs = F.log_softmax(logits.float(), dim=-1)
        targets = full[:, 1:]
        gathered = log_probs[:, :-1, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        n_context = full.shape[1] - continuation_ids.shape[1]
        mask = torch.zeros_like(gathered)
        mask[:, max(0, n_context - 1):] = 1.0
        return (gathered * mask).sum(dim=-1)

def build_backbone(config) -> BackboneLM:
    if config.backbone == 'tiny':
        return TinyBackbone(d_model=config.tiny_d_model, n_layer=config.tiny_n_layer, n_head=config.tiny_n_head, ctx=config.tiny_ctx, dtype=config.dtype, device=config.device)
    return HFBackbone(repo_id=config.backbone, tokenizer_id=config.tokenizer, dtype=config.dtype, device=config.device, trust_remote_code=config.trust_remote_code, attn_implementation=config.attn_implementation)

class FrozenScoringEncoder(nn.Module):

    def __init__(self, backbone: BackboneLM, copy_weights: bool=True):
        super().__init__()
        self.backbone = copy.deepcopy(backbone) if copy_weights else backbone
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.d_model = backbone.d_model
        self.pad_token_id = backbone.pad_token_id
        self.compute_dtype = getattr(backbone, 'compute_dtype', None)
        self.amp_enabled = getattr(backbone, 'amp_enabled', False)

    @property
    def _inner(self) -> BackboneLM:
        return self.backbone

    def context(self):
        return self._inner.context()

    @torch.no_grad()
    def pooled(self, prompt: str, answer: str, pooling: str='mean_last') -> torch.Tensor:
        return self._inner.pooled(prompt, answer, pooling=pooling)

    @torch.no_grad()
    def pooled_texts_batch(self, prompts: Sequence[str], answers: Sequence[str], pooling: str='mean_last') -> torch.Tensor:
        return self._inner.pooled_texts_batch(prompts, answers, pooling=pooling)

    @torch.no_grad()
    def pooled_batch(self, pairs: Sequence[Tuple[str, str]], pooling: str='mean_last'):
        return torch.stack([self.pooled(q, x, pooling=pooling) for q, x in pairs])

    def train(self, mode: bool=True):
        super().train(False)
        return self

def build_frozen_encoder(config, generator: BackboneLM) -> FrozenScoringEncoder:
    mode = getattr(config.model, 'scoring_encoder_init', 'checkpoint')
    if mode == 'shared':
        raise ValueError('A fixed scoring/reference encoder must not share generator parameters')
    if mode == 'checkpoint' and config.model.backbone != 'tiny':
        scorer_backbone = build_backbone(config.model)
        return FrozenScoringEncoder(scorer_backbone, copy_weights=False)
    if mode == 'checkpoint':
        return FrozenScoringEncoder(generator, copy_weights=True)
    return FrozenScoringEncoder(generator, copy_weights=True)
