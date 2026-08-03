from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any

from edgeflow.kernels.rmsnorm.dispatch import dispatch_decision, fused_residual_rmsnorm


@dataclass
class LlamaRMSNormIntegration:
    model: Any
    force_reference: bool = False
    counters: dict[str, int] = field(
        default_factory=lambda: {
            "calls": 0,
            "triton_calls": 0,
            "fallback_calls": 0,
        }
    )
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    _originals: list[tuple[Any, Any]] = field(default_factory=list)

    def enable(self) -> LlamaRMSNormIntegration:
        if self._originals:
            return self
        base = getattr(self.model, "model", None)
        layers = getattr(base, "layers", None)
        if layers is None:
            raise ValueError("expected a Hugging Face LlamaForCausalLM model with model.layers")
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        if model_type != "llama":
            raise ValueError(f"end-to-end RMSNorm integration supports llama, got {model_type!r}")
        for layer in layers:
            required = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
            if not all(hasattr(layer, name) for name in required):
                raise ValueError("decoder layer does not satisfy the registered Llama contract")
            original = layer.forward
            self._originals.append((layer, original))
            def forward(
                decoder_layer: Any,
                hidden_states: Any,
                attention_mask: Any = None,
                position_ids: Any = None,
                past_key_values: Any = None,
                use_cache: bool | None = False,
                position_embeddings: Any = None,
                _edgeflow_integration: LlamaRMSNormIntegration = self,
                **kwargs: Any,
            ) -> Any:
                residual = hidden_states
                hidden_states = decoder_layer.input_layernorm(hidden_states)
                attention_output, _ = decoder_layer.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                combined = residual + attention_output
                flat_attention = attention_output.reshape(-1, attention_output.shape[-1]).contiguous()
                flat_residual = residual.reshape(-1, residual.shape[-1]).contiguous()
                weight = decoder_layer.post_attention_layernorm.weight.contiguous()
                decision = dispatch_decision(flat_attention, flat_residual, weight)
                _edgeflow_integration.counters["calls"] += 1
                if decision["use_triton"] and not _edgeflow_integration.force_reference:
                    _edgeflow_integration.counters["triton_calls"] += 1
                else:
                    _edgeflow_integration.counters["fallback_calls"] += 1
                    reason = (
                        "forced_reference"
                        if _edgeflow_integration.force_reference
                        else str(decision["reason"])
                    )
                    _edgeflow_integration.fallback_reasons[reason] = (
                        _edgeflow_integration.fallback_reasons.get(reason, 0) + 1
                    )
                normalized = fused_residual_rmsnorm(
                    flat_attention,
                    flat_residual,
                    weight,
                    float(decoder_layer.post_attention_layernorm.variance_epsilon),
                    force_reference=_edgeflow_integration.force_reference,
                ).view_as(attention_output)
                hidden_states = decoder_layer.mlp(normalized)
                return combined + hidden_states

            layer.forward = types.MethodType(forward, layer)
        return self

    def disable(self) -> None:
        for layer, original in self._originals:
            layer.forward = original
        self._originals.clear()

    def summary(self) -> dict[str, Any]:
        return {
            **self.counters,
            "fallback_reasons": dict(sorted(self.fallback_reasons.items())),
            "patched_layer_count": len(self._originals),
            "force_reference": self.force_reference,
            "rollback_available": True,
        }
