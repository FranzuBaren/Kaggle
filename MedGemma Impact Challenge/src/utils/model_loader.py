"""
Model Loader — MedGemma 1.5 4B with 4-bit NF4 Quantization.

Loads MedGemma with BitsAndBytes 4-bit quantization for deployment
on consumer GPUs (3 GB VRAM). Supports multimodal inference with
CXR image input via SigLIP encoder and logit-level introspection
for entropy-based uncertainty estimation.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Model identifiers
MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"
TXGEMMA_MODEL_ID = "google/txgemma-2b-predict"


class MedGemmaEngine:
    """Unified MedGemma inference engine with quantized loading.

    Provides generate() with optional entropy profiling for
    uncertainty visualization across generated tokens.
    """

    def __init__(
        self,
        model_id: str = MEDGEMMA_MODEL_ID,
        quantize: bool = True,
        max_memory: Optional[dict] = None,
        device: str = "auto",
    ):
        self.model_id = model_id
        self.quantize = quantize
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> None:
        """Load MedGemma with optional 4-bit quantization."""
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        logger.info(f"Loading {self.model_id} (quantize={self.quantize})")

        quant_config = None
        if self.quantize:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            quantization_config=quant_config,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )

        self._loaded = True
        logger.info(f"Model loaded: {self._get_memory_usage()}")

    def generate(
        self,
        prompt: str,
        image: Optional["PIL.Image.Image"] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        return_entropy: bool = False,
    ) -> tuple[str, list[float]]:
        """Run inference with optional entropy profiling.

        Args:
            prompt: Text prompt for the model.
            image: Optional PIL Image for multimodal input.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            return_entropy: Whether to compute per-token entropy.

        Returns:
            Tuple of (generated_text, entropy_profile).
        """
        if not self._loaded:
            self.load()

        # Build inputs
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False)
            inputs = self.processor(
                text=text, images=image, return_tensors="pt"
            ).to(self.model.device)
        else:
            messages = [{"role": "user", "content": prompt}]
            text = self.processor.apply_chat_template(messages, tokenize=False)
            inputs = self.processor(text=text, return_tensors="pt").to(
                self.model.device
            )

        # Generate with logits for entropy
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                return_dict_in_generate=True,
                output_scores=return_entropy,
            )

        # Decode response
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0][input_len:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)

        # Compute entropy profile
        entropy_profile = []
        if return_entropy and hasattr(outputs, "scores") and outputs.scores:
            for score in outputs.scores:
                probs = torch.softmax(score[0], dim=-1)
                entropy = -(probs * torch.log2(probs + 1e-10)).sum().item()
                entropy_profile.append(entropy)

        return response, entropy_profile

    def _get_memory_usage(self) -> str:
        """Report GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return f"GPU: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved"
        return "CPU mode"

    def unload(self) -> None:
        """Free model from memory."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded")
