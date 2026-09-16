# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen VLM-based progress reward model for world-model rollouts."""

from __future__ import annotations

import base64
import io
import json
import re
import urllib.request
from typing import Any, Optional

import torch
from omegaconf import DictConfig
from PIL import Image

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(dtype_name.lower(), torch.bfloat16)


class QwenVLMProgressRewardModel(BaseRewardModel):
    """Use a VLM response as task progress/confidence reward."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.inference_backend = str(cfg.get("inference_backend", "local_hf")).lower()

        self.dtype = _resolve_dtype(cfg.get("dtype", "bfloat16"))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 64))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.response_key = cfg.get("response_key", "score")
        self.default_score = float(cfg.get("default_score", 0.0))
        self.request_timeout = float(cfg.get("request_timeout", 30.0))
        self.max_retries = int(cfg.get("max_retries", 2))
        self.print_vlm_response = bool(cfg.get("print_vlm_response", False))
        self.force_json_response = bool(cfg.get("force_json_response", True))
        self._log_chunk_size = int(cfg.get("log_chunk_size", 0))

        self.prompt_template = cfg.get(
            "prompt_template",
            (
                "You are a strict robotic task progress evaluator.\n"
                "Task: {task}\n"
                "Look at the current image and estimate task completion progress/confidence.\n"
                "Return ONLY one-line compact JSON with this exact schema: "
                '{"score": <float in [0,1]>, "reason": "<=20 words>"}.\n'
                "No markdown, no code block, no extra text."
            ),
        )

        if self.inference_backend == "local_hf":
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self.model_path = cfg.get("model_path", "")
            if not self.model_path:
                raise ValueError(
                    "QwenVLMProgressRewardModel requires `model_path` "
                    "when `inference_backend=local_hf`."
                )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.vlm = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self.openai_api_base = ""
            self.openai_api_key = ""
            self.service_model_name = ""
        elif self.inference_backend in {
            "openai_api",
            "vllm_service",
            "sglang_service",
            "openai_compatible_service",
        }:
            self.openai_api_base = str(
                cfg.get("openai_api_base", "http://127.0.0.1:8000/v1")
            ).rstrip("/")
            self.openai_api_key = str(cfg.get("openai_api_key", "EMPTY"))
            self.service_model_name = str(cfg.get("service_model_name", ""))
            if not self.service_model_name:
                raise ValueError(
                    "QwenVLMProgressRewardModel requires `service_model_name` "
                    "when using remote service backend."
                )
            self.model_path = ""
            self.processor = None
            self.vlm = None
        else:
            raise ValueError(
                "Unsupported `inference_backend` for QwenVLMProgressRewardModel: "
                f"{self.inference_backend}. Supported: local_hf, openai_api, "
                "vllm_service, sglang_service."
            )

    def _build_prompt(self, task_description: str) -> str:
        task = task_description.strip() if task_description else "Unknown task"
        return self.prompt_template.replace("{task}", task)

    def _to_pil_images(self, images: torch.Tensor) -> list[Image.Image]:
        if images.dim() != 4:
            raise ValueError(f"Expected images with rank 4, got shape {images.shape}")

        if images.shape[1] in [1, 3, 4]:
            images = images.permute(0, 2, 3, 1)

        if images.dtype == torch.uint8:
            arr = images.detach().cpu().numpy()
        else:
            img = images.float().detach().cpu()
            if img.min() < 0:
                img = (img + 1.0) / 2.0
            if img.max() <= 1.0:
                img = img * 255.0
            arr = img.clamp(0, 255).to(torch.uint8).numpy()

        return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]

    def _log_vlm_responses(self, responses: list[str]) -> None:
        chunk_size = self._log_chunk_size
        for idx, text in enumerate(responses):
            if chunk_size > 0:
                env_idx, step_idx = divmod(idx, chunk_size)
                prefix = f"[VLM] env={env_idx} step={step_idx} "
            else:
                prefix = f"[VLM] i={idx}/{len(responses)} "
            print(f"{prefix}reply={' '.join(text.splitlines())!r}")

    def _parse_score(self, text: str) -> float:
        try:
            payload = json.loads(text.strip())
            if isinstance(payload, dict) and self.response_key in payload:
                return float(max(0.0, min(1.0, float(payload[self.response_key]))))
        except Exception:
            pass

        json_obj_match = re.search(r"\{[\s\S]*\}", text)
        if json_obj_match is not None:
            try:
                payload = json.loads(json_obj_match.group(0))
                if isinstance(payload, dict) and self.response_key in payload:
                    return float(max(0.0, min(1.0, float(payload[self.response_key]))))
            except Exception:
                pass

        pattern = rf'"?{re.escape(self.response_key)}"?\s*:\s*([0-9]*\.?[0-9]+)'
        matched = re.search(pattern, text, flags=re.IGNORECASE)
        if matched is None:
            matched = re.search(r"([0-9]*\.?[0-9]+)", text)
        if matched is not None:
            value = float(matched.group(1))
            if value > 1.0:
                value = value / 100.0
            return float(max(0.0, min(1.0, value)))

        return self.default_score

    def _encode_image_to_data_uri(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _request_service_response(self, prompt: str, image: Image.Image) -> str:
        payload = {
            "model": self.service_model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only JSON object with numeric field "
                        f"'{self.response_key}' in [0,1]."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._encode_image_to_data_uri(image)},
                        },
                    ],
                },
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        if self.force_json_response:
            payload["response_format"] = {"type": "json_object"}

        request = urllib.request.Request(
            url=f"{self.openai_api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}",
            },
            method="POST",
        )

        for _ in range(max(1, self.max_retries)):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.request_timeout
                ) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    choices = result.get("choices", [])
                    if not choices:
                        return ""
                    content = choices[0].get("message", {}).get("content", "")
                    if isinstance(content, list):
                        return "\n".join(
                            item.get("text", "")
                            for item in content
                            if isinstance(item, dict) and item.get("text")
                        )
                    return str(content)
            except Exception:
                continue
        return ""

    def forward(
        self,
        input_data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        rewards = self.compute_reward(input_data)
        return {
            "loss": torch.tensor(0.0, device=rewards.device),
            "accuracy": torch.tensor(0.0, device=rewards.device),
            "logits": rewards,
            "probabilities": rewards,
        }

    @torch.no_grad()
    def compute_reward(
        self,
        images: torch.Tensor,
        task_descriptions: Optional[list[str]] = None,
    ) -> torch.Tensor:
        if task_descriptions is None:
            task_descriptions = [""] * images.shape[0]
        if len(task_descriptions) != images.shape[0]:
            raise ValueError(
                "Length mismatch between task_descriptions and images: "
                f"{len(task_descriptions)} vs {images.shape[0]}"
            )

        pil_images = self._to_pil_images(images)
        prompts = [self._build_prompt(task) for task in task_descriptions]

        if self.inference_backend == "local_hf":
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                for prompt in prompts
            ]
            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in messages
            ]
            model_device = next(self.vlm.parameters()).device
            inputs = self.processor(
                text=texts,
                images=pil_images,
                return_tensors="pt",
                padding=True,
            ).to(model_device)

            generation_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.temperature > 1e-6,
            }
            if generation_kwargs["do_sample"]:
                generation_kwargs["temperature"] = self.temperature

            outputs = self.vlm.generate(**inputs, **generation_kwargs)
            prompt_len = inputs["input_ids"].shape[1]
            responses = self.processor.batch_decode(
                outputs[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            if self.print_vlm_response:
                self._log_vlm_responses(responses)
            scores = [self._parse_score(resp) for resp in responses]
            return torch.tensor(scores, dtype=torch.float32, device=model_device)

        responses = [
            self._request_service_response(prompt, image)
            for prompt, image in zip(prompts, pil_images, strict=True)
        ]
        if self.print_vlm_response:
            self._log_vlm_responses(responses)
        scores = [self._parse_score(resp) for resp in responses]
        return torch.tensor(scores, dtype=torch.float32, device=images.device)
