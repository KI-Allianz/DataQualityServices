#Copyright 2024 Mücahit Sahin
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_model(model: str = "", quantization: str = "4bit"):
    """
    Load a causal language model and tokenizer from Hugging Face and wrap them
    in a text-generation pipeline.

    Quantization behavior:
        - "8bit": Use bitsandbytes 8-bit quantization when CUDA is available.
        - "4bit": Use bitsandbytes 4-bit quantization when CUDA is available.
        - "16bit": Load with float16 weights when CUDA is available.
        - "": Load without quantization.

    CPU fallback:
        On systems without a CUDA GPU, such as a Mac Docker container,
        bitsandbytes quantization is disabled automatically and the model is
        loaded on the CPU using float32 weights.

    Args:
        model: Hugging Face model ID or local path.
        quantization: One of {"8bit", "4bit", "16bit", ""}.

    Returns:
        transformers.Pipeline: Text-generation pipeline.
    """
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
        )
    except ImportError as exc:
        raise ImportError(
            "Failed to import the required model libraries. Ensure that "
            "torch, transformers and huggingface_hub are installed."
        ) from exc

    valid_quantization_modes = {"8bit", "4bit", "16bit", ""}

    if quantization not in valid_quantization_modes:
        raise ValueError(
            f"Unsupported quantization mode: {quantization!r}. "
            f"Choose one of {sorted(valid_quantization_modes)!r}."
        )

    cuda_available = torch.cuda.is_available()

    logger.info("Loading model: %s", model)
    logger.info("CUDA available: %s", cuda_available)
    logger.info("Requested quantization mode: %s", quantization or "none")

    tokenizer = AutoTokenizer.from_pretrained(model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def _load_model(
        device_map: str,
        selected_quantization: str,
    ):
        model_kwargs = {
            "device_map": device_map,
        }

        if selected_quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        elif selected_quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        elif selected_quantization == "16bit":
            model_kwargs["torch_dtype"] = torch.float16

        else:
            # CPU loading should use float32 for broad compatibility.
            model_kwargs["torch_dtype"] = torch.float32

        return AutoModelForCausalLM.from_pretrained(
            model,
            **model_kwargs,
        )

    if cuda_available:
        device_map = "auto"
        effective_quantization = quantization
    else:
        device_map = "cpu"

        if quantization in {"4bit", "8bit"}:
            logger.warning(
                "CUDA is unavailable. Disabling bitsandbytes %s quantization "
                "and loading the model on the CPU with float32 weights.",
                quantization,
            )

        elif quantization == "16bit":
            logger.warning(
                "CUDA is unavailable. Disabling float16 loading and loading "
                "the model on the CPU with float32 weights."
            )

        effective_quantization = ""

    try:
        llm_model = _load_model(
            device_map=device_map,
            selected_quantization=effective_quantization,
        )

    except RuntimeError as exc:
        if cuda_available and "out of memory" in str(exc).lower():
            logger.warning(
                "CUDA ran out of memory while loading the model. "
                "Retrying on the CPU without quantization."
            )

            torch.cuda.empty_cache()

            llm_model = _load_model(
                device_map="cpu",
                selected_quantization="",
            )
        else:
            raise

    return pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=tokenizer,
    )


# def load_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", quantization=""):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token_id = tokenizer.eos_token_id  # disable terminal warning

#     model_kwargs = {"device_map": "auto"}  # always applied

#     if quantization == "8bit":
#         model_kwargs["quantization_config"] = BitsAndBytesConfig(
#             load_in_8bit=True
#         )
#     elif quantization == "4bit":
#         model_kwargs["quantization_config"] = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.bfloat16
#         )
#     elif quantization == "16bit":
#         model_kwargs["torch_dtype"] = torch.float16
#     elif quantization != "":
#         raise ValueError(f"Unsupported quantization option: {quantization}")

#     model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#     )
#     return pipe

# def load_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", quantization=""):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token_id = tokenizer.eos_token_id  # disable terminal warning

#     model_kwargs = {"device_map": "auto"}  # always applied

#     if quantization == "8bit":
#         model_kwargs["quantization_config"] = BitsAndBytesConfig(
#             load_in_8bit=True
#         )
#     elif quantization == "4bit":
#         model_kwargs["quantization_config"] = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.bfloat16
#         )
#     elif quantization == "16bit":
#         model_kwargs["torch_dtype"] = torch.float16
#     elif quantization != "":
#         raise ValueError(f"Unsupported quantization option: {quantization}")

#     model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#     )
#     return pipe





def ask_model_mes(messages, pipe=None, max_new_tokens=10000):
    """
    Call a chat-style text-generation pipeline with an already-built list of messages.

    Args:
        messages (list[dict]): Chat messages in the format expected by your pipeline, e.g.:
            [
              {"role": "system", "content": "..."},
              {"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."},
              ...
            ]
        pipe: A Hugging Face `pipeline("text-generation", ...)` (or compatible) object
              that supports chat-format input.
        max_new_tokens (int): Maximum number of tokens the model may generate.

    Returns:
        str: The assistant's textual content from the last generated chat turn.

    Notes:
        - This assumes the pipeline returns a list with one item, whose structure is:
          outputs[0]["generated_text"]  -> a list of chat turns (dicts),
          and the final generated turn is at index -1 with key 'content'.
    """
    # Guard: model pipeline must be provided
    if pipe is None:
        return "Model was not loaded! Use load_model() before running this function."

    # Invoke the model with deterministic generation for structured outputs.
    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        return_full_text=False,
    )

    # Extract the last generated message's content from the pipeline output
    generated_text = outputs[0]["generated_text"]
    if isinstance(generated_text, list):
        response = generated_text[-1]["content"]
    elif isinstance(generated_text, dict):
        response = generated_text.get("content", "")
    else:
        response = generated_text
    return response


def ask_model_prompt(prompt, system_prompt=None, pipe=None):
    """
    Build a chat message list from a raw user prompt (and optional system prompt),
    then call a chat-style text-generation pipeline.

    Args:
        prompt (str): The user instruction/question.
        system_prompt (str | None): Optional system message to steer behavior.
        pipe: A Hugging Face `pipeline("text-generation", ...)` (or compatible) object.

    Returns:
        str: The assistant's textual content from the last generated chat turn.
             Returns empty string if generation completely fails.

    Generation params:
        - max_new_tokens=512 (reduced for stability)
        - temperature=0.7
        - top_p=0.9
        - top_k=50
        - do_sample=True
        - repetition_penalty=1.1
    """
    # Guard: model pipeline must be provided
    if pipe is None:
        return "Model was not loaded! Use load_model() before running this function."
    else:
        # Construct chat messages depending on whether a system prompt is provided
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "user", "content": prompt},
            ]

        # Call the model with stable generation parameters
        try:
            outputs = pipe(
                messages,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.1
            )
        except RuntimeError as e:
            if "probability tensor" in str(e) or "inf" in str(e) or "nan" in str(e):
                logger.warning(f"Model generation failed with probability tensor error: {e}. Retrying with deterministic generation...")
                try:
                    outputs = pipe(
                        messages,
                        max_new_tokens=128,
                        temperature=1.0,
                        do_sample=False
                    )
                except Exception as retry_error:
                    logger.error(f"Deterministic generation also failed: {retry_error}")
                    # Return empty string instead of crashing
                    return ""
            else:
                logger.error(f"Model generation failed: {e}")
                return ""
        except Exception as e:
            logger.error(f"Unexpected error during model generation: {e}")
            return ""

        try:
            # Extract the assistant's final generated message content
            response = outputs[0]["generated_text"][-1]['content']
            return response if response else ""
        except (IndexError, KeyError, TypeError) as e:
            logger.error(f"Failed to extract response from model output: {e}")
            return ""
