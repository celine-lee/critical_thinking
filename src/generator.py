import re
import os

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

from openai import OpenAI
import tiktoken

# openai.api_key = os.getenv("OPENAI_API_KEY")  


class Generator:

    def __init__(self, model_name, sampling_args, force_gen_args):
        self.model_name = model_name
        self.sampling_args = sampling_args
        self.force_gen_args = force_gen_args

    def generate(self, input_prompts):
        raise NotImplementedError

class HFGenerator(Generator):
    def __init__(self, model_name, gen_kwargs, max_batch_size):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", truncation_side="left"
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        sampling_args = {
            "return_dict_in_generate": True,
            "max_new_tokens": gen_kwargs["max_new_tokens"],
            "no_repeat_ngram_size": 0, 
            "num_beams": gen_kwargs["num_beams"],
            "tokenizer": self.tokenizer,
            "stop_strings": gen_kwargs["stop_strings"],
            "num_return_sequences": gen_kwargs["num_return_sequences"],
            "do_sample": gen_kwargs["temperature"] > 0.,
            "temperature": gen_kwargs["temperature"] if gen_kwargs["temperature"] > 0. else None,
            "pad_token_id": self.tokenizer.eos_token_id,
            "top_k": None,
            "top_p": 0.9 if gen_kwargs["temperature"] > 0. else None,
        }
        force_gen_args = {
            "return_dict_in_generate": True,
            "max_new_tokens": 50,
            "tokenizer": self.tokenizer,
            "stop_strings": gen_kwargs["stop_strings"],
            "num_return_sequences": 1,
            "do_sample": False,
            "temperature": None,
            "top_k": None,
            "top_p": None,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        super().__init__(model_name, sampling_args, force_gen_args)
        self.model = self.load_model(model_name)

        self.max_batch_size = max_batch_size


    def load_model(self, model_name, quantize=True):
        config = AutoConfig.from_pretrained(model_name)

        bnb_config = None
        if quantize and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
            )

        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map="auto",
        )
        model.eval()

        for param in model.parameters():
            param._requires_grad = False

        return model

    def generate(self, input_prompts, task):
        disable_cot = input_prompts[-1].endswith(task.reprompt_string)
        input_ids = self.tokenizer(
            input_prompts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
            return_offsets_mapping=True,
        ).to(device)
        
        model_output = self.model.generate(
            input_ids=input_ids.input_ids,
            attention_mask=input_ids.attention_mask,
            **self.sampling_args,
        )
        new_queries = []
        gens_need_augmenting = []
        full_generations = [None for _ in model_output.sequences]
        extracted_answers = [None for _ in model_output.sequences]
        generation_lengths = [None for _ in model_output.sequences]
        for input_idx, generation in enumerate(model_output.sequences):
            
            model_prediction = self.tokenizer.decode(generation, skip_special_tokens=True)
            just_generation = generation[input_ids.input_ids.shape[-1] :]
            generation_lengths[input_idx] = torch.sum(
                just_generation
                != self.tokenizer.pad_token_id
            ).item()
            full_generations[input_idx] = self.tokenizer.decode(just_generation, skip_special_tokens=True)
            parsed_answer = None
            for parsed_answer in re.finditer(task.answer_extraction_regex, model_prediction):
                pass # only get the last
            if parsed_answer is None or ((not disable_cot) and parsed_answer.start() < len(input_prompts[0])):
                gens_need_augmenting.append(input_idx)
                new_queries.append(input_prompts[0] +full_generations[input_idx] + task.reprompt_string)
            else:
                extracted_answers[input_idx] = parsed_answer.group(1).rstrip(" .")
        
        if len(new_queries) > 0:
            new_input_ids = self.tokenizer(
                new_queries,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            ).to(device)

            model_output = self.model.generate(
                input_ids=new_input_ids.input_ids,
                attention_mask=new_input_ids.attention_mask,
                **self.force_gen_args,
            )
            for new_idx, orig_idx in enumerate(gens_need_augmenting):
                generation = model_output.sequences[new_idx]
                model_prediction = self.tokenizer.decode(generation, skip_special_tokens=True)
                just_generation = generation[new_input_ids.input_ids.shape[-1] :]
                parsed_answer = None
                for parsed_answer in re.finditer(task.answer_extraction_regex, model_prediction):
                    pass # only get the last
                if parsed_answer is not None:
                    extracted_answers[orig_idx] = parsed_answer.group(1).rstrip(" .")
                full_generations[orig_idx] += task.reprompt_string + self.tokenizer.decode(just_generation, skip_special_tokens=True)
                num_generated_tokens = torch.sum(
                    just_generation
                    != self.tokenizer.pad_token_id
                ).item()
                generation_lengths[orig_idx] += num_generated_tokens # doesnt account for extra from reprompt but..

        return full_generations, generation_lengths, extracted_answers


class OpenAIGenerator(Generator):
    def __init__(self, model_name, gen_kwargs, max_batch_size):
        self.model_name = model_name
        self.enc = tiktoken.encoding_for_model(model_name)
        self.max_batch_size = max_batch_size
        self.sampling_args = {
            "temperature": gen_kwargs["temperature"],
            "max_completion_tokens": gen_kwargs["max_new_tokens"],
            "n": gen_kwargs["num_return_sequences"],
            "top_p": 0.9 if gen_kwargs["temperature"] > 0 else None,
            "stop": gen_kwargs["stop_strings"],
        }
        self.force_gen_args = {
            "temperature": 0.0,
            "max_completion_tokens": 50,
            "n": 1,
            "top_p": None,
            "stop": gen_kwargs["stop_strings"],
        }
        super().__init__(model_name, self.sampling_args, self.force_gen_args)
        self.client = OpenAI()
        if 'o3' in model_name: 
            # temperature and top_p top_n not supported in reasoning models
            del self.sampling_args["temperature"]
            del self.force_gen_args["temperature"]
            del self.sampling_args["top_p"]
            del self.force_gen_args["top_p"]

    def generate(self, input_prompts, task):
        """
        Generates text completions from OpenAI's API.

        Args:
            input_prompts (list of str): List of input prompts.
            task (object): Task object containing regex for answer extraction.

        Returns:
            tuple: (full_generations, generation_lengths, extracted_answers)
        """
        full_generations = [None] * len(input_prompts)
        extracted_answers = [None] * len(input_prompts)
        generation_lengths = [None] * len(input_prompts)

        responses = []
        for idx, prompt in enumerate(input_prompts):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **self.sampling_args
            )
            responses.append(response)
        for i, response in enumerate(responses):
            model_prediction = response.choices[0].message.content
            full_generations[i] = model_prediction
            generation_lengths[i] = response.usage.completion_tokens

            parsed_answer = None
            for match in re.finditer(task.answer_extraction_regex, model_prediction):
                parsed_answer = match  # Get the last match

            if parsed_answer is None:
                gens_need_augmenting.append(i)
                new_queries.append(input_prompts[i] + full_generations[i] + task.reprompt_string)
            else:
                extracted_answers[i] = parsed_answer.group(1).rstrip(" .")

        return full_generations, generation_lengths, extracted_answers

class VLLMGenerator(Generator):
    def __init__(self, model_name, gen_kwargs, max_batch_size):
        raise NotImplementedError