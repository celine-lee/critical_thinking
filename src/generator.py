import re
import os
import gc

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator:
    def __init__(self, model_name, max_batch_size, sampling_args, force_gen_args):
        self.model_name = model_name
        self.sampling_args = sampling_args
        self.force_gen_args = force_gen_args
        self.max_batch_size = max_batch_size

    def update_sampling_args_bounds(self, low_bound, high_bound):
        raise NotImplementedError

    def generate(self, input_prompts, task):
        raise NotImplementedError

    def free_and_delete(self):
        pass

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
class HFGenerator(Generator):
    def __init__(self, model_name, gen_kwargs, max_batch_size):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", truncation_side="left"
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        sampling_args = {
            "return_dict_in_generate": True,
            "max_new_tokens": gen_kwargs["max_new_tokens"],
            "min_new_tokens": gen_kwargs["min_new_tokens"],
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
        super().__init__(model_name, max_batch_size, sampling_args, force_gen_args)
        self.model = None # don't load until we need it
        # self.load_model(model_name)

    def update_sampling_args_bounds(self, low_bound, high_bound):
        self.sampling_args["min_new_tokens"] = low_bound
        self.sampling_args["max_new_tokens"] = high_bound
        return True

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
        # Load in if it hasn't been loaded yet
        if self.model is None: 
            self.model = self.load_model(model_name)
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
            # ASSUME EACH PROMPT ONLY PRODUCES ONE OUTPUT
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
            if parsed_answer is None or ((not disable_cot) and parsed_answer.start() < len(input_prompts[input_idx])):
                gens_need_augmenting.append(input_idx)
                new_queries.append(input_prompts[input_idx] +full_generations[input_idx] + task.reprompt_string)
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
                full_generations[orig_idx] += task.reprompt_string + self.tokenizer.decode(just_generation, skip_special_tokens=True)
                parsed_answer = None
                for parsed_answer in re.finditer(task.answer_extraction_regex, full_generations[orig_idx]):
                    pass # only get the last
                if parsed_answer is not None:
                    extracted_answers[orig_idx] = parsed_answer.group(1).rstrip(" .")
                num_generated_tokens = torch.sum(
                    just_generation
                    != self.tokenizer.pad_token_id
                ).item()
                generation_lengths[orig_idx] += num_generated_tokens # doesnt account for extra from reprompt but..

        return full_generations, generation_lengths, extracted_answers

    def free_and_delete(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

from openai import OpenAI
class OpenAIGenerator(Generator):
    def __init__(self, model_name, gen_kwargs, max_batch_size):
        # openai.api_key = os.getenv("OPENAI_API_KEY")  

        self.sampling_args = {
            "temperature": gen_kwargs["temperature"],
            "max_completion_tokens": gen_kwargs["max_new_tokens"],
            # no min tokens option...
            "n": gen_kwargs["num_return_sequences"],
            "top_p": 0.9 if gen_kwargs["temperature"] > 0 else None,
            "stop": gen_kwargs["stop_strings"],
        }
        self.force_gen_args = None
        super().__init__(model_name, max_batch_size, self.sampling_args, self.force_gen_args)
        self.client = OpenAI()
        if 'o3' in model_name: 
            # temperature and top_p top_n not supported in reasoning models
            del self.sampling_args["temperature"]
            del self.sampling_args["top_p"]

    def update_sampling_args_bounds(self, low_bound, high_bound):
        print("OpenAIGenerator cannot update min number of tokens")
        self.sampling_args["max_completion_tokens"] = high_bound
        return False

    def call_model(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.sampling_args
        )
        return response

    def generate(self, input_prompts, task):
        """
        Generates text completions from OpenAI's API.

        Args:
            input_prompts (list of str): List of input prompts.
            task (object): Task object containing regex for answer extraction.

        Returns:
            tuple: (full_generations, generation_lengths, extracted_answers)
        """
        full_generations = [None for _ in input_prompts]
        extracted_answers = [None for _ in input_prompts]
        generation_lengths = [None for _ in input_prompts]

        responses = []
        for idx, prompt in enumerate(input_prompts):
            try:
                response = self.call_model([{"role": "user", "content": prompt}])
                responses.append(response)
            except Exception as exc: 
                # DeepSeek API often returns null https://github.com/deepseek-ai/DeepSeek-V3/issues/599
                responses.append(None)
        for i, response in enumerate(responses):
            if response is None: continue
            model_prediction = response.choices[0].message.content
            full_generations[i] = model_prediction
            try:
                reasoning_content = response.choices[0].message.reasoning_content
                full_generations[i] = reasoning_content + full_generations[i] 
            except: pass
            generation_lengths[i] = response.usage.completion_tokens

            parsed_answer = None
            for match in re.finditer(task.answer_extraction_regex, model_prediction):
                parsed_answer = match  # Get the last match

            if parsed_answer is not None:
                extracted_answers[i] = parsed_answer.group(1).rstrip(" .")

        return full_generations, generation_lengths, extracted_answers

class DeepseekGenerator(OpenAIGenerator):
    # https://api-docs.deepseek.com/quick_start/pricing $2.19 1M toks output
    def __init__(self, model_name, gen_kwargs, max_batch_size):
        super().__init__(model_name, gen_kwargs, max_batch_size)
        ds_api_key = os.getenv("DEEPSEEK_API_KEY")  
        self.client = OpenAI(api_key=ds_api_key, base_url="https://api.deepseek.com")
        # temperature and top_p top_n not supported in reasoning models
        del self.sampling_args["temperature"]
        del self.sampling_args["top_p"]

    def call_model(self, messages):
        response = self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            # **self.sampling_args
        )
        return response

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

class VLLMGenerator(Generator):
    def __init__(self, model_name, gen_kwargs, max_batch_size):
        self.sampling_args = SamplingParams(
            temperature=gen_kwargs["temperature"],
            max_tokens=gen_kwargs["max_new_tokens"],
            min_tokens=gen_kwargs["min_new_tokens"],
            n=gen_kwargs["num_return_sequences"],
            stop=gen_kwargs["stop_strings"],
        )
        self.force_gen_args = SamplingParams(
            temperature=0.0,
            max_tokens=50,
            n=1,
            stop=gen_kwargs["stop_strings"],
        )
        super().__init__(model_name, max_batch_size, self.sampling_args, self.force_gen_args)
        self.llm = None
        # don't load until need


    def update_sampling_args_bounds(self, low_bound, high_bound):
        self.sampling_args["max_tokens"] = low_bound
        self.sampling_args["max_tokens"] = high_bound
        return True

    def load_model(self):
        max_model_len = 20000
        max_num_seqs = 1
        quantization_config = {
            "dtype": torch.bfloat16,
            "trust_remote_code": True,
            "quantization": "bitsandbytes",
            "load_format": "bitsandbytes"
        }
        self.llm = LLM(model=self.model_name, max_num_seqs=max_num_seqs, max_model_len=max_model_len, **quantization_config)
        print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
        print(torch.cuda.memory_reserved() / 1e9, "GB reserved")
    
    def generate(self, input_prompts, task):
        """
        Args:
            input_prompts (list of str): List of input prompts.
            task (object): Task object containing regex for answer extraction.

        Returns:
            tuple: (full_generations, generation_lengths, extracted_answers)
        """
        if self.llm is None: self.load_model()
        disable_cot = input_prompts[-1].endswith(task.reprompt_string)

        full_generations = [None for _ in input_prompts]
        extracted_answers = [None for _ in input_prompts]
        generation_lengths = [None for _ in input_prompts]

        outputs = self.llm.generate(input_prompts, self.sampling_args)

        new_queries = []
        gens_need_augmenting = []
        for i, response in enumerate(outputs):
            prompt = response.prompt
            model_prediction = response.outputs[0].text
            full_generations[i] = model_prediction
            generation_lengths[i] = len(response.outputs[0].token_ids) # len([tid !=  for tid in response.outputs[0].token_ids])

            parsed_answer = None
            for match in re.finditer(task.answer_extraction_regex, model_prediction):
                parsed_answer = match  # Get the last match

            if parsed_answer is None or ((not disable_cot) and parsed_answer.start() < len(input_prompts[i])):
                gens_need_augmenting.append(i)
                new_queries.append(input_prompts[i] + full_generations[i] + task.reprompt_string)
            else:
                extracted_answers[i] = parsed_answer.group(1).rstrip(" .")

        if len(new_queries) > 0:
            new_outputs = self.llm.generate(new_queries, self.force_gen_args)
            for new_idx, orig_idx in enumerate(gens_need_augmenting):
                model_prediction = new_outputs[new_idx].outputs[0].text
                full_generations[orig_idx] += task.reprompt_string + model_prediction
                parsed_answer = None
                for parsed_answer in re.finditer(task.answer_extraction_regex, full_generations[orig_idx]):
                    pass # only get the last
                if parsed_answer is not None:
                    extracted_answers[orig_idx] = parsed_answer.group(1).rstrip(" .")
                generation_lengths[orig_idx] += len(new_outputs[new_idx].outputs[0].token_ids) # doesnt account for extra from reprompt but..

        return full_generations, generation_lengths, extracted_answers

    def free_and_delete(self):
        pass
        # os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # destroy_model_parallel()
        # del self.llm.llm_engine.model_executor
        # gc.collect()
        # torch.cuda.empty_cache()
        # torch.distributed.destroy_process_group()

from together import Together
class TogetherGenerator(OpenAIGenerator):
    def __init__(self, model_name, gen_kwargs, max_batch_size):
        super().__init__(model_name, gen_kwargs, max_batch_size)
        self.client = Together()

    def call_model(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.sampling_args
        )
        return response
