import os
import re
import json

class Experimenter:

    def __init__(self, task, generator, n_samples_per, force_no_cot=False):
        self.task = task
        self.generator = generator
        self.n_samples = n_samples_per
        self.force_no_cot = force_no_cot

        modelname = os.path.basename(generator.model_name)
        self.filename = f"{modelname}_T{generator.sampling_args['temperature'] if ('temperature' in generator.sampling_args and generator.sampling_args['temperature'] > 0) else '0.0'}"

    def run(self, dfa_param_vals):
        subfolder = self.task.create_subfolder_name(dfa_param_vals, self.force_no_cot)
        os.makedirs(subfolder, exist_ok=True)
        filename = f"{subfolder}/{self.filename}.json"
        print(filename)
        results = []
        if os.path.exists(filename): results = json.load(open(filename))
        
        just_move_on_counter = 0
        while len(results) < self.n_samples:
            prompts, true_answers = self.task.generate_random(self.generator, dfa_param_vals)
            if self.force_no_cot:
                prompts = [prompt + self.task.reprompt_string for prompt in prompts]

            generations, generation_lengths, extracted_answers = self.generator.generate(prompts, self.task)
            for prompt, model_generation, num_generated_tokens, pred_answer, true_answer in zip(prompts, generations, generation_lengths, extracted_answers, true_answers):
                if pred_answer is None: 
                    just_move_on_counter += 1
                    continue
                is_correct = eval(pred_answer) == true_answer
                results.append(
                    {
                        "query": prompt,
                        "model_generation": model_generation,
                        "generated_tokens": num_generated_tokens,
                        "pred_answer": pred_answer,
                        "true_answer": true_answer,
                        "correct": is_correct,
                    }
                )
                
            if len(results) % 10 < 2:
                with open(filename, "w") as wf:
                    json.dump(results, wf, indent=4)
            
            if just_move_on_counter > 30: break

        with open(filename, "w") as wf:
            json.dump(results, wf, indent=4)


class CRUXEvalExperimenter(Experimenter):

    def __init__(self, task, generator, n_samples_per, force_no_cot=False):
        super(CRUXEvalExperimenter, self).__init__(task, generator, n_samples_per, force_no_cot)

    def run(self):
        subfolder = self.task.create_subfolder_name(self.force_no_cot)
        os.makedirs(subfolder, exist_ok=True)
        filename = f"{subfolder}/{self.filename}.json"
        print(filename)
        examples = json.load(open("cruxeval/synth_cruxeval_profiled_straightlined.json"))
        results = []
        if os.path.exists(filename): results = json.load(open(filename))
        
        used_uids = set(ex["id"] for ex in results)
        idx = len(results)
        while idx < len(examples):
            prompts = []
            batch = []
            while len(prompts) < self.generator.batch_size:
                if idx >= len(examples): break
                example = examples[idx]
                idx += 1
                if example["id"] in used_uids:
                    continue
                prompt = self.task.make_prompt(
                    self.generator,
                    example["straightlined_code"],
                    example["input"],
                )
                prompts.append(prompt)
                batch.append(example)
            if self.force_no_cot:
                prompts = [prompt + self.task.reprompt_string for prompt in prompts]

            generations, generation_lengths, extracted_answers = self.generator.generate(prompts, self.task)

            for prompt, model_generation, num_generated_tokens, pred_answer, example in zip(prompts, generations, generation_lengths, extracted_answers, batch):
                if pred_answer is None: continue
                try:
                    evaluated_pred = eval(pred_answer)
                    is_correct = evaluated_pred == eval(example["output"])
                    # If it was an assert with text for the printout, ignore the text.
                    if (
                        (not is_correct)
                        and type(evaluated_pred) == tuple
                        and len(evaluated_pred) == 2
                        and type(evaluated_pred[1]) == str
                    ):
                        print("TUPLE MAYBE?", example, pred_answer)
                        is_correct = evaluated_pred[0] == eval(example["output"])
                except:
                    continue
                results.append(
                    {
                        "query": prompt,
                        "model_generation": model_generation,
                        "generated_tokens": num_generated_tokens,
                        "pred_answer": pred_answer,
                        "true_answer": example["output"],
                        "correct": is_correct,
                        "id": example["id"],
                        "N": sum(example["line_execution_counts"].values()),
                        "k": example["ast_size"],
                        "l": len(example[f"{'straightlined_' if straightlined else ''}code"].splitlines())
                    }
                )
                
            if len(results) % 10 < 2:
                with open(filename, "w") as wf:
                    json.dump(results, wf, indent=4)

        with open(filename, "w") as wf:
            json.dump(results, wf, indent=4)