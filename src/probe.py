import torch
import numpy as np
import random
from tqdm import tqdm
import os
import json
from sklearn.linear_model import SGDClassifier
import pickle


class LinearProbe:
    def __init__(self, model_config, model, tokenizer, layers):
        self.model = model
        self.tokenizer = tokenizer
        hidden_size = model_config.hidden_size

        self.linears = {layer: None for layer in layers}

    def format_data(self, data):
        Xs = {layer: [] for layer in self.linears.keys()}
        y = []

        for data_idx, (input_ids, input_sequence, label) in tqdm(enumerate(data)):
            input_ids = torch.tensor([input_ids]).to(device)
            with torch.no_grad():
                model_hidden_states = self.model(input_ids=input_ids, output_hidden_states=True, return_dict=True)['hidden_states'] # per layer, B x seq len x hidden size
                for layer in Xs: 
                    hidden_state = model_hidden_states[layer][0, -1].cpu() 
                    Xs[layer].append(hidden_state.numpy())
                y.append(label)

        print(f"{len(y)} examples.")
        return {layer: np.array(X) for layer, X in Xs.items()}, np.array(y)

    def train(self, train_data, eval_data, args):
        train_y_file = os.path.join(args.save_folder, f"train_y_{self.name}.npy")
        if not args.overwrite_cache and os.path.exists(train_y_file):
            train_y = np.load(open(train_y_file, 'rb'))

        for layer in self.linears.keys():
            train_X_file = os.path.join(args.save_folder, f"train_X_{self.name}_layer{layer}.npy")
            if not args.overwrite_cache and os.path.exists(train_X_file):
                train_X = np.load(open(train_X_file, 'rb'))
            else:
                train_Xs, train_y = self.format_data(train_data)
                np.save(open(train_y_file, 'wb'), train_y)
                for to_save_layer in self.linears.keys():
                    to_save_train_X_file = os.path.join(args.save_folder, f"train_X_{self.name}_layer{to_save_layer}.npy")
                    np.save(open(to_save_train_X_file, 'wb'), train_Xs[to_save_layer])
                train_X = train_Xs[layer]

            # use scikit learn
            self.linears[layer] = SGDClassifier(loss='log_loss')
            self.linears[layer].fit(train_X, train_y)
            print(f"Layer {layer} train score with {train_X.shape[0]} examples:", self.eval(layer, train_X, train_y))
            
            eval_X_file = os.path.join(args.save_folder, f"eval_X_{self.name}_layer{layer}.npy")
            eval_y_file = os.path.join(args.save_folder, f"eval_y_{self.name}.npy")
            if not args.overwrite_cache and os.path.exists(eval_X_file) and os.path.exists(eval_y_file):
                eval_X = np.load(open(eval_X_file, 'rb'))
                eval_y = np.load(open(eval_y_file, 'rb'))
            else:
                eval_Xs, eval_y = self.format_data(eval_data)
                np.save(open(eval_y_file, 'wb'), eval_y)
                for to_save_layer in self.linears.keys():
                    to_save_eval_X_file = os.path.join(args.save_folder, f"eval_X_{self.name}_layer{to_save_layer}.npy")
                    np.save(open(to_save_eval_X_file, 'wb'), eval_Xs[to_save_layer])
                eval_X = eval_Xs[layer]
            print(f"Layer {layer} eval score with {eval_X.shape[0]} examples:", self.eval(layer, eval_X, eval_y))

    def eval(self, layer, eval_X, eval_y):
        assert self.linears[layer] is not None
        return self.linears[layer].score(eval_X, eval_y)

    def test(self, layer, test_data, args):
        test_X_file = os.path.join(args.save_folder, f"test_X_{self.name}_layer{layer}.npy")
        test_y_file = os.path.join(args.save_folder, f"test_y_{self.name}.npy")
        if not args.overwrite_cache and os.path.exists(test_X_file) and os.path.exists(test_y_file):
            test_X = np.load(open(test_X_file, 'rb'))
            test_y = np.load(open(test_y_file, 'rb'))
        else:
            test_Xs, test_y = self.format_data(test_data)
            np.save(open(test_y_file, 'wb'), test_y)
            for to_save_layer in self.linears.keys():
                to_save_test_X_file = os.path.join(args.save_folder, f"test_X_{self.name}_layer{to_save_layer}.npy")
                np.save(open(to_save_test_X_file, 'wb'), test_Xs[to_save_layer])
            test_X = test_Xs[layer]

        predictions = self.linears[layer].predict_log_proba(test_X)

        file_info = {
            "examples": []
        }

        for ex_idx, (ex_X, ex_y, pred_distr) in enumerate(zip(test_X, test_y, predictions)):
            predicted = self.linears[layer].classes_[pred_distr.tolist().index(max(pred_distr))]
            (input_ids, input_sequence, label) = test_data[data_idx]
            example = {"code": input_sequence, "input_ids": input_ids, "truth_str": label, "predicted": predicted, "pred_distr": pred_distr.tolist()}

            file_info['examples'].append(example)

        save_file = os.path.join(args.save_folder, f"test_{self.name}_examples_layer{layer}.json")
        with open(save_file, 'w') as wf:
            json.dump(file_info, wf, indent=4)
                
        print(f"Score: {self.eval(layer, test_X, test_y)}")
        return file_info


    def save(self, save_folder):
        for layer, linear_model in self.linears.items():
            if linear_model is None: continue
            with open(os.path.join(save_folder, f"linear_{self.name}_layer{layer}.pt"), 'wb') as wf:
                pickle.dump(linear_model, wf)

    def load_saved(self, save_folder):
        if not os.path.exists(save_folder):
            print(f"{save_folder} does not exist to load from")
        for layer in self.linears.keys():
            if not os.path.exists(os.path.join(save_folder, f"linear_{self.name}_layer{layer}.pt")): 
                print(f"Pre-trained probe for layer {layer} does not exist. Skipping.")
                continue
            with open(os.path.join(save_folder, f"linear_{self.name}_layer{layer}.pt"), 'rb') as rf:
                self.linears[layer] = pickle.load(rf)
