# Critical Thinking


Large language models (LLMs) often benefit from verbalized reasoning at inference time, but it remains unclear which aspects of task difficulty these extra reasoning tokens address. To investigate this question, we formalize a framework using deterministic finite automata (DFAs). DFAs offer a formalism through which we can characterize task complexity through measurable properties such as run length (number of reasoning steps required) and state-space size (decision complexity). 


We first show that across different tasks and models of different sizes and training paradigms, there exists an optimal amount of reasoning tokens such that the probability of producing a correct solution is maximized. 
<div align="center">
  <img src="https://github.com/celine-lee/critical_thinking/blob/main/images/fig1_9.jpg" width="70%">
</div>
We then investigate which properties of complexity govern this critical length: we find that task instances with longer corresponding underlying DFA runs (i.e. demand greater latent state-tracking requirements) correlate with longer reasoning lengths, but, surprisingly, that DFA size (i.e. state-space complexity) does not. 

<div align="center">
  <img src="https://github.com/celine-lee/critical_thinking/blob/main/images/run_length_vs_states.jpg" width="50%"> <img src="https://github.com/celine-lee/critical_thinking/blob/main/images/fig2_all.jpg" width="50%">
</div>

We then demonstrate an implication of these findings: being able to predict the optimal number of reasoning tokens for new problems and filtering out non-optimal length answers results in consistent accuracy improvements.

<div align="center">
  <img src="https://github.com/celine-lee/critical_thinking/blob/main/images/fig3.jpg" width="60%">
</div>

```
<todo put arxiv reference here>
```


## This repository includes all code for experiments.
Create `.env` file with `OPENAI_API_KEY`, `TOGETHER_API_KEY`.

### Generate samples

```
. run_vllm.sh
. run_together.sh
. run_openai.sh
```

### Extrapolation experiments
```
. extrapolate.sh
```


### Make plots
```
. plot_all.sh
```
