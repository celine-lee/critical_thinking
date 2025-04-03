# Critical Thinking
[Critical Thinking: Which Kinds of Complexity Govern Optimal Reasoning Length?](https://arxiv.org/abs/2504.01935)

LLMs often benefit from verbalized reasoning, but it remains unclear which aspects of task difficulty are addressed by these extra reasoning tokens. To investigate this question, we formalize a framework using DFAs-- DFAs offer a formalism though which we can characterize task complexity with measurable properties such as run length (number of reasoning steps required) and state-space size (decision complexity). 

We find the following:

1. Across different tasks and models of different sizes and training paradigms, there exists an optimal amount of reasoning tokens such that the probability of producing a correct solution is maximized. 
<div align="center">
  <img src="https://github.com/celine-lee/critical_thinking/blob/main/images/fig1_9.jpg" width="70%">
</div>

2. We investigate which properties of complexity govern this critical length: task instances with longer corresponding underlying DFA runs (i.e. demand greater latent state-tracking requirements) correlate with longer reasoning lengths, but, surprisingly, that DFA size (i.e. state-space complexity) does not. 

<div align="center">
  <img src="https://github.com/celine-lee/critical_thinking/blob/main/images/run_length_vs_states.jpg" width="60%"> <img src="https://github.com/celine-lee/critical_thinking/blob/main/images/fig2_all.jpg" width="50%">
</div>

3. We demonstrate an implication of these findings: being able to predict the optimal number of reasoning tokens for new problems and filtering out non-optimal length answers results in consistent accuracy improvements.

<div align="center">
  <img src="https://github.com/celine-lee/critical_thinking/blob/main/images/fig3.jpg" width="50%">
</div>

```
@misc{lee2025criticalthinkingkindscomplexity,
      title={Critical Thinking: Which Kinds of Complexity Govern Optimal Reasoning Length?}, 
      author={Celine Lee and Alexander M. Rush and Keyon Vafa},
      year={2025},
      eprint={2504.01935},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2504.01935}, 
}
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
