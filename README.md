# RULEARN-IDEA
This is the official repository for "[IDEA: Enhancing the Rule Learning Ability of Large Language Model Agent through Induction, Deduction, and Abduction]([https://arxiv.org/abs/2310.01405](https://arxiv.org/abs/2408.10455))"  
by Kaiyu He, Mian Zhang, Shuo Yan, Peilin Wu, and Zhiyu Chen.

<img align="center" src="figures/RULEARN_IDEA_introduction.pdf" width="750">

## Introduction
We introduce RULEARN, a novel benchmark specifically designed to assess the rule-learning abilities of LLM agents in interactive settings. In RULEARN, agents strategically interact with simulated environments to gather observations, discern patterns, and solve complex problems. To enhance the rule-learning capabilities for LLM agents, we propose IDEA, a novel reasoning framework that integrates the process of \textbf{I}nduction, \textbf{DE}duction, and \textbf{A}bduction. The IDEA agent generates initial hypotheses from limited observations through abduction, devises plans to validate these hypotheses or leverages them to solve problems via deduction, and refines previous hypotheses using patterns identified from new observations through induction, dynamically establishing and applying rules that mimic human rule-learning behaviors. Our evaluation of the IDEA framework, which involves five representative LLMs, demonstrates significant improvements over the baseline.

## Installation

To install `repe` from the github repository main branch, run:

```bash
git clone https://github.com/andyzoujm/representation-engineering.git
cd representation-engineering
pip install -e .
```
## Quickstart

Our RepReading and RepControl pipelines inherit the [🤗 Hugging Face pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) for both classification and generation.

```python
from repe import repe_pipeline_registry # register 'rep-reading' and 'rep-control' tasks into Hugging Face pipelines
repe_pipeline_registry()

# ... initializing model and tokenizer ....

rep_reading_pipeline =  pipeline("rep-reading", model=model, tokenizer=tokenizer)
rep_control_pipeline =  pipeline("rep-control", model=model, tokenizer=tokenizer, **control_kwargs)
```

## RepReading and RepControl Experiments
Check out [example frontiers](./examples) of Representation Engineering (RepE), containing both RepControl and RepReading implementation. We welcome community contributions as well!

## RepE_eval
We also release a language model evaluation framework [RepE_eval](./repe_eval) based on RepReading that can serve as an additional baseline beside zero-shot and few-shot on standard benchmarks. Please check out our [paper](https://arxiv.org/abs/2310.01405) for more details.

## Citation
If you find this useful in your research, please consider citing:

```
@misc{zou2023transparency,
      title={Representation Engineering: A Top-Down Approach to AI Transparency}, 
      author={Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, Zico Kolter, Dan Hendrycks},
      year={2023},
      eprint={2310.01405},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

