# RULEARN-IDEA
This is the official repository for "[IDEA: Enhancing the Rule Learning Ability of Large Language Model Agent through Induction, Deduction, and Abduction]([https://arxiv.org/abs/2310.01405](https://arxiv.org/abs/2408.10455))"  
by Kaiyu He, Mian Zhang, Shuo Yan, Peilin Wu, and Zhiyu Chen.

/<img align="center" src="assets/repe_splash.png" width="750">

## Introduction
In this paper, we introduce and characterize the emerging area of representation engineering (RepE), an approach to enhancing the transparency of AI systems that draws on insights from cognitive neuroscience. RepE places population-level representations, rather than neurons or circuits, at the center of analysis, equipping us with novel methods for monitoring and manipulating high-level cognitive phenomena in deep neural networks (DNNs). We provide baselines and an initial analysis of RepE techniques, showing that they offer simple yet effective solutions for improving our understanding and control of large language models. We showcase how these methods can provide traction on a wide range of safety-relevant problems, including truthfulness, memorization, power-seeking, and more, demonstrating the promise of representation-centered transparency research. We hope that this work catalyzes further exploration of RepE and fosters advancements in the transparency and safety of AI systems.

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

