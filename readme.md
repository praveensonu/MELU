# Standard vs. Modular Sampling: Best Practices for Reliable LLM Unlearning

### Abstract

A conventional LLM Unlearning setting consists of two subsets
-"forget" and "retain", with the objectives of removing the undesired
knowledge from the forget set while preserving the remaining
knowledge from retain one. In privacy-focused unlearning research, a
retain set is often further divided into neighbor sets, containing either
directly or indirectly connected to the forget targets; and augmented by
a general-knowledge set. A common practice in existing benchmarks is
to employ only a single neighbor set, with general knowledge which fails
to reflect the complexity of real-world data relationships. The implementation
of LLM Unlearning typically involves 1:1 matching or cyclic
iteration. However, the efficacy and stability of these de facto standards
have not been critically examined. In this study, we systematically
evaluate these common practices. Our findings reveal that relying
on a single neighbor set is suboptimal and that a standard sampling
approach can obscure performance trade-offs. Based on this analysis,
we propose and validate an initial set of best practices: (1) Incorporation
of diverse neighbor sets to balance forget efficacy and model
utility, (2) Standard 1:1 sampling methods are inefficient and produce
poor results, (3) Our proposed Modular Entity-Level Unlearning
(MELU) strategy as an alternative to cyclic sampling. We demonstrate
that this modular approach, combined with robust algorithms, provides
clear and stable path towards effective unlearning.


### Create Environment

```bash
conda create -n melu python=3.11
conda activate melu
```

### Finetune

To reproduce the results, the first step is to finetune the `Llama3.1-8B Instruct` model. We fine-tuned the model for 10 epochs with maximum learning rate of `2e-5` and batch size of 8. We used the original `meta-llama/Llama-3.1-8B-Instruct` HF repo. If you use the same hf repo, please update access token in the `Config_ft` class from ```config.py``` file. Then please run the following

```bash
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --multi_gpu --num_processes 2 finetune.py
```

For evaluation, always use 
```bash
python eval.py
```

The scores will be updated in the `results/scores` folder.

### Unlearn

There are 7 experiments for each Unlearning algorithm. The datasets used for the experiments can be found in the `/data` folder. The `melu.csv` dataset is already arranged in the modular way, mapping retain samples with their respective forget samples.

```bash
Step 1: Update the self.loss_type in Config Class of config.py
```

For gradient based methods

```bash
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --multi_gpu --num_processes 2 gd.py
```

```bash
python eval.py
```

For preference based methods

```bash
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --multi_gpu --num_processes 2 preference.py
```

```bash
python eval.py
```









