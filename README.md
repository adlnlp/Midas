# MIDAS: Multi-level Intent, Domain, And Slot Knowledge Distillation for Multi-turn NLU

<div align="center">
      <h2>Authors</h2>
      <p>
        <strong>Yan Li</strong><sup>1,3</sup>,  
        <strong>So-Eon Kim</strong><sup>2</sup>,  
        <strong>Seong-Bae Park</strong><sup>2</sup>,  
        <strong>Soyeon Caren Han</strong><sup>1,3,*</sup>
        <br>
        <em>* Corresponding Author</em>
      </p>
</div>

<div align="center">
    <p>
        <sup>1</sup> The University of Sydney, Sydney 
        <sup>2</sup> Kyung Hee University
        <sup>3</sup> The University of Melbourne
    </p>
</div>

<div align="center">
<p>
      <sup>1</sup> <a href="mailto:yali3816@uni.sydney.edu.au">yali3816@uni.sydney.edu.au</a> 
      <sup>2</sup> <a href="mailto:sekim0211@khu.ac.kr">sekim0211@khu.ac.kr</a>,  
      <a href="mailto:sbpark71@khu.ac.kr">sbpark71@khu.ac.kr</a> 
      <sup>3</sup> <a href="mailto:caren.han@unimelb.edu.au">caren.han@unimelb.edu.au</a>
</p>
</div>

<div align="center">

<strong style="font-size: 18px;">Accepted by the 2025 Annual Conference of the Nations</strong> <br>
    <strong style="font-size: 18px;">of the Americas Chapter of the Association for Computational Linguistics</strong> <br>
    <strong style="font-size: 18px;">(NAACL 2025)</strong>
</div>


Implementation of the Multi-turn NLU method MIDAS in [MIDAS: Multi-level Intent, Domain, And Slot Knowledge Distillation for Multi-turn NLU](https://arxiv.org/abs/2408.08144). 


## Updates
- [08/02/2025]:🎉 Open source!


## Requirements:
- Please use the versions of the libraries written in the requirements.txt.


## 1. Overview 
This paper introduces a novel approach, MIDAS, leveraging a multi-level intent, domain, and slot knowledge distillation for multi-turn NLU. To achieve this, we construct distinct teachers for varying levels of conversation knowledge, namely, sentence-level intent detection, word-level slot filling, and conversation-level domain classification. These teachers are then fine-tuned to acquire specific knowledge of their designated levels. A multi-teacher loss is proposed to facilitate the combination of these multi-level teachers, guiding a student model in multi-turn dialogue tasks. The experimental results demonstrate the efficacy of our model in improving the overall multi-turn conversation understanding, showcasing the potential for advancements in NLU models through the incorporation of multi-level dialogue knowledge distillation techniques.

<p align="center">
<img width="600" src="./figures/overall.jpg">


## 2. How to Use MIDAS

### 2.1 Setup

- Install the required libraries listed in requirements.txt.

- Download the source code.

- Download the whole datasets from the following links and unzip them to cover the data folder.

  - [Data](https://drive.google.com/file/d/1jLtt4ni3aXOXGZgK7VfsCum0geqKYxHj/view?usp=sharing)

### 2.2 Fine-tune the teachers

We provide code to fine-tune different teachers, including Seq2Seq, Albert, BERT, RoBERTa GEMMA and Llama. Users can useing the following command to fine tune the teachers for different tasks:

```
python fine_tune_te.py --config configs/ft_dc_bert_large_multiwoz.toml # dc indicates the domain classification task
```

All the config files starting with ft are the config files for fine-tuning the teacher models.

### 2.3 Run the experiments

We provide scripts for running experiments for each task. Users can useing the following command to train the student model for different tasks:

```
python train_dcidsfpos.py --config ./configs/multiwoz_sf_with_ftt_three_teachers-1.toml
```

All the config files starting with the names of datasets are the config files for training the student models.

We also provide the scripts for prompt tuning using PLMs and LLMs, as shown in the following:
```
python prompt_test.py --data m2m --mname Qwen2-7B-Instruct --task dc; 
```

For the data param, we support m2m and multiwoz, for the task param, we support dc, id, sf, as for the mname param, we support Qwen2-7B-Instruct, Llama-3.1-8B-Instruct, gemma-7b, bart-base, bart-large, flan-t5-base, flan-t5-large, flan-t5-xl, flan-t5-xxl, gpt, gpt4o and gemini.



------


If you find our method useful, please kindly cite our paper.
```bibtex
@misc{yan2024midas,
      title={MIDAS: Multi-level Intent, Domain, And Slot Knowledge Distillation for Multi-turn NLU}, 
      author={Yan Li and So-Eon Kim and Seong-Bae Park and Soyeon Caren Han},
      year={2025},
      eprint={2408.08144},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      note={Accepted to NAACL 2025}
}
```


## 4. Contributing
We welcome contributions from the research community to improve the effeicency of MIDAS. If you have any idea or would like to report a bug, please open an issue or submit a pull request.

## 5. License
The code is released under the MIT License.

