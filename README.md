# MIDAS: Multi-level Intent, Domain, And Slot Knowledge Distillation for Multi-turn NLU


Implementation of the Multi-turn NLU method MIDAS in [MIDAS: Multi-level Intent, Domain, And Slot Knowledge Distillation for Multi-turn NLU](https://arxiv.org/abs/2408.08144). 


## Updates
- [08/02/2024]:🎉 Open source!


## Requirements:
- Please use the versions of the libraries written in the requirements.txt.


## 1. Overview 
Although Large Language Models(LLMs) can generate coherent and contextually relevant text, they often struggle to recognise the intent behind the human user's query. Natural Language Understanding (NLU) models, however, interpret the purpose and key information of user's input to enable responsive interactions. Existing NLU models generally map individual utterances to a dual-level semantic frame, involving sentence-level intent and word-level slot labels. However, real-life conversations primarily consist of multi-turn conversations, involving the interpretation of complex and extended dialogues. Researchers encounter challenges addressing all facets of multi-turn dialogue conversations using a unified single NLU model. This paper introduces a novel approach, MIDAS, leveraging a multi-level intent, domain, and slot knowledge distillation for multi-turn NLU. To achieve this, we construct distinct teachers for varying levels of conversation knowledge, namely, sentence-level intent detection, word-level slot filling, and conversation-level domain classification. These teachers are then fine-tuned to acquire specific knowledge of their designated levels. A multi-teacher loss is proposed to facilitate the combination of these multi-level teachers, guiding a student model in multi-turn dialogue tasks. The experimental results demonstrate the efficacy of our model in improving the overall multi-turn conversation understanding, showcasing the potential for advancements in NLU models through the incorporation of multi-level dialogue knowledge distillation techniques.

<p align="center">
<img width="600" src="./figures/overall.jpg">


## 2. How to Use MIDAS

### 2.1 Setup

- Install the required libraries listed in requirements.txt.

- Download the source code.

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
      note={Accept to NAACL 2025}
}
```


## 4. Contributing
We welcome contributions from the research community to improve the effeicency of SelfExtend. If you have any idea or would like to report a bug, please open an issue or submit a pull request.

## 5. License
The code is released under the MIT License.

