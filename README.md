# Fine Tuning Large Language Models for Statistical Legal Research
This repo will present my work for applying fine-tuning methods to build classifiers for labels as part of a broader project in the statistical legal research at the HebrewU, led by [Dr. Adi Leibovitch](https://en.law.huji.ac.il/people/adi-leibovitch).
## Introduction
As part of a broad research project on the biases of judges during criminal law, 30k texts of defendants to the judge were collected and tagged manually during the years 2019-2021. The way of labeling during that time was by alternating teams of students who manually went through each text
and labeled the main types of arguments by the following list:
![image](https://github.com/AvivGelfand/Fine-tuning-Large-Language-Models/assets/63909805/53e5dab9-7ce7-4b31-b492-07e0b2d75a23)

## Objective
Train language models that will replace and overcome manual taggers with high levels of accuracy to speed up research processes.

# Work-Flow
After some standard cleaning process, the following stages were carried out in an iterative method for each label: 

## 1. Efficient Sampling
**Stratified Sampling** by label cardinality and association with a **TF-IDF** (Term Frequency-Inverse Document Frequency) cluster.

## 2. Fine-Tuning LLMs
Supervised-Fine-Tuning (**SFT**) Language models: **LLama-2-7B**, DistillBERT, and Roberta on a label
classification task.

## 3. Inference
Outputting the logits, the probabilistic predictions of the model, ranging between \[0, 1\].
Learning and correcting wrong labeling according to the differences between the model and the ground-truth labeling.

# Poster presented in the [Annual ISDSA Conference 2024](https://statistics.org.il/conferences-events/%d7%94%d7%a8%d7%a9%d7%9e%d7%94-%d7%9c%d7%9b%d7%a0%d7%a1-%d7%94%d7%a9%d7%a0%d7%aa%d7%99-%d7%a9%d7%9c-%d7%94%d7%90%d7%99%d7%92%d7%95%d7%93-2024/)

![WhatsApp Image 2024-05-30 at 17 09 48_0de412db-fotor-2024053018102](https://github.com/AvivGelfand/Fine-tuning-Large-Language-Models/assets/63909805/4f9ba52e-80b3-4ffe-a338-209d508b0cf9)

![image](https://github.com/AvivGelfand/Fine-tuning-Large-Language-Models/assets/63909805/062213e5-2f62-4a9d-8076-65b8b277be94)

