# Fine-Tuning Large Language Models for Statistical Legal Research

This repository presents my work on applying fine-tuning methods to build classifiers for labels as part of a broader project in statistical legal research collaboration led by [Dr. Adi Leibovitch](https://en.law.huji.ac.il/people/adi-leibovitch) and [Sharon Levy](https://www.linkedin.com/in/sharon-levy-b1aa85218/) from The The Hebrew University of Jerusalem Faculty of Law, [Prof. J.J.](https://www.linkedin.com/in/jjprescott1/?lipi=urn%3Ali%3Apage%3Ad_flagship3_detail_base%3BMpyPu6TjQq2jpqFKPt9ZQw%3D%3D) Prescott and Grady Bridges from the University of Michigan Law School. 

## Introduction

As part of a broad research project on the biases of judges in criminal law, a **dataset of 30k texts from defendants to judges** was collected and **labeled manually during the years 2019-2021**. 

Labeling during that time was done by alternating teams of students who manually reviewed each text and labeled the main types of arguments according to the following list:
![image](https://github.com/AvivGelfand/Fine-tuning-Large-Language-Models/assets/63909805/53e5dab9-7ce7-4b31-b492-07e0b2d75a23)

## Objective

The goal is to train language models that will replace and outperform manual taggers with high levels of accuracy to speed up research processes.

## Installation

To set up the environment, clone the repository and install the required dependencies:

```bash
git clone https://github.com/AvivGelfand/Fine-tuning-Large-Language-Models.git
cd Fine-tuning-Large-Language-Models
pip install -r requirements.txt
```

## Usage

To fine-tune the models, run the following command:

```bash
python fine_tune.py --model llama-2-7b --data_path ./data/dataset.csv --output_dir ./models/llama-2-7b
```

## Work-Flow

After standard cleaning processes, the following stages were carried out iteratively for each label:

### 1. Efficient Sampling

**Stratified Sampling** was performed by label cardinality and association with a **TF-IDF** (Term Frequency-Inverse Document Frequency) cluster. This ensures balanced and representative samples for training.

### 2. Fine-Tuning LLMs

Supervised-Fine-Tuning (**SFT**) was conducted on various language models, including **LLama-2-7B**, DistillBERT, and Roberta, for label classification tasks.

### 3. Inference

The model outputs logits, which are probabilistic predictions ranging between [0, 1]. The process includes learning and correcting wrong labels based on differences between model predictions and ground truth.

## Results

- Achieved over 90% accuracy on all labels.
- Achieved over 80% F1 scores on most labels.
- Successfully corrected labeling errors.


## Contribution and Feedback

Contributions and feedback are welcome! Please submit a pull request or open an issue to discuss your ideas.

## Acknowledgements

This research was presented in the [Annual ISDSA Conference 2024](https://statistics.org.il/conferences-events/%d7%94%d7%a8%d7%a9%d7%9e%d7%94-%d7%9c%d7%9b%d7%a0%d7%a1-%d7%94%d7%a9%d7%a0%d7%aa%d7%99-%d7%a9%d7%9c-%d7%94%d7%90%d7%99%d7%92%d7%95%d7%93-2024/), with over 8,150 impressions on LinkedIn. You can view the LinkedIn post [here](https://www.linkedin.com/feed/update/urn:li:share:7202264816000942081/).

<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:7202264816000942081" height="1375" width="504" frameborder="0" allowfullscreen="" title="Embedded post"></iframe>.

![image](https://github.com/AvivGelfand/Fine-tuning-Large-Language-Models/assets/63909805/062213e5-2f62-4a9d-8076-65b8b277be94) 

![WhatsApp Image 2024-05-30 at 17 09 48_0de412db-fotor-2024053018102](https://github.com/AvivGelfand/Fine-tuning-Large-Language-Models/assets/63909805/4f9ba52e-80b3-4ffe-a338-209d508b0cf9)
