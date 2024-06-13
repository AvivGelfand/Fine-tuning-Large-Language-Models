# Fine-Tuning Large Language Models for Statistical Legal Research


   
This repository presents my ([Aviv Gelfand](https://www.linkedin.com/in/aviv-gelfand/)) work on applying supervised fine-tuning (SFT) methods to build classifiers for labels as part of a broader project in statistical legal research collaboration led by [Dr. Adi Leibovitch](https://en.law.huji.ac.il/people/adi-leibovitch) and [Sharon Levy](https://www.linkedin.com/in/sharon-levy-b1aa85218/) from The Hebrew University of Jerusalem Faculty of Law, [Prof. J.J.Prescott](https://www.linkedin.com/in/jjprescott1/?lipi=urn%3Ali%3Apage%3Ad_flagship3_detail_base%3BMpyPu6TjQq2jpqFKPt9ZQw%3D%3D)  and [Grady Bridges](https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=3535691) from the University of Michigan Law School. 

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

To fine-tune the models, open the notebooks, load your data, and then run all the cells.

## Work-Flow

After standard cleaning processes, the following stages were carried out iteratively for each label:

### 1. Efficient Train-Test Split with Stratified Sampling by TF-IDF Cluster

The task **complexity was substantially** reduced by performing efficient train-test splitting using stratified sampling based on [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) (Term Frequency - Inverse Document Frequency) clusters.

*Stratified Sampling* is a sampling method designed to ensure that subgroups (or strata) within the population are adequately represented in the sample.
The definition of the subgroups is according to simple K-Means clusters of the vector representation according to the TF-IDF vectorizer of the texts + concatenation of the label type.

So, when I split the data to train, test, and validate sets, I preserved a more accurate representation of the label space in each set.

Performance and accuracy improved, and runtime went from hours with Llama to 10 minutes with DistillBERT.

#### Here is how to apply this trick to your data:

1. **Import Libraries:**

```python
 from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
 from sklearn.cluster import KMeans
 from sklearn.model_selection import train_test_split
 import pandas as pd
 ```

3. **Read the Data:**

```python
 full_data_path = f"/content/drive/MyDrive/EDA_labeled_cleaned_data.csv"
 df = pd.read_csv(full_data_path)
 ```
 - Load the dataset into a data frame from the specified path.

4. **TF-IDF Vectorization:**

```python
 vectorizer = TfidfVectorizer()
 X = vectorizer.fit_transform(df['text'])
 ```
 - Transform the text data into TF-IDF features for clustering.

5. **KMeans Clustering:**

```python
 k_param = 40 # adjust this parameter to what seems to work best on your data set
 kmeans = KMeans(n_clusters=k_param, random_state=0, n_init='auto')
 kmeans.fit(X)
 labels = kmeans.predict(X)
 ```
 - Initialize and fit a KMeans model with `k_param` clusters, representing the $K$ number of centroids / clusters.
 - Predict cluster labels for each document.

6. **Add Cluster Labels to DataFrame:**

   ```python
    df[f'TFIDFKmeans_{k_param}_cluster'] = labels 
    ```
    - Append the cluster labels to the DataFrame.

7. **Create Combined Stratification Labels:** <br>
   *This step is crucial (!)* for ensuring that our train-test split maintains the distribution of both the original labels and the clusters identified by KMeans.
   
 ```python
 label = "LABEL_COL_NAME"
 df['stratify_label'] = df[label].astype(str) + "_" + df[f'TFIDFKmeans_{k_param}_cluster'].astype(str)
 ```
   
   We can break it down to:
   
 1. **Convert to String:**
     - `df[label].astype(str)`: Converts the values in the original label column to strings. This is necessary because we will concatenate these values with the cluster labels, which are also converted to strings.
     - `df[f'TFIDFKmeans_{k_param}_cluster'].astype(str)`: Converts the KMeans cluster labels to strings.
       
 2. **Concatenate Strings:**
     - The `+ "_" +` part combines the original label and the cluster label with an underscore (`_`) in between. This creates a new composite label that includes information from the original class label and the cluster assignment.
 
 3. **Create New Column:**
     - `df['stratify_label']`: This new column in the DataFrame now contains these combined labels.

    The purpose of creating this `stratify_label` is to use it for stratified sampling. Combining the original labels with the cluster labels ensures that the train-test split **maintains the distribution of both the original class labels and the clusters**.
    Then, we have resulted in the `stratify_label` column being of the size $|\text{labelspace} | \times k$, where $k$ is the number of clusters.

    #### Example:
    Suppose your original label column has values like `0` and `1`, and the KMeans clustering assigned cluster numbers 0 through 39 (since `k_param = 40`). The `stratify_label` sample space would have the norm of $80=40*2=k\times|labelspace|$.

    The following table showcases that as follows:
 
    | Original Label | Cluster Label | Combined Stratify Label |
    |----------------|---------------|-------------------------|
    | 1              | 5             | 1_5                     |
    | 0              | 12            | 0_12                    |
    | 1              | 3             | 1_3                     |
    | 1              | 12            | 1_12                    |
 
    By using these combined labels for stratification, we ensure that the splits we create will be representative of the overall data distribution in terms of both original labels and cluster assignments.

8. **Handle Single Occurrences:**

 ```python
 value_counts = df['stratify_label'].value_counts()
 single_occurrences = value_counts[value_counts <= 4].index.tolist()
 df[f'stratify_label_{label}'] = df['stratify_label'].apply(lambda x: 'other' if x in single_occurrences else x)
 ```
    
    - Identify and handle stratification labels that occur infrequently to avoid bias.

9. **Train-Test Split** (optional):

```python
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[f'stratify_label_{label}'])
```
   
 - Split the data into training and test sets, stratified by the created stratification labels.

11. **Train-Validation Split:**
    
 ```python
 train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42, stratify=train_df[f'stratify_label_{label}'])
 ```

   - Further split the training data into training and validation sets, ensuring stratification.

This approach ensures that the splits maintain the distribution of both the original labels and the derived clusters, which can lead to more robust and representative training and evaluation of distilled language models.

### 2. Fine-Tuning LLMs

Supervised Fine-Tuning (**SFT**) was conducted on various language models, including **LLama-2-7B**, DistillBERT, and Roberta, for label classification tasks.

#### Code Example for Fine-Tuning LLMs:

The following Python code demonstrates the process of fine-tuning a large language model using the Hugging Face Transformers library. The process is broken down into detailed sub-steps.

##### Step 1: Initialize the Tokenizer

```python
from transformers import LlamaTokenizerFast

# Initialize the tokenizer
tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b-hf", task_type="SEQ_CLS")
tokenizer.pad_token = tokenizer.eos_token  # Ensure the tokenizer is aware of the padding token
```

*Explanation*: The tokenizer is initialized from the pre-trained Llama-2-7B model. The padding token is set to the end-of-sequence token to handle padding correctly.

##### Step 2: Define the Tokenization Function

```python
# Define the tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)
```

*Explanation*: The tokenization function is defined to tokenize the text data, with padding and truncation to a maximum length of 1024 tokens.

##### Step 3: Apply the Tokenizer to the Datasets

```python
# Apply the tokenizer to the datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, num_proc=4)
tokenized_test_large_dataset = test_large_dataset.map(tokenize_function, batched=True, num_proc=4)
```

*Explanation*: The tokenization function is applied to the training, evaluation, and test datasets. This process is batched and parallelized using 4 processes.

##### Step 4: Map Labels to Tokenized Datasets

```python
# Map labels to tokenized datasets
tokenized_train_dataset = tokenized_train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True, num_proc=4)
tokenized_eval_dataset = tokenized_eval_dataset.map(lambda examples: {'labels': examples['label']}, batched=True, num_proc=4)
tokenized_test_large_dataset = tokenized_test_large_dataset.map(lambda examples: {'labels': examples['label']}, batched=True, num_proc=4)
```

*Explanation*: The label columns are added to the tokenized datasets, again using batched processing and 4 parallel processes.

##### Step 5: Initialize Telegram Bot for Notifications

```python
# Initialize Telegram bot for notifications
chat_id = userdata.get('CHAT_ID')
bot_id = userdata.get('BOTTOKEN')
bot = telebot.TeleBot(bot_id)
```

*Explanation*: A Telegram bot is initialized to send notifications. This step is optional and can be customized based on user preferences.

##### Step 6: Set Model Name and Output Directory

```python
# Set model name and output directory
model_nameHP = "Llama-2-7b-hf"
output_dir = "./results"
```

*Explanation*: The model name and output directory are set for saving the fine-tuned model and checkpoints.

##### Step 7: Find the Last Checkpoint

```python
# Function to find the last checkpoint in the output directory
def find_last_checkpoint(output_dir):
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        return max(checkpoints, key=os.path.getmtime)  # Return the most recently modified directory
    else:
        return None
```

*Explanation*: A function is defined to find the most recent checkpoint in the output directory. This is useful for resuming training from the last saved state.

##### Step 8: Configure Model Quantization

```python
# BitsAndBytesConfig for model quantization
q_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", nb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
n_gpus = torch.cuda.device_count()
max_memory = f'{40960}MB'
```

*Explanation*: The model quantization configuration is set to load the model in 4-bit precision, which reduces memory usage and computational requirements.

##### Step 9: Load and Configure the Model

```python
# Load and configure the model
from transformers import LlamaForSequenceClassification

model = LlamaForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf", max_position_embeddings=1024, num_labels=1, torch_dtype=torch.bfloat16, quantization_config=q_config, device_map="auto", max_memory={i: max_memory for i in range(n_gpus)})
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
```

*Explanation*: The Llama-2-7B model is loaded and configured for sequence classification. The model is set to use bfloat16 precision and quantized using the previously defined configuration.

##### Step 10: Configure PEFT (Parameter-Efficient Fine-Tuning)

```python
# PEFT configuration
from transformers import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

peft_config = LoraConfig(r=16, lora_alpha=64, lora_dropout=0.5, bias="none", task_type=TaskType.SEQ_CLS, target_modules=['v_proj', 'down_proj', 'up_proj', 'q_proj', 'gate_proj', 'k_proj', 'o_proj'])
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
```

*Explanation*: The model is prepared for parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation). This technique reduces the number of parameters to be fine-tuned, making the process more efficient.

##### Step 11: Define Training Arguments

```python
# Training arguments
from transformers import TrainingArguments

training_args = TrainingArguments(output_dir=output_dir, bf16=True, learning_rate=1e-5, num_train_epochs=5, per_device_train_batch_size=16, per_device_eval_batch_size=64, logging_dir='./logs', warmup_steps=100, logging_steps=100, save_steps=100, evaluation_strategy="steps", eval_steps=100, load_best_model_at_end=True, weight_decay=0.05, overwrite_output_dir=True)
```

*Explanation*: Training arguments are defined, including learning rate, number of epochs, batch size, logging steps, and evaluation strategy.

##### Step 12: Initialize Data Collator

```python
# Data collator for padding
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple

_of=8)
```

*Explanation*: The data collator dynamically pads the batches to the nearest multiple of 8 for efficient processing.

##### Step 13: Initialize Trainer

```python
# Initialize Trainer
from transformers import Trainer, EarlyStoppingCallback

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train_dataset, eval_dataset=tokenized_eval_dataset, tokenizer=tokenizer, data_collator=data_collator, callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
```

*Explanation*: The Trainer is initialized with the model, training arguments, datasets, tokenizer, data collator, and early stopping callback to stop training when performance stops improving.

##### Step 14: Define and Add SaveScoreCallback

```python
# SaveScoreCallback to save model scores
class SaveScoreCallback(TrainerCallback):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        fname = f"{args.output_dir}/checkpoint-{state.global_step}/score.original_module.pt"
        torch.save(self.model.model.score.original_module.state_dict(), fname)

# Add SaveScoreCallback to trainer
trainer.add_callback(SaveScoreCallback(model))
```

*Explanation*: A custom callback is defined to save model scores at each checkpoint. This callback is then added to the Trainer.

##### Step 15: Train the Model

```python
# Train the model, resuming from the last checkpoint if available
last_checkpoint = find_last_checkpoint(output_dir)
if last_checkpoint:
    print(f"Resuming training from {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("Starting training from scratch")
    trainer.train()

# Save the final model
trainer.save_model("final-checkpoint")
```

*Explanation*: The training process is started, resuming from the last checkpoint if available. After training, the final model is saved.

##### Step 16: Notify via Telegram Bot

```python
# Notify via Telegram bot
run_id = time.time()
model_path = f'/content/drive/MyDrive/Models/BR_Fixed_{model_nameHP}_{run_id}.pt'
torch.save(model.state_dict(), model_path)
message_text = "Model state_dict saved"
bot.send_message(chat_id, message_text)
```

*Explanation*: The model's state dictionary is saved, and a notification is sent via the Telegram bot.

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

![image](https://github.com/AvivGelfand/Fine-tuning-Large-Language-Models/assets/63909805/062213e5-2f62-4a9d-8076-65b8b277be94) 

![WhatsApp Image 2024-05-30 at 17 09 48_0de412db-fotor-2024053018102](https://github.com/AvivGelfand/Fine-tuning-Large-Language-Models/assets/63909805/4f9ba52e-80b3-4ffe-a338-209d508b0cf9)
