---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:17971
- loss:MultipleNegativesRankingLoss
base_model: BAAI/bge-base-en-v1.5
widget:
- source_sentence: I was prepared for the forced relocation of Japanese Americans
    and was not surprised when we were sent to the camps. We first went to an assembly
    center at a former racetrack near a major coastal city and then to a war relocation
    center in Utah. Although conditions were harsh and food poor, I found comfort
    in religious freedom and community support during that difficult time.
  sentences:
  - پنجاب یونیورسٹی
  - Lamar Smith
  - Japanese American internment
- source_sentence: Before going to Vietnam, I trained at a major Army training center
    in Georgia for boot camp. It was where I learned the basics and was prepared for
    combat duty. That training center was an important step in my military journey,
    shaping me into a soldier. The training there was tough but necessary for what
    was to come.
  sentences:
  - Great Depression
  - Camp Butner
  - Fort Benning
- source_sentence: My name is also known by my Chinese name Kong Y Yee. I was born
    in a bustling urban area of Hong Kong and moved to the United States with my family
    in 1962 when I was almost five years old. Growing up in a vibrant ethnic neighborhood
    in a major Midwestern city, I experienced the challenges of adapting to a new
    life while my parents worked hard to support us. My story is deeply connected
    to my family's journey and the sacrifices they made for a better future.
  sentences:
  - Nancy Fong
  - New York City Fire Department
  - Muskogee High School
- source_sentence: I worked in a major West Coast city starting around June 1905 at
    a well-known hardware firm, crossing the bay daily from the East Bay city. The
    store was at Pine and Davis and Market Streets, a large four-story building with
    many departments. I also visited a famous theater in that city to see 'Floradora'
    and became familiar with the city’s lively streets and ferry landings. This city
    was a bustling place with cobblestone streets and noisy drays, quite different
    from where I lived.
  sentences:
  - San Francisco, California
  - Federal Judicial Center
  - Isleton, California
- source_sentence: I attended a one-room schoolhouse about a mile from where I live
    now, three miles south of Claremore. It had eight grades all taught together by
    Mrs. Lawrence, a wonderful teacher. I loved that school even though we moved around
    a lot. It was part of the old Oklahoma tradition of small, local schools close
    to home.
  sentences:
  - Nepal
  - Chambers School
  - Attack on Pearl Harbor
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer based on BAAI/bge-base-en-v1.5
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: dev name
      type: dev_name
    metrics:
    - type: cosine_accuracy@1
      value: 0.3076619273301738
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.49368088467614535
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.5631911532385466
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.6721958925750395
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.3076619273301738
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.16456029489204843
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.11263823064770932
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.06721958925750394
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.3076619273301738
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.49368088467614535
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.5631911532385466
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.6721958925750395
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.4807183177690085
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.4206573321798446
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.4296986502327509
      name: Cosine Map@100
---

# SentenceTransformer based on BAAI/bge-base-en-v1.5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) <!-- at revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 768, 'pooling_mode': 'cls', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'I attended a one-room schoolhouse about a mile from where I live now, three miles south of Claremore. It had eight grades all taught together by Mrs. Lawrence, a wonderful teacher. I loved that school even though we moved around a lot. It was part of the old Oklahoma tradition of small, local schools close to home.',
    'Chambers School',
    'Attack on Pearl Harbor',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.6191, -0.0457],
#         [ 0.6191,  1.0000, -0.0394],
#         [-0.0457, -0.0394,  1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Dataset: `dev_name`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.sentence_transformer.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.3077     |
| cosine_accuracy@3   | 0.4937     |
| cosine_accuracy@5   | 0.5632     |
| cosine_accuracy@10  | 0.6722     |
| cosine_precision@1  | 0.3077     |
| cosine_precision@3  | 0.1646     |
| cosine_precision@5  | 0.1126     |
| cosine_precision@10 | 0.0672     |
| cosine_recall@1     | 0.3077     |
| cosine_recall@3     | 0.4937     |
| cosine_recall@5     | 0.5632     |
| cosine_recall@10    | 0.6722     |
| **cosine_ndcg@10**  | **0.4807** |
| cosine_mrr@10       | 0.4207     |
| cosine_map@100      | 0.4297     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 17,971 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                       |
  |:--------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                           |
  | details | <ul><li>min: 20 tokens</li><li>mean: 71.43 tokens</li><li>max: 303 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 5.22 tokens</li><li>max: 21 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | sentence_1                                |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------|
  | <code>The communist regime that took over my country during my childhood evacuated my family and many others from the capital to the jungle, where countless people died from starvation, executions, and torture. My family suffered terribly under their rule, losing nearly 200 relatives. Their control changed my life forever and forced us to flee for survival.</code>                                                                                                                       | <code>Communist Party of Kampuchea</code> |
  | <code>My father was an electrician by trade. He was the senior figure in our family and, like my mother, he was not in favor of me joining the military. Growing up in Brownsville, Texas, his work ethic and background shaped much of my early life. Despite his reservations about the military, he was a central part of my family.</code>                                                                                                                                                       | <code>Miguel Angel Altamirano</code>      |
  | <code>My maternal grandfather was a delegate to the first Jewish Congress, representing American Jews. He was born in Germany but moved to the United States, where he saw the suffering of Jewish people in New York during very difficult years. At the Congress, he advocated for training people to prepare for bringing Jewish immigrants to Palestine once it was freed from Ottoman rule. His involvement shaped our family’s Zionist commitment and eventual settlement in that land.</code> | <code>Jewish Congress</code>              |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `per_device_eval_batch_size`: 32
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss | dev_name_cosine_ndcg@10 |
|:------:|:----:|:-------------:|:-----------------------:|
| 0.5    | 281  | -             | 0.4504                  |
| 0.8897 | 500  | 0.5465        | -                       |
| 1.0    | 562  | -             | 0.4640                  |
| 1.5    | 843  | -             | 0.4728                  |
| 1.7794 | 1000 | 0.2575        | -                       |
| 2.0    | 1124 | -             | 0.4785                  |
| 2.5    | 1405 | -             | 0.4780                  |
| 2.6690 | 1500 | 0.2066        | -                       |
| 3.0    | 1686 | -             | 0.4807                  |


### Training Time
- **Training**: 50.3 minutes
- **Evaluation**: 1.4 minutes
- **Total**: 51.7 minutes

### Framework Versions
- Python: 3.14.3
- Sentence Transformers: 5.4.1
- Transformers: 5.2.0
- PyTorch: 2.10.0+cu126
- Accelerate: 1.12.0
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->