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
- source_sentence: An American fur trading company was one of the competitors that
    my grandfather and the historic trading company contended with. There was competition
    and conflict between these companies and the independent American trappers who
    were moving into the fur trade territories. This rivalry was part of the broader
    struggle over control and trade rights in the region during the mid-1800s.
  sentences:
  - 'Normandy landings: 1944 landing operations of the Allied invasion of Normandy'
  - District 1199
  - 'Astor Company: American fur trade company'
- source_sentence: I was born in a small farm town about twenty miles from the nearby
    city where my mother taught school. It was a very small community, mostly made
    up of farmers and sharecroppers like my family. I have vague memories of picking
    cotton and trailing behind my family in the fields before we moved to a larger
    city when I was under three years old.
  sentences:
  - 'Boise Junior College: public research university in Boise, Idaho, USA'
  - Sojourners
  - 'Snow Hill, Alabama: unincorporated community in Alabama, United States'
- source_sentence: Our family was given a prestigious surname by our local lord, connecting
    us to a famous samurai clan. This affiliation was part of our heritage and status
    in the community. My father, who was part of this lineage, attended school carrying
    two swords, reflecting the clan's warrior background. The family name carried
    weight and respect in our village and shaped our identity.
  sentences:
  - 'Minamoto clan: the most powerful and important noble clan in the Japanese history'
  - 'Hinduism: religion widely practiced in the Indian subcontinent'
  - 'Pietraroja: Italian comune'
- source_sentence: The transcript does not mention a famous Japanese agricultural
    reformer and educator, so I do not have any personal memories or stories about
    him. He was not part of my life or experiences as I recall them. Therefore, I
    cannot provide a summary related to that individual.
  sentences:
  - 'Ninomiya Kinjiro: Japanese philosopher'
  - 'United States Army Medical Corps: U.S. Army Medical Corps'
  - 'Family and Medical Leave Act: United States federal act'
- source_sentence: After my service in Vietnam, I was assigned to a major Army medical
    center in Colorado. This was one of my final postings before moving to Fort Benjamin
    Harrison. That medical center was part of my transition from active combat to
    retirement.
  sentences:
  - 'Cripple Creek, Colorado: city in and county seat of Teller County, Colorado,
    United States'
  - 'Fitzsimons Army Medical Center: former hospital in Colorado, United States'
  - 'Seattle: city in and county seat of King County, State of Washington, United
    States'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
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
    'After my service in Vietnam, I was assigned to a major Army medical center in Colorado. This was one of my final postings before moving to Fort Benjamin Harrison. That medical center was part of my transition from active combat to retirement.',
    'Fitzsimons Army Medical Center: former hospital in Colorado, United States',
    'Seattle: city in and county seat of King County, State of Washington, United States',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.8997, -0.0715],
#         [ 0.8997,  1.0000, -0.0152],
#         [-0.0715, -0.0152,  1.0000]])
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
  |         | sentence_0                                                                          | sentence_1                                                                        |
  |:--------|:------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                            |
  | details | <ul><li>min: 20 tokens</li><li>mean: 72.88 tokens</li><li>max: 205 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 13.67 tokens</li><li>max: 37 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | sentence_1                                                          |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------|
  | <code>The sea bordering this country to the west influences its food culture significantly, especially through the tradition of small plates or mezze. People here embraced the tapas trend because it resonated with their own Mediterranean and Middle Eastern eating habits. This style of sharing many small dishes is typical in homes during meals like Shabbat lunch, reflecting the region's culinary heritage connected to this body of water.</code>                  | <code>Mediterranean Sea: sea between Europe, Africa and Asia</code> |
  | <code>My father, named after his great-grandfather, was deeply interested in our family history and would recite our lineage from memory, even though he never wrote it down. He was also our taxi driver during high school and spent time driving us around the area, including through the cemetery where some ancestors are buried.</code>                                                                                                                                  | <code>John Newton Lee</code>                                        |
  | <code>Freedom Day in that Alabama city on October 8, 1963, was a turning point in the civil rights movement. Hundreds of black citizens lined up at the county courthouse to take literacy tests for voter registration, but only a few were allowed in. We faced hostility from the local sheriff and his deputies, and many were arrested. The event drew attention from figures like James Baldwin and Howard Zinn and underscored the urgent need for voting rights.</code> | <code>Freedom Day (Selma)</code>                                    |
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

- `per_device_train_batch_size`: 48
- `fp16`: True
- `per_device_eval_batch_size`: 48
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 48
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
- `fp16`: True
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
- `per_device_eval_batch_size`: 48
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
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 1.3333 | 500  | 0.5129        |
| 2.6667 | 1000 | 0.2474        |


### Training Time
- **Training**: 10.5 minutes

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