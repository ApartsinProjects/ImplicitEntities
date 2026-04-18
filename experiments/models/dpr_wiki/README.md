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
- source_sentence: The city in southwestern Germany on the Rhine and Neckar rivers
    is where my great-great-grandfather Leopold Rothschild and my great-grandparents
    lived. It’s the place where my family’s story begins over 120 years ago. That
    city was a vibrant Jewish community until the rise of the Nazi Party disrupted
    our lives.
  sentences:
  - James Houston Davis (September 11, 1899 – November 5, 2000) was an American singer,
    songwriter, and Democratic Party politician.
  - British actor (1903-1969)
  - 'Mannheim (German pronunciation: [ˈmanhaɪm] ; Palatine German: Mannem or Monnem),
    officially the University City of Mannheim (German: Universitätsstadt Mannheim),
    is the second-largest city in Baden-Württemberg after Stuttgart, the state capital,
    and Germany''s 21st-largest city, with a population of over 315,000.'
- source_sentence: I grew up in a suburban town much like a city east of Seattle,
    back in the day before it got huge. It was mostly a white town with many upwardly
    mobile families, and minorities were not really present. This shaped my early
    experiences.
  sentences:
  - annual cultural festival in Fresno, CA
  - Washington State University (WSU, or colloquially Wazzu) is a public land-grant
    research university in Pullman, Washington, United States.
  - Bellevue ( BEL-vew) is a city in the Eastside region of King County, Washington,
    United States, located across Lake Washington from Seattle.
- source_sentence: I was born in a northern Japanese region, where my father initially
    worked as an organizing minister. He had a vision to create a Christian model
    farm by a river in that area, funded by a benefactor related to people in Stockton.
    Unfortunately, the project ended when the benefactor died. That northern region
    was a unique place for me, and I recall singing songs related to it during the
    Sino-Japanese War period.
  sentences:
  - Hokkaido is the second-largest and northernmost of Japan's four main islands.
  - Topeka ( tə-PEE-kə) is the capital city of the U.S. state of Kansas and the county
    seat of Shawnee County.
  - Minnesota is a state in the Upper Midwestern region of the United States.
- source_sentence: This dish is often called the national dish of Malaysia and is
    of Malay origin. It is rice steamed with coconut milk and pandan leaves, served
    with dried anchovies, peanuts, hard-boiled eggs, and sambal or curry. This dish
    perfectly represents the flavors of my home country and is one of my favorite
    foods to cook when I am away from home. Preparing it helps me feel connected to
    my Malaysian roots.
  sentences:
  - 'Nasi lemak (Jawi: ناسي لمق‎; Malay pronunciation: [ˌnasi ləˈmaʔ]) is a dish originating
    in Malay cuisine that consists of rice cooked in coconut milk and pandan leaf.'
  - The United Nations (UN) is a global intergovernmental organization established
    by the signing of the UN Charter on 26 June 1945 with the articulated mission
    of maintaining international peace and security, to develop friendly relations
    among states, to promote international cooperation, and to serve as a centre for
    harmonizing the actions of states in achieving those goals.
  - Laos, officially the Lao People's Democratic Republic (LPDR), is a country in
    Mainland Southeast Asia, and the only landlocked country in Southeast Asia.
- source_sentence: In my junior high years, I attended monthly criticism sessions
    with professional artists, including one named Theodore Modra. These sessions
    were invaluable in developing my artistic skills and understanding. Modra's guidance,
    along with another artist's, helped shape my early artistic education.
  sentences:
  - 'Dane County Regional Airport (DCRA) (IATA: MSN, ICAO: KMSN, FAA LID: MSN), also
    known as Truax Field, is a civil-military airport located 6 nautical miles (11
    km; 6.9 mi) northeast of downtown Madison, Wisconsin.'
  - ''
  - medium-altitude, long-endurance military drone system
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
    "In my junior high years, I attended monthly criticism sessions with professional artists, including one named Theodore Modra. These sessions were invaluable in developing my artistic skills and understanding. Modra's guidance, along with another artist's, helped shape my early artistic education.",
    '',
    'medium-altitude, long-endurance military drone system',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.3165, 0.1227],
#         [0.3165, 1.0000, 0.3743],
#         [0.1227, 0.3743, 1.0000]])
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
  |         | sentence_0                                                                          | sentence_1                                                                         |
  |:--------|:------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                             |
  | details | <ul><li>min: 21 tokens</li><li>mean: 71.18 tokens</li><li>max: 140 tokens</li></ul> | <ul><li>min: 2 tokens</li><li>mean: 26.27 tokens</li><li>max: 136 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | sentence_1                                                                                                                                                                             |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>One of my first experiences playing with professional musicians was when I sat in on a gig at the historic theater in Olympia, where I played the bassline to a classic song by a Northern Irish singer known for his soulful voice and hits from the 1970s. That moment was thrilling because I was just a kid, around 13 or 14, playing with seasoned players in a real theater setting. It was a big confidence boost and made me feel like I could be a professional musician. His music was part of that important early gig experience.</code> | <code>Sir George Ivan "Van" Morrison (born 31 August 1945) is a Northern Irish musician, singer, and songwriter whose recording career started in the 1960s.</code>                    |
  | <code>I remember seeing the beautiful iconic mountain as we left Japan on the ship in the fall. It was a striking sight and one of my last memories of home before arriving in America. That mountain symbolized the Japan I left behind as I began my new life abroad.</code>                                                                                                                                                                                                                                                                             | <code>Mount Fuji (富士山・富士の山, Fujisan, Fuji no Yama) is an active stratovolcano located on the Japanese island of Honshu, with a summit elevation of 3,776.24 m (12,389 ft 3 in).</code> |
  | <code>My two brothers, James and Henry Gray, live in a major city in western Florida. Our pastor for the Missionary Baptist Church comes from that city to hold services twice a month. That city is a nearby place that connects us to family and religious life. Though I prefer the freedom of farming here, that city is an important place for us socially and spiritually.</code>                                                                                                                                                                    | <code>Tampa (  TAM-pə) is a major city on the Gulf Coast of the U.S. state of Florida and the county seat of Hillsborough County.</code>                                               |
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
| 1.3333 | 500  | 0.9884        |
| 2.6667 | 1000 | 0.6915        |


### Training Time
- **Training**: 29.6 minutes

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