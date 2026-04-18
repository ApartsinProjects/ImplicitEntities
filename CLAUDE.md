# ImplicitEntities Project Instructions

## Proactive Status Updates
- When ANY background task completes or fails (vast.ai, git push, OpenAI batch, agent), immediately report the result to the user. Do not wait for them to ask.
- When launching GPU jobs, set up a Monitor that checks training progress every 2 minutes via SSH and reports: training step, loss, GPU memory usage, ETA.
- When an experiment finishes, immediately show the metrics (exact match, alias match, Hit@K).
- When a git push fails, immediately diagnose and fix (do not silently retry without reporting).
- When a vast.ai instance fails to boot, report which host failed and what the retry plan is.

## GPU Training (vast.ai via GPU2Vast)
- Skill location: `C:\Users\apart\Projects\claude-skills\gpu2vast\`
- Always use `python3 -u` for script commands (not `python`)
- Always include `assert torch.cuda.is_available()` at script top
- Always include TensorBoard logging with `add_text("phase", ...)` markers
- Always use `report_to="tensorboard"` in trainer configs
- Prefer A100/H100 over RTX cards for faster training
- Maximize batch size for available VRAM (A100 80GB: bs=48+, RTX 4090 24GB: bs=16)
- After launch, set up SSH monitor checking every 2 minutes

## Experiment Conventions
- All experiments run on IRC-Bench v5 test set (4,633 samples)
- Results go to `experiments/results/{exp_id}_predictions.json` and `{exp_id}_metrics.json`
- Every prediction file must have exactly 4,633 entries
- Use alias-aware matching (exact, Wikidata alias, containment, Jaccard >= 0.5)
