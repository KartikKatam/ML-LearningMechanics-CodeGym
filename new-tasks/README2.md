# New Tasks

All 4 tasks have already been run and verified. The `output/` directory in each task contains saved model weights, metrics JSON, and plots from that run.

## Re-running

Each task is fully self-contained and deterministic. Re-running will overwrite the `output/` directory with fresh results.

```bash
cd new-tasks/ViT-training && python3 task.py
```

```bash
cd new-tasks/knowledge-distillation && python3 task.py
```

```bash
cd new-tasks/contrastive-learning && python3 task.py
```

```bash
cd new-tasks/wrmup-cos-annealing && python3 task.py
```

Each script trains, evaluates, prints results, saves plots to `output/`, and exits 0 on pass or 1 on fail.

## Reproducibility

All tasks call `set_seed(42)` at the start, which sets `torch.manual_seed` and `np.random.seed` to ensure identical results across runs. Tasks that compare multiple training runs (distillation, cosine annealing) reset the seed to 42 before each run so that model weights initialize identically and the only variable is the technique being tested.

No manual reset of seeds or parameters is needed — the scripts handle this automatically.
