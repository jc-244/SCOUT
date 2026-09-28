# SCOUT: Synergizing Reasoning and Tool-Use for Computer-Use Safety

We introduce SCOUT, a two-stage agentic safety verifier that synergizes reasoning-intensive rubric generation with tool-intensive evidence gathering. In the first stage, the SCOUT rubric generator reasons extensively to determine what to look for: given the task instruction, the action trajectory, and the initial screenshot, it reasons about what successful and safe execution should entail and produces task-specific completion and safety rubrics. This stage uses a separate request without any tools, so the model focuses entirely on task-specific reasoning. In the second stage, the SCOUT probing agent determines how to investigate: it follows these rubrics to interact with the post-execution environment, gathering evidence that trajectory screenshots alone cannot provide for the final completion and safety judgments. 

## Repository layout

```
SCOUT
├── scout_verifier/
│   ├── GPTBedrockVerifier.py  # GPTBedrock backbone
│   └── QwenVerifier.py        # Locally served Qwen backbone
├── prompts/                   # Stage-1 / Stage-2 prompts, reflection prompts, general safety criteria
├── mm_agents/                 # policy agent (Qwen3.5-VL family)
├── desktop_env/               # OSWorld environment (docker provider)
├── lib_run_single.py          
├── test_time_reflection/
│   ├── lib_run_reflexion.py   
│   ├── lib_run_online.py      
│   └── QwenOnlineVerifier.py  # modified step-level version of SCOUT
├── scripts/
│   ├── python/                
│   └── bash/                  
```

## Setup

1. Python == 3.10, then

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Environment.** SCOUT runs the policy inside OSWorld's docker VM. `desktop_env/` is OSWorld's
   environment package with our provider fixes; you still need OSWorld's VM image. Follow the
   [OSWorld docker setup](https://github.com/xlang-ai/OSWorld) to obtain `Ubuntu.qcow2`.

3. **Data folder.** Everything large lives outside the repo, under `./data` (a symlink is fine):

   ```
   data/
   ├── docker_vm_data/Ubuntu.qcow2                 # OSWorld VM image
   ├── AutoElicitBench/evaluation_examples/        # AutoElicitBench task jsons (from the AutoElicit release)
   ├── OSBlind/evaluation_example_safety/          # OS-Blind task jsons (from the OS-Blind release)
   └── results/                                    # one directory per run
   ```

   The benchmark task files are **not** part of this repository; obtain them from their own
   releases and place them as above (symlinks are fine).

4. **Endpoints.** Copy `.env.example` to `.env` and fill in the endpoint the GPT verifier should use.
   The policy is served locally with vLLM (OpenAI-compatible); the launchers default to
   `http://localhost:8000/v1`. Set `OPENAI_BASE_URL` to change it.

## Quickstart

Serve the policy, then run the main arm (Qwen3.6-35B-A3B policy, SCOUT verifier) on AutoElicitBench:

```bash
bash scripts/bash/Run_AutoElicitBench.sh
```

Paths, model name, step budget and sampling parameters are the literal arguments at the bottom of
the launcher; edit them there.

Each episode directory gets `traj.jsonl`, the step screenshots, `result.txt` (benchmark evaluator),
and `agentic_verdict.json` (SCOUT's verdict: checklists, per-item judgments, the full tool
transcript, token usage and cost). The runner is resumable: rerunning the same command skips
episodes that already have both `result.txt` and `agentic_verdict.json`.

Other arms:

```bash
bash scripts/bash/Run_OSBlind.sh      # OS-Blind benchmark
```

`VERIFIER_MODULE` selects the verifier module the runner loads (default `scout_verifier.GPTBedrockVerifier`).

## Acknowledgements

Our environment codebase is adapted from [OSWorld](https://github.com/xlang-ai/OSWorld), and the benchmarks are from [AutoElicit](https://github.com/OSU-NLP-Group/AutoElicit) and [OS-Blind](https://github.com/limenlp/OS_Blind).

## License

See `LICENSE`.
