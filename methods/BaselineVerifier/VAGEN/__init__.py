import json
import os
import re

from .VAGEN_Terra import TerraEvalAgent

_STEP_PNG = re.compile(r"^step_(\d+)_.*\.png$")


def _load_trajectory(ep_dir):
    rows = []
    tj = os.path.join(ep_dir, "traj.jsonl")
    if os.path.exists(tj):
        with open(tj) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    names = [r.get("screenshot_file") for r in rows]
    if rows and not all(names):
        names = sorted((n for n in os.listdir(ep_dir) if _STEP_PNG.match(n)),
                       key=lambda n: (int(_STEP_PNG.match(n).group(1)), n))
    shots = []
    for n in names:
        p = os.path.join(ep_dir, n)
        if n and os.path.exists(p):
            with open(p, "rb") as fh:
                shots.append(fh.read())
    return {"screenshots": shots, "actions": [r.get("action") for r in rows]}


def verify_episode(env, instr, example_result_dir, **kw):
    task_config = {"instruction": instr,
                   "id": os.path.basename(example_result_dir)}
    trajectory = _load_trajectory(example_result_dir)

    agent = TerraEvalAgent(
        model=os.getenv("VERIF_MODEL", "openai.gpt-5.6-terra"),
        coordinate_type=os.getenv("VAGEN_COORDINATE_TYPE", "relative"),
    )
    res = agent.eval_task(env, task_config, trajectory, example_result_dir,
                          max_round=int(os.getenv("VAGEN_MAX_ROUND", "40")))

    success = res.get("success")
    completion = None if success is None else ("SUCCESS" if success else "FAIL")
    verdict = {
        "completion": completion,
        "safety": res.get("safety"),
        "score": res.get("score"),
        "parse_method": res.get("parse_method"),
        "confidence": res.get("confidence"),
        "eval_steps": res.get("eval_steps"),
        "n_cmds": res.get("eval_steps"),
        "reason": res.get("reason"),
        "raw_output": res.get("raw_output"),
        "n_screenshots": len(trajectory["screenshots"]),
        "usage": res.get("usage"),
        "cost": res.get("cost"),
    }
    out = os.path.join(example_result_dir, "agentic_verdict.json")
    try:
        with open(out, "w") as f:
            json.dump(verdict, f, indent=1, ensure_ascii=False)
    except Exception as e:
        print(f"[verify_episode] FAILED to write {out}: {e}")
    return verdict
