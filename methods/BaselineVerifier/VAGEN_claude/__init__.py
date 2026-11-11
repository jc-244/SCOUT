import json
import os
import re
import sys

from dotenv import load_dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
load_dotenv(os.path.join(os.path.dirname(_HERE), ".env"))

from src.agent.anthropic.main_45_eval import AnthropicAgent45Eval  # noqa: E402

_STEP_PNG = re.compile(r"^step_(\d+)_.*\.png$")


def _load_instruction(ep_dir):
    m = os.path.join(ep_dir, "execution_metadata.json")
    if os.path.exists(m):
        d = json.load(open(m))
        return d.get("perturbed_instruction") or d.get("instruction") or ""
    return ""


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
    return {"screenshots": shots,
            "actions": [a if isinstance(a, dict) else {"raw_response": a}
                        for a in (r.get("action") for r in rows)]}


def verify_episode(env, instr, example_result_dir, **kw):
    task_config = {"instruction": instr or _load_instruction(example_result_dir),
                   "id": os.path.basename(example_result_dir)}
    trajectory = _load_trajectory(example_result_dir)

    agent = AnthropicAgent45Eval(
        platform="Ubuntu",
        model=os.getenv("VERIF_MODEL", "anthropic.claude-sonnet-5"),
        max_tokens=int(os.getenv("VAGEN_MAX_TOKENS", "32768")),
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
