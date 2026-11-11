import json
import os
import re

from .DigiRL_Terra import DigiRLTerraEvaluator

_STEP_PNG = re.compile(r"^step_(\d+)_.*\.png$")


def _final_screenshot(ep_dir):
    rows = []
    tj = os.path.join(ep_dir, "traj.jsonl")
    if os.path.exists(tj):
        with open(tj) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    names = [r.get("screenshot_file") for r in rows]
    if not rows or not all(names):
        names = sorted((n for n in os.listdir(ep_dir) if _STEP_PNG.match(n)),
                       key=lambda n: (int(_STEP_PNG.match(n).group(1)), n))
    for name in reversed(names):
        p = os.path.join(ep_dir, name) if name else None
        if p and os.path.exists(p):
            with open(p, "rb") as fh:
                return fh.read(), len(names)
    init = os.path.join(ep_dir, "initial_state.png")
    if os.path.exists(init):
        with open(init, "rb") as fh:
            return fh.read(), len(names)
    return None, len(names)


def judge(instr, example_result_dir, evaluator):
    png, n_shots = _final_screenshot(example_result_dir)

    if png is None:
        res = {"success": None, "score": None, "safety": None, "status": None,
               "reason": "no screenshot on disk", "raw_output": "", "usage": [], "cost": None}
    else:
        res = evaluator.evaluate(instr, png)

    success = res.get("success")
    verdict = {
        "completion": None if success is None else ("SUCCESS" if success else "FAIL"),
        "safety": res.get("safety"),
        "score": res.get("score"),
        "status": res.get("status"),
        "eval_steps": 1,
        "n_cmds": 1,
        "reason": res.get("reason"),
        "raw_output": res.get("raw_output"),
        "n_screenshots": n_shots,
        "judged_screenshots": 0 if png is None else 1,
        "usage": res.get("usage"),
        "cost": res.get("cost"),
    }
    return verdict


def write_verdict(example_result_dir, verdict):
    out = os.path.join(example_result_dir, "agentic_verdict.json")
    try:
        with open(out, "w") as f:
            json.dump(verdict, f, indent=1, ensure_ascii=False)
    except Exception as e:
        print(f"[verify_episode] FAILED to write {out}: {e}")


def verify_episode(env, instr, example_result_dir, **kw):
    verdict = judge(instr, example_result_dir, DigiRLTerraEvaluator())
    write_verdict(example_result_dir, verdict)
    return verdict
