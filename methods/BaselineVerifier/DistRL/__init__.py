import json
import os

from ..DigiRL import _final_screenshot
from .DistRL_Terra import DistRLTerraEvaluator


def judge(instr, example_result_dir, evaluator):
    png, n_shots = _final_screenshot(example_result_dir)

    if png is None:
        res = {"success": None, "score": None, "safety": None, "status": None,
               "fewshot_set": None, "reason": "no screenshot on disk", "raw_output": "",
               "usage": [], "cost": None}
    else:
        res = evaluator.evaluate(instr, png)

    success = res.get("success")
    verdict = {
        "completion": None if success is None else ("SUCCESS" if success else "FAIL"),
        "safety": res.get("safety"),
        "score": res.get("score"),
        "status": res.get("status"),
        "fewshot_set": res.get("fewshot_set"),
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
    verdict = judge(instr, example_result_dir, DistRLTerraEvaluator())
    write_verdict(example_result_dir, verdict)
    return verdict
