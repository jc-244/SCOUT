import json
import os

from .ZeroGUI_Terra import ZeroGUITerraEvaluator, load_initial_frame, load_trajectory


def judge(instr, example_result_dir, evaluator):
    task_config = {"instruction": instr, "id": os.path.basename(example_result_dir)}
    trajectory = load_trajectory(example_result_dir)
    outputs, res = evaluator.judge(task_config, trajectory, load_initial_frame(example_result_dir))

    votes = outputs.get("llm_output") or []
    if isinstance(votes, str):
        votes = [votes]
    success = res.get("success")
    return {
        "completion": None if success is None else ("SUCCESS" if success else "FAIL"),
        "safety": res.get("safety"),
        "score": res.get("score"),
        "parse_method": res.get("parse_method"),
        "eval_steps": 1,
        "n_cmds": 1,
        "reason": res.get("reason"),
        "raw_output": res.get("raw_output"),
        "votes": votes,
        "voting_rewards": res.get("voting_rewards"),
        "safety_votes": res.get("safety_votes"),
        "n_screenshots": res.get("n_screenshots"),
        "usage": outputs.get("usage"),
        "cost": outputs.get("cost"),
    }


def write_verdict(example_result_dir, verdict):
    out = os.path.join(example_result_dir, "agentic_verdict.json")
    try:
        with open(out, "w") as f:
            json.dump(verdict, f, indent=1, ensure_ascii=False)
    except Exception as e:
        print(f"[verify_episode] FAILED to write {out}: {e}")


def verify_episode(env, instr, example_result_dir, **kw):
    verdict = judge(instr, example_result_dir,
                    ZeroGUITerraEvaluator(model=os.getenv("VERIF_MODEL", "openai.gpt-5.6-terra")))
    write_verdict(example_result_dir, verdict)
    return verdict
