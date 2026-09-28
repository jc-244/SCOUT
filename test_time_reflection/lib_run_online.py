import datetime
import json
import logging
import os
import time

from lib_results_logger import log_task_completion
from test_time_reflection.lib_run_reflexion import setup_and_reset
from test_time_reflection.QwenOnlineVerifier import QwenOnlineVerifier, needs_gate

logger = logging.getLogger("desktopenv.experiment")

GATE_MAX_ROUNDS = int(os.getenv("GATE_MAX_ROUNDS", "5"))


def _append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_verdict(ep_dir):
    try:
        return json.load(open(os.path.join(ep_dir, "agentic_verdict.json")))
    except Exception:
        return {}


def hand_note_to_agent(agent, note):
    notes = getattr(agent, "gate_notes", None)
    if isinstance(notes, list) and notes:
        notes[-1] = note
        return True
    logger.warning("[online] agent has no gate_notes hook; block not fed back")
    return False


def run_online_example(agent, env, example, max_steps, instruction, args, ep_dir, scores,
                       gate_cls=QwenOnlineVerifier):
    obs = setup_and_reset(env, example, agent, ep_dir, args, write_initial=True)

    gate = gate_cls(env, instruction, ep_dir, max_rounds=GATE_MAX_ROUNDS)

    gate_log = os.path.join(ep_dir, "gate_log.jsonl")
    recs, turn, n_exec, n_blocked, done = [], 0, 0, 0, False
    env.controller.start_recording()
    while not done and turn < max_steps:
        turn += 1
        response, actions = agent.predict(instruction, obs)
        if not actions:
            logger.warning("Agent returned no actions at step %d; ending episode so "
                           "evaluate+verify can run. Response: %s", turn, str(response)[:150])
            break

        if needs_gate(actions):
            g = gate.judge(turn, recs, response, actions, obs["screenshot"])
        else:
            g = {"step_num": turn, "decision": "ALLOW", "reason": "terminal marker, not gated",
                 "verdict_found": None, "note": None, "pending_actions": list(actions),
                 "response": response, "checklist": [], "checklist_status": "skipped",
                 "checklist_raws": [], "per_item": {}, "extra_unsafe": False,
                 "responses": [], "transcript": [], "rounds": 0, "reasoning": [], "usage": []}
        _append_jsonl(gate_log, dict(g, turn=turn))
        logger.info("[online] turn %d gate=%s checklist=%d rounds=%d probes=%d %s", turn,
                    g["decision"], len(g["checklist"]), g["rounds"], len(g["transcript"]),
                    (g["reason"] or "")[:120])

        if g["decision"] == "BLOCK":
            n_blocked += 1
            hand_note_to_agent(agent, g["note"])
            continue

        for action in actions:
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            logger.info("Step %d: %s", turn, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)
            n_exec += 1
            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            with open(os.path.join(ep_dir, f"step_{turn}_{action_timestamp}.png"), "wb") as _f:
                _f.write(obs["screenshot"])
            traj_entry = {
                "step_num": turn,
                "action_timestamp": action_timestamp,
                "action": action,
                "response": response,
                "reward": reward,
                "done": done,
                "info": info,
                "screenshot_file": f"step_{turn}_{action_timestamp}.png",
            }
            _append_jsonl(os.path.join(ep_dir, "traj.jsonl"), traj_entry)
            recs.append(traj_entry)
            if done:
                logger.info("The episode is done.")
                break

    time.sleep(20)
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    with open(os.path.join(ep_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

    try:
        _mod = os.getenv("VERIFIER_MODULE",
                         "scout_verifier.GPTBedrockVerifier")
        verify_episode = __import__(_mod, fromlist=["verify_episode"]).verify_episode
        v = verify_episode(env, instruction, ep_dir)
        logger.info("Agentic verifier: SAFETY=%s COMPLETION=%s cmds=%s usd=%s (oracle result=%.2f)",
                    v.get("safety"), v.get("completion"), v.get("n_cmds"),
                    (v.get("cost") or {}).get("usd"), result)
    except Exception as _e:
        logger.error("Agentic verifier failed: %s", _e)

    env.controller.end_recording(os.path.join(ep_dir, "recording.mp4"))

    verdict = _read_verdict(ep_dir)
    with open(os.path.join(ep_dir, "gate_summary.json"), "w") as f:
        json.dump({"turns": turn, "executed_actions": n_exec, "blocked_turns": n_blocked,
                   "max_steps": max_steps, "result": result,
                   "safety": verdict.get("safety"), "completion": verdict.get("completion")},
                  f, indent=2)
    scores.append(result if result is not None else 0.0)
    log_task_completion(example, result or 0.0, ep_dir, args)
    return turn, result
