import datetime
import json
import logging
import os
import time
from lib_results_logger import log_task_completion

logger = logging.getLogger("desktopenv.experiment")


def write_execution_metadata(example, args, example_result_dir):
    metadata = example.get("robustness_benchmark_metadata")
    if not metadata:
        return

    metadata_file = os.path.join(example_result_dir, "execution_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump({
            "example_id": metadata.get("example_id", example.get("id")),
            "task_id": metadata.get("task_id", example.get("id")),
            "domain": metadata.get("domain"),
            "actual_domain": metadata.get("actual_domain"),
            "severity": metadata.get("severity"),
            "agent_model": getattr(args, "model", None),
            "perturbed_instruction": example.get("instruction"),
            "execution_timestamp": datetime.datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)


def run_single_example(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    write_execution_metadata(example, args, example_result_dir)

    env.reset(task_config=example)

    try:
        agent.reset(runtime_logger)
    except Exception:
        agent.reset()

    time.sleep(60)
    obs = env._get_obs()

    try:
        with open(os.path.join(example_result_dir, "initial_state.png"), "wb") as _f:
            _f.write(obs['screenshot'])
    except Exception as _e:
        logger.error(f"failed to save initial_state.png: {_e}")

    done = False
    step_idx = 0
    env.controller.start_recording()
    while not done and step_idx < max_steps:
        response, actions = agent.predict(
            instruction,
            obs
        )
        if not actions:
            logger.warning("Agent returned no actions at step %d; ending episode so evaluate+verify can run. Response: %s",
                           step_idx + 1, str(response)[:150])
            break
        for action in actions:
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])
            traj_entry = {
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
            }
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps(traj_entry))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    time.sleep(20)
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

    try:
        _mod = os.getenv("VERIFIER_MODULE",
                         "methods.scout_verifier.GPTBedrockVerifier")
        verify_episode = __import__(_mod, fromlist=["verify_episode"]).verify_episode
        v = verify_episode(env, instruction, example_result_dir)
        logger.info("Agentic verifier: SAFETY=%s COMPLETION=%s cmds=%s usd=%s (oracle result=%.2f)",
                    v.get("safety"), v.get("completion"), v.get("n_cmds"),
                    (v.get("cost") or {}).get("usd"), result)
    except Exception as _e:
        logger.error("Agentic verifier failed: %s", _e)

    log_task_completion(example, result, example_result_dir, args)
    
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


def setup_logger(example, example_result_dir):
    runtime_logger = logging.getLogger(f"desktopenv.example.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(logging.FileHandler(os.path.join(example_result_dir, "runtime.log")))
    return runtime_logger
