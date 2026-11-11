import json
import os

from . import LOG_SUBDIR, to_verdict
from .IRA_Terra import IRATerra

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXAMPLES_DIR = os.path.join(_REPO, "OSBlind", "evaluation_example_safety", "examples")
TOOL_APP = "all"


def load_task(example_result_dir):
    example_result_dir = os.path.normpath(example_result_dir)
    example_id = os.path.basename(example_result_dir)
    category = os.path.basename(os.path.dirname(example_result_dir))
    apps = None
    p = os.path.join(EXAMPLES_DIR, category, example_id + ".json")
    if os.path.exists(p):
        try:
            apps = json.load(open(p)).get("related_apps")
        except Exception as e:
            print(f"[IRA-OSB] failed to read task config {p}: {e}")
    else:
        print(f"[IRA-OSB] task config not found: {p}")
    return category, list(apps) if apps else [category]


def verify_episode(env, instr, example_result_dir, **kw):
    category, apps = load_task(example_result_dir)
    initial = os.path.join(example_result_dir, "initial_state.png")
    log_dir = os.path.join(example_result_dir, LOG_SUBDIR)
    os.makedirs(log_dir, exist_ok=True)

    agent = IRATerra(env, app=TOOL_APP)
    res = agent.evaluate(
        task_instruction=instr,
        apps=apps,
        output_dir=log_dir,
        initial_state_screenshot_path=initial if os.path.exists(initial) else None,
    )
    verdict = to_verdict(res, agent.usage_log, category=category, apps=apps, app=TOOL_APP,
                         cost=agent.cost)

    out = os.path.join(example_result_dir, "agentic_verdict.json")
    try:
        with open(out, "w") as f:
            json.dump(verdict, f, indent=1, ensure_ascii=False)
    except Exception as e:
        print(f"[verify_episode] FAILED to write {out}: {e}")
    return verdict
