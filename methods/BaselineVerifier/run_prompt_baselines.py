import argparse
import glob
import importlib
import json
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_instruction(ep_dir, tasks_dir=None):
    if tasks_dir is None:
        p = os.path.join(ep_dir, "execution_metadata.json")
        d = json.load(open(p))
        s = d.get("perturbed_instruction") or d.get("instruction")
    else:
        p = os.path.join(tasks_dir, os.path.basename(os.path.dirname(ep_dir)),
                         os.path.basename(ep_dir) + ".json")
        s = json.load(open(p)).get("instruction")
    if not s:
        raise ValueError(f"no instruction in {p}")
    return s


def mirror_episode(src, dst):
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name in ("agentic_verdict.json",):
            continue
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if not os.path.exists(d) and not os.path.islink(d):
            os.symlink(s, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--tasks", default=None,
                    help="OS-Blind task-config root; omit for AEB runs (execution_metadata.json)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    eps = sorted(os.path.dirname(p) for p in glob.glob(
        os.path.join(args.episodes, "**", "traj.jsonl"), recursive=True))
    print(f"{len(eps)} episodes under {args.episodes}", flush=True)

    arms = {a: importlib.import_module(f"methods.BaselineVerifier.{a}")
            for a in args.arms}

    jobs = []
    for src in eps:
        name = os.path.join(os.path.basename(os.path.dirname(src)), os.path.basename(src))
        for arm in arms:
            dst = os.path.join(args.out, arm, name)
            if not os.path.exists(os.path.join(dst, "agentic_verdict.json")):
                jobs.append((src, arm, dst, name))
    print(f"{len(jobs)} jobs to run on {args.workers} workers "
          f"({len(eps) * len(arms) - len(jobs)} already judged)", flush=True)

    done = [0]
    lock = threading.Lock()
    t0 = time.time()

    def run(job):
        src, arm, dst, name = job
        try:
            mirror_episode(src, dst)
            v = arms[arm].verify_episode(None, load_instruction(src, args.tasks), dst)
            line = (f"{arm:11} {name[-56:]} -> {v.get('completion')}/{v.get('safety')} "
                    f"${(v.get('cost') or {}).get('usd', 0)}")
        except Exception:
            line = f"{arm:11} {name[-56:]} -> ERROR"
            traceback.print_exc()
        with lock:
            done[0] += 1
            print(f"[{done[0]}/{len(jobs)}] {line}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(run, j) for j in jobs]):
            f.result()
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
