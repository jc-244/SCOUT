#!/usr/bin/env python3

import json
import os
import time
import fcntl
from pathlib import Path
from typing import Dict, Any, Optional


def extract_domain_from_path(result_path: str) -> str:
    path_parts = Path(result_path).parts
    if len(path_parts) >= 2:
        return path_parts[-2]
    return "unknown"


def append_task_result(
    task_id: str,
    domain: str, 
    score: float,
    result_dir: str,
    args: Any,
    error_message: Optional[str] = None
) -> None:
    result_entry = {
        "application": domain,
        "task_id": task_id,
        "status": "error" if error_message else "success",
        "score": score,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if error_message:
        result_entry["err_message"] = error_message
    
    base_result_dir = Path(args.result_dir)
    summary_dir = base_result_dir / "summary"
    results_file = summary_dir / "results.json"
    
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(results_file, 'a+') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            
            try:
                f.seek(0)
                content = f.read().strip()
                
                if content:
                    try:
                        existing_results = json.loads(content)
                        if not isinstance(existing_results, list):
                            existing_results = []
                    except json.JSONDecodeError:
                        existing_results = []
                else:
                    existing_results = []
                
                existing_results.append(result_entry)
                
                f.seek(0)
                f.truncate()
                json.dump(existing_results, f, indent=2)
                f.write('\n')
                
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
        print(f"📝 Logged result: {domain}/{task_id} -> {result_entry['status']} (score: {score})")
        
    except Exception as e:
        print(f"⚠️  Failed to log result for {task_id}: {e}")


def log_task_completion(example: Dict, result: float, result_dir: str, args: Any) -> None:
    task_id = example.get('id', 'unknown')
    domain = extract_domain_from_path(result_dir)
    append_task_result(task_id, domain, result, result_dir, args)


def log_task_error(example: Dict, error_msg: str, result_dir: str, args: Any) -> None:
    task_id = example.get('id', 'unknown')
    domain = extract_domain_from_path(result_dir) 
    append_task_result(task_id, domain, 0.0, result_dir, args, error_msg)