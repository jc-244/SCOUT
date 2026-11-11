from __future__ import annotations

import os
import json
import base64
import re
import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from smolagents import Tool

from RewardAgent.prompts.improved_system_prompt import build_improved_system_prompt
from RewardAgent.tools.computer_use import process_image, smart_resize


class ParsedToolCall:
    def __init__(self, name: str, arguments: Dict[str, Any]):
        self.id = f"parsed_{name}_{datetime.datetime.now().strftime('%H%M%S')}"
        self.function = self.FunctionCall(name, arguments)

    def model_dump(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments
            }
        }

    class FunctionCall:
        def __init__(self, name: str, arguments: Dict[str, Any]):
            self.name = name
            self.arguments = json.dumps(arguments) if isinstance(arguments, dict) else arguments


class MessageWrapper:
    def __init__(self, content: str, tool_calls: List[ParsedToolCall], reasoning: str, finish_reason: str = "stop"):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning = reasoning
        self.finish_reason = finish_reason


class RewardAgentImproved:

    CLIENT_TYPE_CONFIGS = {
        "deerapi": {
            "base_url": "https://api.cometapi.com/v1/",
            "env_key": "deerapi_key",
        },
        "vllm": {
            "base_url": "http://0.0.0.0:8100/v1",
            "env_key": None,
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "env_key": "OPENAI_API_KEY",
        },
    }

    def __init__(
        self,
        model_id: str,
        env: Any,
        app: str = "all",
        max_images: int = 5,
        max_steps: int = 30,
        system_prompt: Optional[str] = None,
        client: Optional[OpenAI] = None,
        client_config: Optional[Dict[str, Any]] = None,
        client_type: Optional[str] = 'deerapi',
        temperature: Optional[float] = 0.0,
    ):
        if client is not None:
            self.client = client
            self._client_type = "passed"
        elif client_config is not None:
            self.client = self._init_client_from_config(client_config)
            self._client_type = client_config.get("type", "config")
        elif client_type is not None:
            self.client = self._init_client_from_type(client_type)
            self._client_type = client_type
        else:
            self.client = self._init_default_client()
            self._client_type = "vllm_default"

        self._base_url = getattr(self.client, 'base_url', None)
        self._api_key = getattr(self.client, 'api_key', 'EMPTY')

        self.model_id = model_id
        self.env = env
        self.max_images = max_images
        self.max_steps = max_steps
        self.temperature = temperature
        self.app = app.lower() if app else "all"

        self.tools = self._build_tools(env, self.app)
        self._tools_map = {t.name: t for t in self.tools}

        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = build_improved_system_prompt(self.tools)

        self.messages: List[Dict[str, Any]] = []
        self._image_count = 0

        self.log_history: List[Dict[str, Any]] = []

        self.output_dir: Optional[str] = None

        self.step_counter = 0
        self.steps_log: List[Dict[str, Any]] = []
        self.llm_call_counter = 0

    def _init_client_from_config(self, config: Dict[str, Any]) -> OpenAI:
        api_key = config.get("api_key")
        base_url = config.get("base_url")
        type_config = self.CLIENT_TYPE_CONFIGS.get(config.get("type"), {})

        if not api_key:
            for env_name in (config.get("env_key"), "IRA_API_KEY", "OPENAI_API_KEY", type_config.get("env_key")):
                if env_name and os.getenv(env_name):
                    api_key = os.getenv(env_name)
                    break
            if not api_key:
                if base_url and ("localhost" in base_url or "127.0.0.1" in base_url):
                    api_key = "EMPTY"
                else:
                    raise ValueError(
                        "API key not found. Set IRA_API_KEY (or the env var named by "
                        "'api_key_env' in the config) in the environment or .env file."
                    )

        if not base_url:
            base_url = (
                os.getenv("IRA_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or type_config.get("base_url")
            )
            if not base_url:
                raise ValueError(
                    "No base_url configured. Set IRA_BASE_URL in the environment or "
                    "provide 'base_url' in the client config."
                )

        return OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def _init_client_from_type(self, client_type: str) -> OpenAI:
        if client_type not in self.CLIENT_TYPE_CONFIGS:
            raise ValueError(
                f"Unknown client_type '{client_type}'. "
                f"Supported types: {list(self.CLIENT_TYPE_CONFIGS.keys())}"
            )

        type_config = self.CLIENT_TYPE_CONFIGS[client_type]
        base_url = os.getenv("IRA_BASE_URL") or os.getenv("OPENAI_BASE_URL") or type_config["base_url"]
        env_key = type_config.get("env_key")

        if env_key:
            api_key = os.getenv(env_key) or os.getenv("IRA_API_KEY")
            if not api_key:
                raise ValueError(
                    f"No API key found for client_type '{client_type}'. "
                    f"Set IRA_API_KEY or '{env_key}' in the environment or .env file."
                )
        else:
            api_key = os.getenv("IRA_API_KEY") or "EMPTY"

        return OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def _init_default_client(self) -> OpenAI:
        vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8100/v1")

        print(f"[RewardAgentImproved] Using default vLLM client at {vllm_base_url}")

        return OpenAI(
            api_key="EMPTY",
            base_url=vllm_base_url,
        )
    
    def evaluate(
        self,
        task_instruction: str,
        apps: List[str],
        output_dir: Optional[str] = None,
        initial_state_screenshot_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.output_dir = output_dir
        
        if output_dir:
            self._propagate_output_dir(output_dir)
        
        self.log_history = []
        self.step_counter = 0
        self.steps_log = []
        self.llm_call_counter = 0

        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        task_prompt = self._build_task_prompt(task_instruction, apps)
        
        initial_state_b64 = self._load_screenshot_from_path(initial_state_screenshot_path)
        current_state_b64 = self._get_current_screenshot()

        first_message_content: List[Dict[str, Any]] = [
            {"type": "text", "text": task_prompt}
        ]
        if initial_state_b64:
            first_message_content.append({
                "type": "text",
                "text": "Image A: Initial state after env.reset and before replay."
            })
            first_message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{initial_state_b64}"}
            })

        first_message_content.append({
            "type": "text",
            "text": "Image B: Current state after replay (final state before evaluation actions)."
        })
        first_message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{current_state_b64}"}
        })

        self.messages.append({
            "role": "user",
            "content": first_message_content,
        })
        self._image_count = (1 if initial_state_b64 else 0) + 1

        self.log_history.append({
            "step": 0,
            "type": "initial",
            "timestamp": datetime.datetime.now().isoformat(),
            "thought": task_prompt,
            "action": None,
            "observation": None,
            "screenshot_before_b64": initial_state_b64,
            "screenshot_after_b64": current_state_b64,
            "annotated_b64": None,
        })
        
        result = None
        error_occurred = None

        invalid_response_count = 0
        max_invalid_responses = 3

        self.llm_call_counter = 0

        try:
            while self.step_counter < self.max_steps:
                try:
                    self.llm_call_counter += 1

                    message = self._call_llm()

                    if message is None:
                        invalid_response_count += 1
                        print(f"[RewardAgentImproved] Null response from model (count: {invalid_response_count})")
                        full_request_context = self._get_full_request_for_debug()
                        response_raw = {
                            "content": None,
                            "tool_calls": None,
                            "finish_reason": "null_response",
                        }
                        if output_dir:
                            self._save_debug_payload(full_request_context, response_raw, output_dir, self.llm_call_counter)
                        self.log_history.append({
                            "step": self.llm_call_counter,
                            "type": "invalid",
                            "invalid_reason": "null_response",
                            "timestamp": datetime.datetime.now().isoformat(),
                            "thought": None,
                            "action": None,
                            "observation": "Model returned null response",
                            "screenshots": {},
                            "debug_info": {
                                "llm_request_context_summary": self._get_request_context_for_debug(),
                                "llm_request_full": full_request_context,
                                "llm_response_raw": response_raw,
                            },
                        })
                        if invalid_response_count >= max_invalid_responses:
                            print(f"[RewardAgentImproved] Too many invalid responses, aborting evaluation")
                            error_occurred = "model service not available"
                            break
                        continue

                    if not message.tool_calls:
                        thought = message.content or ""

                        if thought and thought.strip():
                            invalid_response_count += 1
                            print(f"[RewardAgentImproved] Text-only response without tool_calls (count: {invalid_response_count})")
                            print(f"[RewardAgentImproved] Model text: {thought[:200]}...")


                            full_request_context = self._get_full_request_for_debug()
                            response_raw = {
                                "content": thought,
                                "tool_calls": None,
                                "finish_reason": getattr(message, 'finish_reason', 'unknown'),
                            }
                            if output_dir:
                                self._save_debug_payload(full_request_context, response_raw, output_dir, self.llm_call_counter)

                            self.log_history.append({
                                "step": self.llm_call_counter,
                                "type": "invalid",
                                "invalid_reason": "text_without_tool_calls",
                                "timestamp": datetime.datetime.now().isoformat(),
                                "thought": thought,
                                "action": None,
                                "observation": "Model returned text without calling any tool",
                                "screenshots": {},
                                "debug_info": {
                                    "llm_request_context_summary": self._get_request_context_for_debug(),
                                    "llm_request_full": full_request_context,
                                    "llm_response_raw": response_raw,
                                },
                            })

                            if invalid_response_count >= max_invalid_responses:
                                print(f"[RewardAgentImproved] Too many text-only responses, aborting evaluation")
                                error_occurred = "model service not available"
                                break
                            continue
                        else:
                            invalid_response_count += 1
                            print(f"[RewardAgentImproved] Empty response without tool_calls (count: {invalid_response_count})")
                            full_request_context = self._get_full_request_for_debug()
                            response_raw = {
                                "content": "",
                                "tool_calls": None,
                                "finish_reason": getattr(message, 'finish_reason', 'unknown'),
                            }
                            if output_dir:
                                self._save_debug_payload(full_request_context, response_raw, output_dir, self.llm_call_counter)
                            self.log_history.append({
                                "step": self.llm_call_counter,
                                "type": "invalid",
                                "invalid_reason": "empty_response",
                                "timestamp": datetime.datetime.now().isoformat(),
                                "thought": None,
                                "action": None,
                                "observation": "Model returned empty response without tool_calls",
                                "screenshots": {},
                                "debug_info": {
                                    "llm_request_context_summary": self._get_request_context_for_debug(),
                                    "llm_request_full": full_request_context,
                                    "llm_response_raw": response_raw,
                                },
                            })
                            if invalid_response_count >= max_invalid_responses:
                                print(f"[RewardAgentImproved] Too many empty responses, aborting evaluation")
                                error_occurred = "model service not available"
                                break
                            continue

                    invalid_response_count = 0

                    result = self._handle_response(message)

                    if result is not None:
                        result["steps"] = self.step_counter
                        result["invalid_steps"] = invalid_response_count
                        result["total_llm_calls"] = self.llm_call_counter
                        if output_dir:
                            try:
                                self._save_improved_agent_log(
                                    output_dir=output_dir,
                                    task_instruction=task_instruction,
                                    apps=apps,
                                    evaluation=result
                                )
                            except Exception as e:
                                print(f"[RewardAgentImproved] Failed to save agent log: {e}")
                        return result

                except Exception as e:
                    print(f"[RewardAgentImproved] Error at LLM call {self.llm_call_counter}: {e}")
                    import traceback
                    traceback.print_exc()
                    error_occurred = str(e)
                    invalid_response_count += 1
                    full_request_context = self._get_full_request_for_debug()
                    response_raw = {
                        "content": None,
                        "tool_calls": None,
                        "finish_reason": "exception",
                        "exception": str(e),
                    }
                    if output_dir:
                        self._save_debug_payload(full_request_context, response_raw, output_dir, self.llm_call_counter)
                    self.log_history.append({
                        "step": self.llm_call_counter,
                        "type": "invalid",
                        "invalid_reason": "exception",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "thought": None,
                        "action": None,
                        "observation": f"Exception: {str(e)}",
                        "screenshots": {},
                        "debug_info": {
                            "llm_request_context_summary": self._get_request_context_for_debug(),
                            "llm_request_full": full_request_context,
                            "llm_response_raw": response_raw,
                            "exception_traceback": traceback.format_exc(),
                        },
                    })
                    if invalid_response_count >= max_invalid_responses:
                        print(f"[RewardAgentImproved] Too many errors, aborting evaluation")
                        break
                    continue

            if result is None:
                if error_occurred == "model service not available" or invalid_response_count >= max_invalid_responses:
                    result = {
                        "reward": 0.0,
                        "verdict": "Failure",
                        "reasoning": "Model service not available - received too many responses without tool_calls",
                        "steps": self.step_counter,
                        "invalid_steps": invalid_response_count,
                        "total_llm_calls": self.llm_call_counter,
                    }
                elif self.step_counter >= self.max_steps:
                    result = {
                        "reward": 0.0,
                        "verdict": "Failure",
                        "reasoning": "Maximum evaluation steps reached without final answer",
                        "steps": self.step_counter,
                        "invalid_steps": invalid_response_count,
                        "total_llm_calls": self.llm_call_counter,
                    }
                else:
                    result = {
                        "reward": 0.0,
                        "verdict": "Failure",
                        "reasoning": f"Evaluation stopped: {error_occurred}" if error_occurred else "Evaluation incomplete",
                        "steps": self.step_counter,
                        "invalid_steps": invalid_response_count,
                        "total_llm_calls": self.llm_call_counter,
                    }

        finally:
            if output_dir:
                try:
                    if result is None:
                        result = {
                            "reward": 0.0,
                            "verdict": "Failure",
                            "reasoning": f"Evaluation failed with error: {error_occurred}" if error_occurred else "Evaluation incomplete",
                            "steps": self.step_counter,
                            "invalid_steps": invalid_response_count,
                            "total_llm_calls": self.llm_call_counter,
                        }

                    self._save_improved_agent_log(
                        output_dir=output_dir,
                        task_instruction=task_instruction,
                        apps=apps,
                        evaluation=result
                    )
                except Exception as e:
                    print(f"[RewardAgentImproved] Failed to save agent log in finally block: {e}")
                    import traceback
                    traceback.print_exc()

        return result
    
    def _build_tools(self, env: Any, app: str) -> List[Tool]:
        tools: List[Tool] = []
        
        from RewardAgent.tools.final_answer import FinalAnswerTool
        from RewardAgent.tools.CaptionImage import CaptionImageTool
        from RewardAgent.tools.environment_tools import (
            VMCommandLineTool,
            VMCommandErrorTool,
            VMFileTool,
            VMTerminalOutputTool,
        )
        from RewardAgent.tools.file_getters import GetHostFileContentTool
        from RewardAgent.tools.app_getters import GetAccessibilityTreeTool
        
        tools.append(FinalAnswerTool())
        tools.extend([
            VMCommandLineTool(env),
            VMCommandErrorTool(env),
            VMFileTool(env),
            VMTerminalOutputTool(env),
            GetHostFileContentTool(),
            GetAccessibilityTreeTool(env),
        ])
        
        if app == "all" or app == "multi_apps":
            self._add_all_app_tools(tools, env)
        else:
            self._add_specific_app_tools(tools, env, app)
        
        return tools
    
    def _add_all_app_tools(self, tools: List[Tool], env: Any):
        try:
            from RewardAgent.tools.chrome_getters import (
                GetActiveTabInfoTool, GetDefaultSearchEngineTool,
                GetCookieDataTool, GetBookmarksTool, GetOpenTabsInfoTool,
                GetBrowserHistoryTool, GetPageInfoTool, GetChromeLanguageTool,
                GetChromeFontSizeTool
            )
            from RewardAgent.tools.environment_tools import GetActiveURLTool
            tools.extend([
                GetActiveTabInfoTool(env), GetDefaultSearchEngineTool(env),
                GetCookieDataTool(env), GetBookmarksTool(env),
                GetOpenTabsInfoTool(env), GetBrowserHistoryTool(env),
                GetPageInfoTool(env), GetChromeLanguageTool(env),
                GetChromeFontSizeTool(env), GetActiveURLTool(env)
            ])
        except Exception:
            pass
        
        try:
            from RewardAgent.tools.vscode_getters import (
                GetVSCodeUserSettingsFileTool, GetVSCodeKeybindingsFileTool
            )
            tools.extend([
                GetVSCodeUserSettingsFileTool(env),
                GetVSCodeKeybindingsFileTool(env)
            ])
        except Exception:
            pass
        
        self._add_document_tools(tools, env)
    
    def _add_specific_app_tools(self, tools: List[Tool], env: Any, app: str):
        return
    
    def _add_document_tools(self, tools: List[Tool], env: Any):
        try:
            from RewardAgent.tools.CheckPptFile import CheckPptFileTool
            from RewardAgent.tools.GetPptXml import GetPptXmlTool
            from RewardAgent.tools.CheckWordFile import CheckWordFileTool
            from RewardAgent.tools.CheckExcelFile import CheckExcelFileTool
            tools.extend([
                CheckPptFileTool(),
                GetPptXmlTool(),
                CheckWordFileTool(),
                CheckExcelFileTool()
            ])
        except Exception:
            pass
    
    def _propagate_output_dir(self, output_dir: str):
        for tool in self.tools:
            tool_name = getattr(tool, "name", "")
            if tool_name in ("checkexcelfile", "checkpptfile", "checkwordfile"):
                if hasattr(tool, "set_output_dir"):
                    tool.set_output_dir(output_dir)
    
    def _build_task_prompt(self, task_instruction: str, apps: List[str]) -> str:
        return (
            f"# Task Evaluation Request\n\n"
            f"**Task Instruction**: {task_instruction}\n\n"
            f"**Related apps**: {apps}\n\n"
            f"## Your Mission\n"
            f"Evaluate whether the task described above has been successfully completed. You are judging the result of another model's attempt, not planning how to do the task.\n\n"
            f"## Available Visual Evidence\n"
            f"- Image A is the state at the beginning of the task.\n"
            f"- Image B is the end state after another model attempted to complete the task.\n"
            f"- These attached screenshots are already primary evidence. Do NOT default to recaptioning the same screen.\n"
            f"- After any `computer` action, a fresh screenshot will be attached automatically and should be used directly as the new visual evidence.\n\n"
            f"## Required Internal Reasoning Order\n"
            f"1. Summarize the expected successful end state implied by the task instruction.\n"
            f"2. Compare the task-start state and the attempted end state to identify what actually changed.\n"
            f"3. Identify the exact object(s) the task refers to, and keep your verification anchored to those objects before checking related artifacts.\n"
            f"4. Separate the completion criteria into:\n"
            f"   - visible criteria already checkable from the attached screenshots,\n"
            f"   - hidden/system/file-backed criteria that require stronger verification,\n"
            f"   - uncertain criteria that may require UI exploration.\n"
            f"5. For each unresolved criterion, follow a stable escalation path: visual evidence → specialized tool/getter → command/file/config verification → computer interaction.\n"
            f"6. Do not require proof of the exact action provenance unless the task explicitly asks for it; verify the required end state itself.\n"
            f"7. Only conclude Success if all critical criteria are verified or strongly evidenced.\n\n"
            f"## Verification Policy\n"
            f"- Use the attached screenshots first for visible UI judgments.\n"
            f"- Then use the most relevant specialized tool or getter when it can directly reveal app state, browser state, document state, tab state, or structured UI evidence.\n"
            f"- If the task may depend on preferences, config files, terminal state, URLs, filesystem outputs, document contents, or other hidden state, prefer command/file/config verification over screenshot-only judgment.\n"
            f"- If the required evidence likely exists in the GUI but is not currently visible, or if higher-confidence checks still leave the answer unresolved, use `computer` to reveal it.\n"
            f"- For visually ambiguous properties such as alignment, formatting, selection state, or panel state, do not rely on appearance alone if a stronger confirmation path exists.\n"
            f"- If a critical criterion remains unverified, do not overclaim success.\n"
            f"- If further actions would mostly repeat low-yield exploration, stop exploring and treat that criterion as unverified or not satisfied.\n"
            f"- However, do not stop early while a clear GUI inspection path still exists that could reveal the answer.\n"
        )

    def _load_screenshot_from_path(self, image_path: Optional[str]) -> str:
        if not image_path:
            return ""
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            processed_b64, _ = process_image(image_bytes)
            return processed_b64
        except Exception as e:
            print(f"[RewardAgentImproved] Failed to load screenshot from path {image_path}: {e}")
            return ""
    
    def _get_current_screenshot(self) -> str:
        try:
            obs = self.env._get_obs() if hasattr(self.env, "_get_obs") else {
                "screenshot": self.env.controller.get_screenshot()
            }
            screenshot_bytes = obs["screenshot"]
            
            processed_b64, _ = process_image(screenshot_bytes)
            return processed_b64
        except Exception as e:
            print(f"[RewardAgentImproved] Failed to get screenshot: {e}")
            return ""

    def _get_request_context_for_debug(self) -> Dict[str, Any]:
        try:
            role_counts = {}
            for msg in self.messages:
                role = msg.get("role", "unknown")
                role_counts[role] = role_counts.get(role, 0) + 1

            last_msg_summary = None
            if self.messages:
                last_msg = self.messages[-1]
                last_msg_summary = {
                    "role": last_msg.get("role"),
                    "content_type": "list" if isinstance(last_msg.get("content"), list) else "string",
                }
                content = last_msg.get("content")
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            text_parts.append(text[:200] + "..." if len(text) > 200 else text)
                    last_msg_summary["text_preview"] = text_parts[:2]
                elif isinstance(content, str):
                    last_msg_summary["text_preview"] = content[:200] + "..." if len(content) > 200 else content

            return {
                "model_id": self.model_id,
                "total_messages": len(self.messages),
                "role_counts": role_counts,
                "image_count": self._image_count,
                "last_message_summary": last_msg_summary,
                "step_counter": self.step_counter,
                "llm_call_counter": self.llm_call_counter,
            }
        except Exception as e:
            return {
                "error": f"Failed to capture request context: {e}",
                "model_id": self.model_id,
            }

    def _get_full_request_for_debug(self) -> Dict[str, Any]:
        try:
            tool_schemas = self._build_tool_schemas()


            safe_messages = []
            full_messages = []

            for msg in self.messages:
                safe_msg = {"role": msg.get("role")}
                full_msg = {"role": msg.get("role")}

                content = msg.get("content")

                if msg.get("tool_calls"):
                    safe_msg["tool_calls"] = msg["tool_calls"]
                    full_msg["tool_calls"] = msg["tool_calls"]

                if msg.get("tool_call_id"):
                    safe_msg["tool_call_id"] = msg["tool_call_id"]
                    full_msg["tool_call_id"] = msg["tool_call_id"]

                if isinstance(content, list):
                    safe_content = []
                    full_content = []
                    for item in content:
                        if item.get("type") == "text":
                            safe_content.append(item)
                            full_content.append(item)
                        elif item.get("type") == "image_url":
                            image_url = item.get("image_url", {})
                            url_value = image_url.get("url", "")
                            if url_value.startswith("data:image/png;base64,"):
                                b64_data = url_value[len("data:image/png;base64,"):]
                                safe_content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"[IMAGE_BASE64_PLACEHOLDER: {len(b64_data)} chars]"
                                    }
                                })
                                full_content.append(item)
                            else:
                                safe_content.append(item)
                                full_content.append(item)
                        else:
                            safe_content.append(item)
                            full_content.append(item)
                    safe_msg["content"] = safe_content
                    full_msg["content"] = full_content
                elif isinstance(content, str):
                    safe_msg["content"] = content
                    full_msg["content"] = content
                else:
                    safe_msg["content"] = str(content)
                    full_msg["content"] = str(content)

                safe_messages.append(safe_msg)
                full_messages.append(full_msg)

            return {
                "model_id": self.model_id,
                "base_url": str(self._base_url) if self._base_url else "unknown",
                "tool_schemas": tool_schemas,
                "temperature": 0.0,
                "safe_messages": safe_messages,
                "full_messages": full_messages,
                "total_messages": len(self.messages),
                "image_count": self._image_count,
                "llm_call_counter": self.llm_call_counter,
            }
        except Exception as e:
            import traceback
            return {
                "error": f"Failed to capture full request: {e}",
                "error_traceback": traceback.format_exc(),
                "model_id": self.model_id,
            }

    def _save_debug_payload(self, request_context: Dict, response_raw: Dict, output_dir: str, llm_call_num: int) -> str:
        try:
            debug_dir = os.path.join(output_dir, "debug_logs")
            os.makedirs(debug_dir, exist_ok=True)

            debug_filename = f"llm_call_{llm_call_num}_debug.json"
            debug_path = os.path.join(debug_dir, debug_filename)

            debug_payload = {
                "timestamp": datetime.datetime.now().isoformat(),
                "llm_call_number": llm_call_num,
                "request": request_context,
                "response": response_raw,
                "instructions": {
                    "how_to_resend": "To manually test this LLM call, use the 'full_messages' field with actual base64 images.",
                    "api_format": "POST to base_url with model_id, messages, tools, max_tokens, temperature",
                    "python_example": """
import requests
import json

payload = {
    "model": request['model_id'],
    "messages": request['full_messages'],  # Contains actual base64 images
    "tools": request['tool_schemas'],
    "max_tokens": request['max_tokens'],
    "temperature": request['temperature']
}

response = requests.post(
    request['base_url'] + '/chat/completions',
    headers={'Authorization': 'Bearer YOUR_API_KEY', 'Content-Type': 'application/json'},
    json=payload
)
print(response.json())
"""
                }
            }

            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(debug_payload, f, ensure_ascii=False, indent=2)

            print(f"[RewardAgentImproved] Saved debug payload to: {debug_path}")
            return debug_path
        except Exception as e:
            print(f"[RewardAgentImproved] Failed to save debug payload: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _parse_tool_calls_from_reasoning(self, reasoning: str) -> List[ParsedToolCall]:
        tool_calls = []

        function_pattern = r'<function=([^>]+)>(.*?)</function>'
        function_matches = re.findall(function_pattern, reasoning, re.DOTALL)

        for func_name, func_body in function_matches:
            func_name = func_name.strip()
            args_dict = {}

            param_pattern = r'<parameter=([^>]+)>(.*?)</parameter>'
            param_matches = re.findall(param_pattern, func_body, re.DOTALL)

            for param_name, param_value in param_matches:
                param_name = param_name.strip()
                param_value = param_value.strip()

                if param_value.startswith('{') or param_value.startswith('['):
                    try:
                        param_value = json.loads(param_value)
                    except json.JSONDecodeError:
                        pass
                elif param_value.isdigit():
                    param_value = int(param_value)
                elif re.match(r'^-?\d+\.?\d*$', param_value):
                    try:
                        param_value = float(param_value)
                    except ValueError:
                        pass
                elif param_value.lower() == 'true':
                    param_value = True
                elif param_value.lower() == 'false':
                    param_value = False

                args_dict[param_name] = param_value

            if args_dict:
                tool_calls.append(ParsedToolCall(func_name, args_dict))
                print(f"[RewardAgentImproved] Parsed tool call from reasoning: {func_name} with args: {args_dict}")

        return tool_calls

    def _call_llm(self) -> Any:
        tool_schemas = self._build_tool_schemas()

        request_kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "messages": self.messages,
            "tools": tool_schemas if tool_schemas else None,
        }
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        response = self.client.chat.completions.create(**request_kwargs)

        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason

        message.finish_reason = finish_reason

        has_content = message.content and message.content.strip()
        has_tool_calls = message.tool_calls and len(message.tool_calls) > 0

        if not has_content and not has_tool_calls:
            reasoning_content = getattr(message, 'reasoning', None)
            if reasoning_content and reasoning_content.strip():
                print(f"[RewardAgentImproved] Found content in reasoning field (Qwen3 style)")
                print(f"[RewardAgentImproved] Reasoning content preview: {reasoning_content[:200]}...")

                parsed_tool_calls = self._parse_tool_calls_from_reasoning(reasoning_content)

                if parsed_tool_calls:
                    return MessageWrapper(
                        content=reasoning_content,
                        tool_calls=parsed_tool_calls,
                        reasoning=reasoning_content,
                        finish_reason=finish_reason
                    )
                else:
                    return MessageWrapper(
                        content=reasoning_content,
                        tool_calls=[],
                        reasoning=reasoning_content,
                        finish_reason=finish_reason
                    )

        return message
    
    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = []
        
        schemas.append({
            "type": "function",
            "function": {
                "name": "computer",
                "description": (
                    "Perform GUI actions: click, type, key press, scroll, mouse move, wait, etc. "
                    "Use relative coordinates (0-999 range) which will be scaled to screen resolution."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "left_click", "right_click", "double_click", "middle_click",
                                "type", "key", "scroll", "mouse_move", "left_click_drag",
                                "wait", "terminate"
                            ],
                            "description": "The GUI action to perform"
                        },
                        "coordinate": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "For mouse actions: [x, y] in relative coords (0-999)"
                        },
                        "start_coordinate": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "For left_click_drag: start point [x, y] in relative coords (0-999)"
                        },
                        "end_coordinate": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "For left_click_drag: end point [x, y] in relative coords (0-999)"
                        },
                        "duration": {
                            "type": "number",
                            "description": "For left_click_drag: drag duration in seconds"
                        },
                        "text": {
                            "type": "string",
                            "description": "For type action: text to type"
                        },
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "For key action: list of keys to press"
                        },
                        "pixels": {
                            "type": "number",
                            "description": "For scroll action: scroll amount in pixels"
                        },
                        "time": {
                            "type": "number",
                            "description": "For wait action: seconds to wait"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["success", "failure"],
                            "description": "For terminate action: task completion status"
                        }
                    },
                    "required": ["action"]
                }
            }
        })
        
        schemas.append({
            "type": "function",
            "function": {
                "name": "final_answer",
                "description": "Output final evaluation result with reward, verdict, and reasoning",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reward": {
                            "type": "number",
                            "description": "Reward score between 0.0 and 1.0"
                        },
                        "verdict": {
                            "type": "string",
                            "enum": ["Success", "Partial Success", "Failure"],
                            "description": "Overall verdict"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Detailed reasoning for the evaluation"
                        }
                    },
                    "required": ["reward", "verdict", "reasoning"]
                }
            }
        })
        
        for tool in self.tools:
            if tool.name in ("final_answer", "computer_use"):
                continue
            
            try:
                schema = self._tool_to_openai_schema(tool)
                if schema:
                    schemas.append(schema)
            except Exception:
                continue
        
        return schemas
    
    def _tool_to_openai_schema(self, tool: Tool) -> Optional[Dict[str, Any]]:
        try:
            properties = {}
            required = []
            
            if hasattr(tool, "inputs") and tool.inputs:
                for param_name, param_info in tool.inputs.items():
                    param_type = param_info.get("type", "string")
                    param_desc = param_info.get("description", "")
                    
                    type_map = {
                        "string": "string",
                        "text": "string",
                        "integer": "integer",
                        "number": "number",
                        "boolean": "boolean",
                        "image": "string",
                    }
                    
                    properties[param_name] = {
                        "type": type_map.get(param_type, "string"),
                        "description": param_desc
                    }
                    required.append(param_name)
            
            return {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            }
        except Exception:
            return None
    
    def _save_improved_agent_log(
        self,
        output_dir: str,
        task_instruction: str,
        apps: List[str],
        evaluation: Dict[str, Any],
    ) -> None:
        try:
            log_dir = os.path.join(output_dir, "improved_log")
            os.makedirs(log_dir, exist_ok=True)
            
            screenshots_dir = os.path.join(log_dir, "screenshots")
            os.makedirs(screenshots_dir, exist_ok=True)
            
            screenshot_count = 0
            for entry in self.log_history:
                step_num = entry.get("step", 0)
                
                if entry.get("screenshot_before_b64"):
                    screenshot_count += 1
                    filename = f"step_{step_num:03d}_before.png"
                    filepath = os.path.join(screenshots_dir, filename)
                    try:
                        img_bytes = base64.b64decode(entry["screenshot_before_b64"])
                        with open(filepath, "wb") as f:
                            f.write(img_bytes)
                    except Exception as e:
                        print(f"[RewardAgentImproved] Failed to save {filename}: {e}")
                
                if entry.get("annotated_b64"):
                    screenshot_count += 1
                    filename = f"step_{step_num:03d}_annotated.png"
                    filepath = os.path.join(screenshots_dir, filename)
                    try:
                        img_bytes = base64.b64decode(entry["annotated_b64"])
                        with open(filepath, "wb") as f:
                            f.write(img_bytes)
                    except Exception as e:
                        print(f"[RewardAgentImproved] Failed to save {filename}: {e}")
                
                if entry.get("screenshot_after_b64"):
                    screenshot_count += 1
                    filename = f"step_{step_num:03d}_after.png"
                    filepath = os.path.join(screenshots_dir, filename)
                    try:
                        img_bytes = base64.b64decode(entry["screenshot_after_b64"])
                        with open(filepath, "wb") as f:
                            f.write(img_bytes)
                    except Exception as e:
                        print(f"[RewardAgentImproved] Failed to save {filename}: {e}")
            
            log_entries_json = []
            for entry in self.log_history:
                step_num = entry.get("step", 0)
                json_entry = {
                    "step": step_num,
                    "type": entry.get("type", "unknown"),
                    "timestamp": entry.get("timestamp"),
                    "thought": entry.get("thought"),
                    "action": entry.get("action"),
                    "observation": entry.get("observation"),
                    "screenshots": {}
                }
                
                if entry.get("screenshot_before_b64"):
                    json_entry["screenshots"]["before"] = f"step_{step_num:03d}_before.png"
                if entry.get("annotated_b64"):
                    json_entry["screenshots"]["annotated"] = f"step_{step_num:03d}_annotated.png"
                if entry.get("screenshot_after_b64"):
                    json_entry["screenshots"]["after"] = f"step_{step_num:03d}_after.png"
                
                log_entries_json.append(json_entry)
            
            payload = {
                "meta": {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "task_instruction": task_instruction,
                    "apps": apps,
                    "model_id": self.model_id,
                    "max_images": self.max_images,
                    "max_steps": self.max_steps,
                    "agent_type": "RewardAgentImproved",
                    "total_steps": len(self.log_history),
                    "total_screenshots": screenshot_count,
                },
                "evaluation": evaluation,
                "log_history": log_entries_json,
            }
            
            json_path = os.path.join(log_dir, "improved_agent_run.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"[RewardAgentImproved] Saved log history to: {json_path}")
            
            html_path = self._generate_html_report(log_dir, payload)
            print(f"[RewardAgentImproved] Generated HTML report: {html_path}")
            
        except Exception as e:
            print(f"[RewardAgentImproved] Failed to save agent log: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_html_report(self, log_dir: str, payload: Dict[str, Any]) -> str:
        html_path = os.path.join(log_dir, "conversation_report.html")
        
        html_parts = []
        html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RewardAgentImproved - Conversation Report</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 32px; margin-bottom: 10px; }
        .header p { opacity: 0.9; font-size: 16px; }
        .meta-section {
            padding: 25px 30px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }
        .meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        .meta-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .meta-label {
            font-weight: 600;
            color: #495057;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        .meta-value {
            color: #212529;
            font-size: 14px;
            word-break: break-word;
        }
        .evaluation-section {
            padding: 25px 30px;
            background: white;
            border-bottom: 2px solid #e9ecef;
        }
        .evaluation-section h2 {
            color: #212529;
            margin-bottom: 20px;
            font-size: 24px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .verdict-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
            text-transform: uppercase;
        }
        .verdict-success { background: #d4edda; color: #155724; }
        .verdict-partial { background: #fff3cd; color: #856404; }
        .verdict-failure { background: #f8d7da; color: #721c24; }
        .reward-bar {
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin: 15px 0;
        }
        .reward-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            transition: width 0.5s ease;
        }
        .reasoning-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin-top: 15px;
        }
        .conversation-section {
            padding: 25px 30px;
        }
        .conversation-section h2 {
            color: #212529;
            margin-bottom: 20px;
            font-size: 24px;
        }
        .message {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #e9ecef;
        }
        .message-system { background: #f8f9fa; border-left-color: #6c757d; }
        .message-user { background: #e3f2fd; border-left-color: #2196f3; }
        .message-assistant { background: #f3e5f5; border-left-color: #9c27b0; }
        .message-tool { background: #fff3e0; border-left-color: #ff9800; }
        .message-role {
            font-weight: 600;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .message-content {
            font-size: 14px;
            color: #212529;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .message-image {
            margin-top: 10px;
            border-radius: 8px;
            overflow: hidden;
            border: 2px solid #dee2e6;
        }
        .message-image img {
            max-width: 100%;
            height: auto;
            display: block;
        }
        .tool-call {
            background: #fff;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 10px;
            margin-top: 10px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }
        .tool-name {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 5px;
        }
        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 RewardAgentImproved</h1>
            <p>Conversation History & Evaluation Report</p>
        </div>
""")
        
        meta = payload.get("meta", {})
        html_parts.append("""
        <div class="meta-section">
            <div class="meta-grid">
                <div class="meta-item">
                    <div class="meta-label">Task Instruction</div>
                    <div class="meta-value">{}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Related Apps</div>
                    <div class="meta-value">{}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Model</div>
                    <div class="meta-value">{}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Timestamp</div>
                    <div class="meta-value">{}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Screenshots Captured</div>
                    <div class="meta-value">{}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Max Steps</div>
                    <div class="meta-value">{}</div>
                </div>
            </div>
        </div>
""".format(
            meta.get("task_instruction", "N/A"),
            ", ".join(meta.get("apps", [])),
            meta.get("model_id", "N/A"),
            meta.get("timestamp", "N/A"),
            meta.get("total_screenshots", 0),
            meta.get("max_steps", "N/A")
        ))
        
        evaluation = payload.get("evaluation", {})
        reward = evaluation.get("reward", 0.0)
        verdict = evaluation.get("verdict", "Unknown")
        reasoning = evaluation.get("reasoning", "")
        
        verdict_class = "verdict-success" if verdict == "Success" else \
                       "verdict-partial" if verdict == "Partial Success" else "verdict-failure"
        
        html_parts.append("""
        <div class="evaluation-section">
            <h2>📊 Evaluation Result</h2>
            <div>
                <span class="verdict-badge {}">{}</span>
            </div>
            <div class="reward-bar">
                <div class="reward-fill" style="width: {}%">
                    Reward: {:.2f}
                </div>
            </div>
            <div class="reasoning-box">
                <strong>Reasoning:</strong><br>
                {}
            </div>
        </div>
""".format(verdict_class, verdict, reward * 100, reward, reasoning))
        
        conversation = payload.get("log_history", [])
        html_parts.append("""
        <div class="conversation-section">
            <h2>💬 Thought-Action-Observation History ({} steps)</h2>
""".format(len(conversation)))
        
        for i, entry in enumerate(conversation):
            step_num = entry.get("step", 0)
            step_type = entry.get("type", "unknown")
            thought = entry.get("thought", "")
            action = entry.get("action")
            observation = entry.get("observation", "")
            screenshots = entry.get("screenshots", {})
            
            if step_type == "initial":
                role_class = "message-user"
                role_icon = "🎯"
                role_label = "TASK"
            elif step_type == "computer":
                role_class = "message-assistant"
                role_icon = "🖱️"
                role_label = f"STEP {step_num}"
            elif step_type == "tool":
                role_class = "message-tool"
                role_icon = "🔧"
                role_label = f"STEP {step_num}"
            elif step_type == "final_answer":
                role_class = "message-system"
                role_icon = "✅"
                role_label = "FINAL"
            else:
                role_class = "message-system"
                role_icon = "📝"
                role_label = f"STEP {step_num}"
            
            html_parts.append(f"""
            <div class="message {role_class}">
                <div class="message-role">{role_icon} {role_label}</div>
""")
            
            if thought:
                html_parts.append(f"""
                <div style="margin-bottom: 10px;">
                    <strong>💭 Thought:</strong>
                    <div class="message-content">{thought[:500]}{'...' if len(thought) > 500 else ''}</div>
                </div>
""")
            
            if action:
                tool_name = action.get("tool_name", "unknown") if isinstance(action, dict) else "unknown"
                arguments = action.get("arguments", "") if isinstance(action, dict) else ""
                
                if tool_name == "computer":
                    try:
                        args_dict = json.loads(arguments) if isinstance(arguments, str) else arguments
                        action_type = args_dict.get("action", "unknown")
                        
                        action_details = f"<strong>Action:</strong> {action_type}"
                        if "coordinate" in args_dict:
                            coord = args_dict["coordinate"]
                            action_details += f"<br><strong>Coordinate:</strong> [{coord[0]}, {coord[1]}] (relative 0-999)"
                        if "text" in args_dict:
                            action_details += f"<br><strong>Text:</strong> {args_dict['text']}"
                        if "keys" in args_dict:
                            action_details += f"<br><strong>Keys:</strong> {', '.join(args_dict['keys'])}"
                        if "pixels" in args_dict:
                            action_details += f"<br><strong>Scroll:</strong> {args_dict['pixels']} pixels"
                        
                        html_parts.append(f"""
                <div class="tool-call">
                    <div class="tool-name">⚡ Action: Computer</div>
                    <div style="margin-top: 8px;">{action_details}</div>
                </div>
""")
                    except Exception:
                        html_parts.append(f"""
                <div class="tool-call">
                    <div class="tool-name">⚡ Action: {tool_name}</div>
                    <pre>{arguments}</pre>
                </div>
""")
                else:
                    html_parts.append(f"""
                <div class="tool-call">
                    <div class="tool-name">⚡ Action: {tool_name}</div>
                    <pre>{arguments}</pre>
                </div>
""")
            
            if observation:
                html_parts.append(f"""
                <div style="margin-top: 10px; padding: 10px; background: #e3f2fd; border-radius: 6px; border-left: 3px solid #2196f3;">
                    <strong>👁️ Observation:</strong> {observation[:500]}{'...' if len(observation) > 500 else ''}
                </div>
""")
            
            if screenshots:
                if "before" in screenshots:
                    img_path = f"screenshots/{screenshots['before']}"
                    html_parts.append(f"""
                <div class="message-image" style="margin-top: 10px;">
                    <img src="{img_path}" alt="Before action">
                    <div style="text-align: center; font-size: 12px; color: #666; margin-top: 5px;">Before action</div>
                </div>
""")
                
                if "annotated" in screenshots:
                    img_path = f"screenshots/{screenshots['annotated']}"
                    html_parts.append(f"""
                <div class="message-image" style="margin-top: 10px;">
                    <img src="{img_path}" alt="Action annotation">
                    <div style="text-align: center; font-size: 12px; color: #666; margin-top: 5px;">Action visualization</div>
                </div>
""")
                
                if "after" in screenshots:
                    img_path = f"screenshots/{screenshots['after']}"
                    html_parts.append(f"""
                <div class="message-image" style="margin-top: 10px;">
                    <img src="{img_path}" alt="After action">
                    <div style="text-align: center; font-size: 12px; color: #666; margin-top: 5px;">After action</div>
                </div>
""")
            
            html_parts.append("            </div>\n")
        
        html_parts.append("""
        </div>
""")
        
        html_parts.append("""
        <div class="footer">
            Generated by RewardAgentImproved • {} 
        </div>
    </div>
</body>
</html>
""".format(meta.get("timestamp", "")))
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("".join(html_parts))
        
        return html_path
    def _parse_final_answer_from_json(self, arguments: str) -> Dict[str, Any]:
        try:
            args = json.loads(arguments)
            reward = float(args.get("reward", 0.0))
            verdict = args.get("verdict", "Failure")
            reasoning = args.get("reasoning", "")
            
            if not (0 <= reward <= 1):
                reward = max(0.0, min(1.0, reward))
            
            valid_verdicts = ["Success", "Partial Success", "Failure"]
            if verdict not in valid_verdicts:
                verdict = "Failure"
            
            return {
                "reward": reward,
                "verdict": verdict,
                "reasoning": reasoning,
            }
        except Exception as e:
            print(f"[RewardAgentImproved] Failed to parse final_answer: {e}")
            return {
                "reward": 0.0,
                "verdict": "Failure",
                "reasoning": f"Failed to parse final answer: {e}",
            }
    
    def _parse_final_answer_from_text(self, text: str) -> Dict[str, Any]:
        try:
            match = re.search(
                r'final_answer\s*\(\s*reward\s*[=:]\s*([0-9.]+)\s*,\s*verdict\s*[=:]\s*["\']([^"\']+)["\']\s*,\s*reasoning\s*[=:]\s*["\']([^"\']*)["\']',
                text,
                re.DOTALL
            )
            if match:
                reward = float(match.group(1))
                verdict = match.group(2)
                reasoning = match.group(3)
                
                return {
                    "reward": max(0.0, min(1.0, reward)),
                    "verdict": verdict if verdict in ["Success", "Partial Success", "Failure"] else "Failure",
                    "reasoning": reasoning,
                }
            
            json_match = re.search(r'\{[^{}]*"reward"[^{}]*\}', text, re.DOTALL)
            if json_match:
                return self._parse_final_answer_from_json(json_match.group(0))
            
            return {
                "reward": 0.0,
                "verdict": "Failure",
                "reasoning": "Could not parse final_answer from text",
            }
        except Exception as e:
            return {
                "reward": 0.0,
                "verdict": "Failure",
                "reasoning": f"Error parsing final_answer: {e}",
            }
    
    def _execute_computer_action_from_json(self, arguments: str) -> str:
        try:
            args = json.loads(arguments)
            action = args.get("action", "")

            if action == "left_click_drag":
                return self._execute_drag_action(args)
            
            pyautogui_cmd = self._convert_action_to_pyautogui(args)

            if pyautogui_cmd == "WAIT":
                wait_time = args.get("time", 0)
                self.env.step("WAIT", pause=float(wait_time) if wait_time else 0)
                return f"Waited {wait_time} seconds"
            elif pyautogui_cmd == "DONE":
                return "Terminate requested"
            elif pyautogui_cmd:
                obs, _, _, _ = self.env.step(pyautogui_cmd, pause=0)
                return f"Computer action executed: {action}"
            else:
                return f"No action executed for: {action}"
        except Exception as e:
            return f"Failed to execute computer action: {e}"

    def _get_screen_resolution(self) -> Tuple[int, int]:
        try:
            from PIL import Image
            obs = self.env._get_obs() if hasattr(self.env, "_get_obs") else {
                "screenshot": self.env.controller.get_screenshot()
            }
            img = Image.open(BytesIO(obs["screenshot"]))
            return img.size
        except Exception:
            return 1920, 1080

    def _scale_relative_coordinate(self, x: float, y: float, width: int, height: int) -> Tuple[int, int]:
        x_scale = width / 999
        y_scale = height / 999
        return int(x * x_scale), int(y * y_scale)

    def _execute_drag_action(self, args: Dict[str, Any]) -> str:
        try:
            width, height = self._get_screen_resolution()
            duration = float(args.get("duration", 0.5))

            if "start_coordinate" in args and "end_coordinate" in args:
                start_x, start_y = args["start_coordinate"]
                end_x, end_y = args["end_coordinate"]

                adj_start_x, adj_start_y = self._scale_relative_coordinate(start_x, start_y, width, height)
                adj_end_x, adj_end_y = self._scale_relative_coordinate(end_x, end_y, width, height)

                self.env.step(f"pyautogui.moveTo({adj_start_x}, {adj_start_y})", pause=0)
                self.env.step(
                    f"pyautogui.dragTo({adj_end_x}, {adj_end_y}, duration={duration}, button='left')",
                    pause=0,
                )
                return (
                    "Computer action executed: left_click_drag "
                    f"from [{start_x}, {start_y}] to [{end_x}, {end_y}]"
                )

            if "coordinate" in args:
                end_x, end_y = args["coordinate"]
                adj_end_x, adj_end_y = self._scale_relative_coordinate(end_x, end_y, width, height)
                self.env.step(
                    f"pyautogui.dragTo({adj_end_x}, {adj_end_y}, duration={duration}, button='left')",
                    pause=0,
                )
                return (
                    "Computer action executed: left_click_drag using legacy format "
                    f"(drag from current mouse position to [{end_x}, {end_y}])"
                )

            return "Failed to execute left_click_drag: missing coordinates"
        except Exception as e:
            return f"Failed to execute left_click_drag: {e}"
    
    def _convert_action_to_pyautogui(self, args: Dict[str, Any]) -> str:
        action = args.get("action", "")
        
        try:
            from PIL import Image
            obs = self.env._get_obs() if hasattr(self.env, "_get_obs") else {
                "screenshot": self.env.controller.get_screenshot()
            }
            img = Image.open(BytesIO(obs["screenshot"]))
            orig_w, orig_h = img.size
        except Exception:
            orig_w, orig_h = 1920, 1080
        
        def adjust_coords(x: float, y: float) -> Tuple[int, int]:
            x_scale = orig_w / 999
            y_scale = orig_h / 999
            return int(x * x_scale), int(y * y_scale)
        
        if action == "left_click":
            if "coordinate" in args:
                x, y = args["coordinate"]
                adj_x, adj_y = adjust_coords(x, y)
                return f"pyautogui.click({adj_x}, {adj_y})"
            return "pyautogui.click()"
        
        elif action == "right_click":
            if "coordinate" in args:
                x, y = args["coordinate"]
                adj_x, adj_y = adjust_coords(x, y)
                return f"pyautogui.rightClick({adj_x}, {adj_y})"
            return "pyautogui.rightClick()"
        
        elif action == "double_click":
            if "coordinate" in args:
                x, y = args["coordinate"]
                adj_x, adj_y = adjust_coords(x, y)
                return f"pyautogui.doubleClick({adj_x}, {adj_y})"
            return "pyautogui.doubleClick()"
        
        elif action == "middle_click":
            if "coordinate" in args:
                x, y = args["coordinate"]
                adj_x, adj_y = adjust_coords(x, y)
                return f"pyautogui.middleClick({adj_x}, {adj_y})"
            return "pyautogui.middleClick()"
        
        elif action == "type":
            text = args.get("text", "")
            safe_text = text.replace("'", "\\'")
            return f"pyautogui.typewrite('{safe_text}')"
        
        elif action == "key":
            keys = args.get("keys", [])
            if isinstance(keys, list):
                keys_str = ", ".join([f"'{k}'" for k in keys])
                if len(keys) > 1:
                    return f"pyautogui.hotkey({keys_str})"
                elif len(keys) == 1:
                    return f"pyautogui.press({keys_str})"
            return ""
        
        elif action == "scroll":
            pixels = args.get("pixels", 0)
            return f"pyautogui.scroll({pixels})"
        
        elif action == "mouse_move":
            if "coordinate" in args:
                x, y = args["coordinate"]
                adj_x, adj_y = adjust_coords(x, y)
                return f"pyautogui.moveTo({adj_x}, {adj_y})"
            return ""
        
        elif action == "left_click_drag":
            if "coordinate" in args:
                x, y = args["coordinate"]
                adj_x, adj_y = adjust_coords(x, y)
                duration = args.get("duration", 0.5)
                return f"pyautogui.dragTo({adj_x}, {adj_y}, duration={duration})"
            return ""
        
        elif action == "wait":
            return "WAIT"
        
        elif action == "terminate":
            return "DONE"
        
        return ""
    
    def _parse_react_action_from_text(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        try:
            action_pattern = r'Action:\s*(\w+)\s*\(([^)]*)\)'

            matches = re.findall(action_pattern, text)
            if not matches:
                return None

            for tool_name, args_str in matches:
                tool_name = tool_name.strip()

                if tool_name.lower() in ('thought', 'think', 'reasoning'):
                    continue

                args_dict = {}

                if args_str.strip():
                    try:
                        arg_matches = re.findall(
                            r'(\w+)\s*[=:]\s*("[^"]*"|\'[^\']*\'|\[[^\]]*\]|[^,)\s]+)',
                            args_str.strip()
                        )
                        for key, value in arg_matches:
                            value = value.strip()
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            elif value.startswith("'") and value.endswith("'"):
                                value = value[1:-1]
                            elif value.startswith('[') and value.endswith(']'):
                                try:
                                    value = json.loads(value)
                                except:
                                    pass
                            else:
                                try:
                                    if '.' in value:
                                        value = float(value)
                                    else:
                                        value = int(value)
                                except ValueError:
                                    pass

                            args_dict[key] = value
                    except Exception as e:
                        print(f"[RewardAgentImproved] Failed to parse args '{args_str}': {e}")

                return (tool_name, args_dict)

            return None
        except Exception as e:
            print(f"[RewardAgentImproved] Failed to parse ReAct action from text: {e}")
            return None

    def _parse_and_execute_computer_action_from_text(self, text: str) -> Optional[Tuple[bool, str]]:
        react_result = self._parse_react_action_from_text(text)
        if react_result:
            tool_name, args_dict = react_result

            if tool_name in ("computer", "computer_use"):
                action = args_dict.get("action", "")

                if action == "terminate":
                    return (True, "Agent called terminate")

                if action == "left_click_drag":
                    result_text = self._execute_drag_action(args_dict)
                    return (False, result_text)

                pyautogui_cmd = self._convert_action_to_pyautogui(args_dict)
                if pyautogui_cmd and pyautogui_cmd not in ("WAIT", "DONE"):
                    self.env.step(pyautogui_cmd, pause=0)
                    return (False, f"Executed (ReAct): {action}")
                elif pyautogui_cmd == "WAIT":
                    wait_time = args_dict.get("time", 0)
                    self.env.step("WAIT", pause=float(wait_time) if wait_time else 0)
                    return (False, f"Waited {wait_time} seconds (ReAct)")

                return (False, f"ReAct action: {action}")

            elif tool_name in self._tools_map:
                try:
                    result = self._tools_map[tool_name](**args_dict)
                    return (False, f"ReAct tool result: {str(result)[:200]}")
                except Exception as e:
                    return (False, f"ReAct tool error: {e}")

        try:
            match = re.search(r'<\|im_start\|>\s*(\{.*?\})\s*<\|im_end\|>', text, re.DOTALL)
            if not match:
                return None

            json_str = match.group(1)
            tool_call = json.loads(json_str)

            if tool_call.get("name") not in ("computer_use", "computer"):
                return None

            args = tool_call.get("arguments", {})
            action = args.get("action", "")

            if action == "terminate":
                return (True, "Agent called terminate")

            if action == "left_click_drag":
                result_text = self._execute_drag_action(args)
                return (False, result_text)

            pyautogui_cmd = self._convert_action_to_pyautogui(args)
            if pyautogui_cmd and pyautogui_cmd not in ("WAIT", "DONE"):
                self.env.step(pyautogui_cmd, pause=0)
                return (False, f"Executed (XML): {action}")
            elif pyautogui_cmd == "WAIT":
                wait_time = args.get("time", 0)
                self.env.step("WAIT", pause=float(wait_time) if wait_time else 0)
                return (False, f"Waited {wait_time} seconds (XML)")

            return (False, f"XML action: {action}")
        except Exception as e:
            print(f"[RewardAgentImproved] Failed to parse/execute XML action: {e}")
            return None
    
    def _execute_smolagent_tool(self, tool_name: str, arguments: str) -> str:
        try:
            tool = self._tools_map.get(tool_name)
            if not tool:
                return f"Tool '{tool_name}' not found"
            
            args = json.loads(arguments)
            
            result = tool(**args)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"
    
    def _trim_screenshots_if_needed(self):
        while self._image_count > self.max_images:
            self._trim_oldest_screenshot()
    
    def _trim_oldest_screenshot(self):
        for msg in self.messages[1:]:
            content = msg.get("content")
            
            if isinstance(content, list):
                new_content = []
                removed = False
                for item in content:
                    if item.get("type") == "image_url" and not removed:
                        removed = True
                        self._image_count -= 1
                    else:
                        new_content.append(item)
                
                if removed:
                    if new_content:
                        msg["content"] = new_content
                    else:
                        msg["content"] = [{"type": "text", "text": "[Screenshot removed to save context]"}]
                    return
    
    def _handle_response(self, message: Any) -> Optional[Dict[str, Any]]:
        thought = message.content or ""
        
        if thought.strip():
            print(f"[RewardAgentImproved] Thought: {thought[:200]}...")
        
        if message.tool_calls:
            self.messages.append({
                "role": "assistant",
                "content": thought,
                "tool_calls": [tc.model_dump() for tc in message.tool_calls]
            })
            
            for tc in message.tool_calls:
                if tc.function.name == "final_answer":
                    self.step_counter += 1
                    self.log_history.append({
                        "step": self.step_counter,
                        "type": "final_answer",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "thought": thought,
                        "action": {
                            "tool_name": "final_answer",
                            "arguments": tc.function.arguments,
                        },
                        "observation": "Evaluation completed",
                        "screenshot_before_b64": None,
                        "screenshot_after_b64": None,
                        "annotated_b64": None,
                    })
                    
                    return self._parse_final_answer_from_json(tc.function.arguments)
                elif tc.function.name == "computer":
                    screenshot_before_b64 = self._get_current_screenshot()

                    result_text = self._execute_computer_action_from_json(tc.function.arguments)
                    
                    print(f"[RewardAgentImproved] Observation: {result_text}")
                    
                    screenshot_b64 = self._get_current_screenshot()

                    try:
                        args_dict = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                        annotated_b64 = self._annotate_screenshot(screenshot_before_b64, args_dict)
                    except Exception:
                        annotated_b64 = screenshot_before_b64

                    self.step_counter += 1
                    self.log_history.append({
                        "step": self.step_counter,
                        "type": "computer",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "thought": thought,
                        "action": {
                            "tool_name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                        "observation": result_text,
                        "screenshot_before_b64": screenshot_before_b64,
                        "screenshot_after_b64": screenshot_b64,
                        "annotated_b64": annotated_b64,
                    })
                    
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Observation: {result_text}"
                    })
                    
                    self.messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Observation: the requested computer action has been executed and a fresh screenshot "
                                    "of the current state is attached below. Use this screenshot directly as updated visual "
                                    "evidence, reassess any remaining unresolved criteria, and avoid redundant recaptioning "
                                    "of the same screen."
                                )
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}}
                        ]
                    })
                    self._image_count += 1
                    self._trim_screenshots_if_needed()
                    
                    self._save_step_incrementally(thought, tc.function.name, tc.function.arguments, result_text, screenshot_b64)
                else:
                    result_text = self._execute_smolagent_tool(tc.function.name, tc.function.arguments)
                    
                    print(f"[RewardAgentImproved] Observation ({tc.function.name}): {result_text[:200]}...")

                    self.step_counter += 1
                    self.log_history.append({
                        "step": self.step_counter,
                        "type": "tool",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "thought": thought,
                        "action": {
                            "tool_name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                        "observation": result_text,
                        "screenshot_before_b64": None,
                        "screenshot_after_b64": None,
                        "annotated_b64": None,
                    })
                    
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Observation: {result_text}"
                    })
            
            return None
        
        self.messages.append({
            "role": "assistant",
            "content": thought
        })
        
        if "final_answer(" in thought:
            return self._parse_final_answer_from_text(thought)
        
        action_result = self._parse_and_execute_computer_action_from_text(thought)
        if action_result:
            is_done, result_text = action_result
            if is_done:
                return {
                    "reward": 0.0,
                    "verdict": "Failure",
                    "reasoning": "Agent terminated without providing final_answer",
                }

            print(f"[RewardAgentImproved] Observation: {result_text}")

            screenshot_before_b64 = None
            screenshot_after_b64 = self._get_current_screenshot()

            react_result = self._parse_react_action_from_text(thought)
            action_info = None
            if react_result:
                tool_name, args_dict = react_result
                action_info = {
                    "tool_name": tool_name,
                    "arguments": json.dumps(args_dict) if args_dict else "",
                }

            self.step_counter += 1
            self.log_history.append({
                "step": self.step_counter,
                "type": "react_action",
                "timestamp": datetime.datetime.now().isoformat(),
                "thought": thought,
                "action": action_info,
                "observation": result_text,
                "screenshot_before_b64": screenshot_before_b64,
                "screenshot_after_b64": screenshot_after_b64,
                "annotated_b64": None,
            })

            self.messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Observation: {result_text}\n\n"
                            "A fresh screenshot of the post-action state is attached. Use it directly as updated visual evidence, "
                            "reassess remaining criteria, and avoid redundant recaptioning of the same screen."
                        )
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_after_b64}"}}
                ]
            })
            self._image_count += 1
            self._trim_screenshots_if_needed()

        return None
    
    def _annotate_screenshot(self, screenshot_b64: str, args: Dict[str, Any]) -> str:
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io as _io

            if not screenshot_b64:
                return screenshot_b64

            img_bytes = base64.b64decode(screenshot_b64)
            img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
            draw = ImageDraw.Draw(img, "RGBA")
            w, h = img.size

            action = args.get("action", "")

            def _font(size: int):
                try:
                    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
                except Exception:
                    try:
                        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
                    except Exception:
                        return ImageFont.load_default()

            def _px(coord):
                return int(coord[0] / 999 * w), int(coord[1] / 999 * h)

            if action in ("left_click", "right_click", "double_click", "middle_click", "mouse_move"):
                coord = args.get("coordinate")
                if coord:
                    px, py = _px(coord)
                    r = max(18, int(min(w, h) * 0.025))
                    color_map = {
                        "left_click":   (255,  70,  70, 220),
                        "right_click":  ( 70,  70, 255, 220),
                        "double_click": (255, 165,   0, 220),
                        "middle_click": (160,  32, 240, 220),
                        "mouse_move":   ( 30, 200,  80, 220),
                    }
                    color = color_map.get(action, (255, 0, 0, 220))
                    rgb = color[:3]
                    draw.ellipse([px - r, py - r, px + r, py + r], fill=color[:3] + (60,), outline=rgb, width=3)
                    gap = r + 4
                    draw.line([px - gap - 8, py, px - gap, py], fill=rgb, width=2)
                    draw.line([px + gap, py, px + gap + 8, py], fill=rgb, width=2)
                    draw.line([px, py - gap - 8, px, py - gap], fill=rgb, width=2)
                    draw.line([px, py + gap, px, py + gap + 8], fill=rgb, width=2)
                    label = action.replace("_", " ").upper()
                    fnt = _font(14)
                    try:
                        bbox = fnt.getbbox(label)
                        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    except Exception:
                        tw, th = len(label) * 8, 14
                    lx, ly = px + r + 6, py - th // 2 - 4
                    draw.rectangle([lx - 2, ly - 2, lx + tw + 8, ly + th + 4], fill=rgb)
                    draw.text((lx + 4, ly), label, fill=(255, 255, 255), font=fnt)
                    badge = str(self.step_counter)
                    bfnt = _font(13)
                    br = 10
                    draw.ellipse([px - r - 2, py - r - 2 - br * 2, px - r - 2 + br * 2, py - r - 2], fill=(30, 30, 30, 200))
                    draw.text((px - r + 1, py - r - 2 - br * 2 + 1), badge, fill=(255, 255, 100), font=bfnt)

            elif action == "scroll":
                pixels = args.get("pixels", 0)
                cx, cy = w // 2, h // 2
                arrow_color = (255, 140, 0, 230)
                half = 28
                tip = 18
                if pixels > 0:
                    pts = [(cx, cy - half - tip), (cx - tip, cy - half), (cx + tip, cy - half)]
                    draw.polygon(pts, fill=arrow_color)
                    draw.line([cx, cy - half, cx, cy + half], fill=arrow_color, width=4)
                    lbl = f"SCROLL UP ({pixels}px)"
                else:
                    pts = [(cx, cy + half + tip), (cx - tip, cy + half), (cx + tip, cy + half)]
                    draw.polygon(pts, fill=arrow_color)
                    draw.line([cx, cy + half, cx, cy - half], fill=arrow_color, width=4)
                    lbl = f"SCROLL DOWN ({abs(pixels)}px)"
                fnt = _font(16)
                try:
                    bbox = fnt.getbbox(lbl)
                    tw = bbox[2] - bbox[0]
                except Exception:
                    tw = len(lbl) * 9
                draw.rectangle([cx - tw // 2 - 6, cy + half + tip + 8, cx + tw // 2 + 6, cy + half + tip + 34], fill=(255, 140, 0, 200))
                draw.text((cx - tw // 2, cy + half + tip + 10), lbl, fill=(255, 255, 255), font=fnt)

            elif action == "type":
                text_val = args.get("text", "")
                display = f"TYPE: {text_val[:60]}{'…' if len(text_val) > 60 else ''}"
                fnt = _font(16)
                bar_h = 40
                draw.rectangle([0, 0, w, bar_h], fill=(20, 20, 20, 210))
                draw.text((10, 10), display, fill=(255, 230, 80), font=fnt)

            elif action == "key":
                keys = args.get("keys", [])
                key_str = " + ".join(keys) if isinstance(keys, list) else str(keys)
                display = f"KEY: {key_str}"
                fnt = _font(16)
                bar_h = 40
                draw.rectangle([0, 0, w, bar_h], fill=(20, 20, 80, 210))
                draw.text((10, 10), display, fill=(120, 200, 255), font=fnt)

            out = _io.BytesIO()
            img.save(out, format="PNG")
            return base64.b64encode(out.getvalue()).decode("utf-8")

        except Exception as e:
            print(f"[RewardAgentImproved] _annotate_screenshot failed: {e}")
            return screenshot_b64

    def _save_step_incrementally(
        self,
        thought: str,
        tool_name: str,
        tool_arguments: str,
        observation: str,
        screenshot_b64: str
    ):
        if not self.output_dir:
            return

        try:
            step_data = {
                "step_number": self.step_counter,
                "timestamp": datetime.datetime.now().isoformat(),
                "thought": thought[:500] if thought else "",
                "action": {
                    "tool_name": tool_name,
                    "arguments": tool_arguments
                },
                "observation": observation[:500] if observation else "",
                "screenshot_saved": f"step_{self.step_counter:03d}_screenshot.png"
            }

            steps_log_path = os.path.join(self.output_dir, "steps_log.jsonl")
            with open(steps_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(step_data, ensure_ascii=False) + "\n")

            if screenshot_b64:
                screenshot_path = os.path.join(self.output_dir, f"step_{self.step_counter:03d}_screenshot.png")
                try:
                    image_bytes = base64.b64decode(screenshot_b64)
                    with open(screenshot_path, "wb") as f:
                        f.write(image_bytes)
                except Exception as e:
                    print(f"[RewardAgentImproved] Failed to save step screenshot: {e}")

            self.steps_log.append(step_data)

            print(f"[RewardAgentImproved] ✓ Saved step {self.step_counter} to {steps_log_path}")

        except Exception as e:
            print(f"[RewardAgentImproved] Failed to save step incrementally: {e}")
            import traceback
            traceback.print_exc()
