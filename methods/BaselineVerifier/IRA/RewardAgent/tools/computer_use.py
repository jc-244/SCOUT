from __future__ import annotations

import os
import json
import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from smolagents import Tool

try:
    from PIL import Image
except Exception as e:
    Image = None

try:
    import openai
except Exception:
    openai = None

try:
    import backoff
except Exception:
    backoff = None


def _ensure_pillow():
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required for ComputerUseTool. Please install 'Pillow'.")


def smart_resize(height: int, width: int, factor: int = 32, max_pixels: int = 512 * 512) -> Tuple[int, int]:
    if height <= 0 or width <= 0:
        return height, width

    aspect = width / height
    total = height * width
    if total > max_pixels:
        scale = (max_pixels / total) ** 0.5
        height = max(1, int(height * scale))
        width = max(1, int(width * scale))
        width = max(1, int(round(height * aspect)))

    def round_to_factor(x: int) -> int:
        rem = x % factor
        if rem == 0:
            return x
        down = x - rem
        up = down + factor
        return up

    height = round_to_factor(height)
    width = round_to_factor(width)

    return height, width


def process_image(image_bytes: bytes, max_pixels: int = 512 * 512) -> Tuple[str, Tuple[int, int]]:
    _ensure_pillow()
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size

    resized_height, resized_width = smart_resize(height=height, width=width, factor=32, max_pixels=max_pixels)
    image = image.resize((resized_width, resized_height))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    processed_bytes = buffer.getvalue()

    return base64.b64encode(processed_bytes).decode("utf-8"), (resized_width, resized_height)


class QwenClient:

    def __init__(self, model: str = "qwen3-vl-flash", max_tokens: int = 1500, temperature: float = 0.0, top_p: float = 0.9):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        if openai is None:
            raise RuntimeError("openai python package is required. Please install 'openai'.")

        api_key = os.environ.get("QWEN_API_KEY")
        if not api_key:
            raise RuntimeError("QWEN_API_KEY environment variable is not set.")

        base_url = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def _call_once(self, messages: List[Dict[str, Any]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return response.choices[0].message.content

    def call(self, messages: List[Dict[str, Any]]) -> str:
        tries = 5
        for attempt in range(tries):
            try:
                return self._call_once(messages)
            except Exception as e:
                if attempt == tries - 1:
                    raise
                import time
                time.sleep(5)
        return ""


class ComputerUseTool(Tool):

    name = "computer_use"
    description = (
        "Take a GUI screenshot, let a Qwen VL model propose the next action, "
        "execute it via env.step, and save pre/post screenshots into output_dir/eval_pic."
    )
    inputs = {
        "instruction": {
            "description": "Natural language instruction for the step",
            "type": "string",
        }
    }
    output_type = "string"

    def __init__(self, env: Any, model: str = "qwen3-vl-flash", coordinate_type: str = "relative"):
        self.env = env
        self.output_dir: Optional[str] = None
        self.eval_pic_dir: Optional[str] = None
        self.step_counter: int = 1
        self.coordinate_type: str = coordinate_type
        self.model = model
        self._qwen: Optional[QwenClient] = None

        self.history_n: int = 4
        self._responses: List[str] = []
        self._screenshots_b64: List[str] = []

        super().__init__()

    def set_output_dir(self, output_dir: str):
        self.output_dir = output_dir
        self.eval_pic_dir = os.path.join(output_dir, "eval_pic")
        os.makedirs(self.eval_pic_dir, exist_ok=True)

    def _get_qwen(self) -> QwenClient:
        if self._qwen is None:
            self._qwen = QwenClient(model=self.model)
        return self._qwen

    def to_code_prompt(self) -> str:
        return (
            "def computer_use(instruction: str) -> dict:\n"
            "    \"\"\"Perform one GUI step planned by Qwen based on a screenshot; returns a dict summary.\"\"\"\n"
        )

    def _save_png(self, image_bytes: bytes, path: str):
        with open(path, "wb") as f:
            f.write(image_bytes)

    def _build_tools_def(self) -> Dict[str, Any]:
        action_description_prompt = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) coordinate.
* `left_click`: Click the left mouse button at a specified (x, y) coordinate.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) coordinate.
* `right_click`: Click the right mouse button at a specified (x, y) coordinate.
* `middle_click`: Click the middle mouse button at a specified (x, y) coordinate.
* `double_click`: Double-click the left mouse button at a specified (x, y) coordinate.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
        """

        description_prompt_lines = [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
            (
                "* The screen's resolution is 1000x1000."
            ),
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element.",
        ]

        description_prompt = "\n".join(description_prompt_lines)

        return {
            "type": "function",
            "function": {
                "name_for_human": "computer_use",
                "name": "computer_use",
                "description": description_prompt,
                "parameters": {
                    "properties": {
                        "action": {
                            "description": action_description_prompt,
                            "enum": [
                                "key", "type", "mouse_move", "left_click", "left_click_drag",
                                "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"
                            ],
                            "type": "string"
                        },
                        "keys": {"description": "Required by action=key.", "type": "array"},
                        "text": {"description": "Required by action=type.", "type": "string"},
                        "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                        "pixels": {"description": "The amount of scrolling.", "type": "number"},
                        "time": {"description": "The seconds to wait.", "type": "number"},
                        "status": {"description": "The status of the task.", "type": "string", "enum": ["success", "failure"]},
                    },
                    "required": ["action"],
                    "type": "object"
                },
                "args_format": "Format the arguments as a JSON object."
            }
        }

    def _build_messages(self, instruction: str, processed_image_b64: str) -> List[Dict[str, Any]]:
        tools_def = self._build_tools_def()
        system_prompt = (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n" + json.dumps(tools_def) + "\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\n"
            "# Response format\n\n"
            "Response format for every step:\n"
            "1) Action: a short imperative describing what to do in the UI.\n"
            "2) A single <tool_call>...</tool_call> block containing only the JSON.\n\n"
            "Rules:\n"
            "- Output exactly in the order: Action, <tool_call>.\n"
            "- Be brief: one sentence for Action.\n"
            "- Do not output anything else outside those parts.\n"
            "- If finishing, use action=terminate in the tool call."
        )

        current_step = len(self._responses)

        instruction_prompt = (
            "Please generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
            f"Instruction: {instruction}\n\n"
            "Previous actions:\n" + ("\n".join([f"Step {i+1}: {self._responses[i]}" for i in range(max(0, current_step - self.history_n))]) or "None")
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]

        history_len = min(self.history_n, len(self._responses))
        if history_len > 0:
            history_screenshots = self._screenshots_b64[-history_len - 1 : -1]
            history_responses = self._responses[-history_len:]
            for idx in range(history_len):
                if idx < len(history_screenshots):
                    img_url = f"data:image/png;base64,{history_screenshots[idx]}"
                    messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_url}}]})
                messages.append({"role": "assistant", "content": [{"type": "text", "text": f"{history_responses[idx]}"}]})

            curr_img_url = f"data:image/png;base64,{processed_image_b64}"
            messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": curr_img_url}}]})
        else:
            curr_img_url = f"data:image/png;base64,{processed_image_b64}"
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": curr_img_url}},
                    {"type": "text", "text": instruction_prompt},
                ],
            })

        return messages

    def _parse_response(self, response: str, original_width: int, original_height: int) -> Tuple[str, List[str], Optional[float]]:
        low_level_instruction = ""
        pyautogui_code: List[str] = []
        wait_time: Optional[float] = None

        def adjust_coordinates(x: float, y: float) -> Tuple[int, int]:
            x_scale = original_width / 999
            y_scale = original_height / 999
            return int(x * x_scale), int(y * y_scale)

        def process_tool_call(json_str: str) -> None:
            try:
                tool_call = json.loads(json_str)
                if tool_call.get("name") == "computer_use":
                    args = tool_call["arguments"]
                    action = args["action"]

                    if action == "left_click":
                        if "coordinate" in args:
                            x, y = args["coordinate"]
                            adj_x, adj_y = adjust_coordinates(x, y)
                            pyautogui_code.append(f"pyautogui.click({adj_x}, {adj_y})")
                        else:
                            pyautogui_code.append("pyautogui.click()")

                    elif action == "right_click":
                        if "coordinate" in args:
                            x, y = args["coordinate"]
                            adj_x, adj_y = adjust_coordinates(x, y)
                            pyautogui_code.append(f"pyautogui.rightClick({adj_x}, {adj_y})")
                        else:
                            pyautogui_code.append("pyautogui.rightClick()")

                    elif action == "middle_click":
                        if "coordinate" in args:
                            x, y = args["coordinate"]
                            adj_x, adj_y = adjust_coordinates(x, y)
                            pyautogui_code.append(f"pyautogui.middleClick({adj_x}, {adj_y})")
                        else:
                            pyautogui_code.append("pyautogui.middleClick()")

                    elif action == "double_click":
                        if "coordinate" in args:
                            x, y = args["coordinate"]
                            adj_x, adj_y = adjust_coordinates(x, y)
                            pyautogui_code.append(f"pyautogui.doubleClick({adj_x}, {adj_y})")
                        else:
                            pyautogui_code.append("pyautogui.doubleClick()")

                    elif action == "type":
                        text = args.get("text", "")
                        safe_text = text.replace("'", "\\'")
                        pyautogui_code.append(f"pyautogui.typewrite('{safe_text}')")

                    elif action == "key":
                        keys = args.get("keys", [])
                        if isinstance(keys, list):
                            keys = [str(k).strip() for k in keys]
                        keys_str = ", ".join([f"'{key}'" for key in keys])
                        if len(keys) > 1:
                            pyautogui_code.append(f"pyautogui.hotkey({keys_str})")
                        elif len(keys) == 1:
                            pyautogui_code.append(f"pyautogui.press({keys_str})")

                    elif action == "scroll":
                        pixels = args.get("pixels", 0)
                        pyautogui_code.append(f"pyautogui.scroll({pixels})")

                    elif action == "wait":
                        pyautogui_code.append("WAIT")
                        try:
                            wt = args.get("time", 0)
                            wait_time = float(wt) if wt is not None else 0.0
                        except Exception:
                            wait_time = 0.0

                    elif action == "terminate":
                        pyautogui_code.append("DONE")

                    elif action == "mouse_move":
                        if "coordinate" in args:
                            x, y = args["coordinate"]
                            adj_x, adj_y = adjust_coordinates(x, y)
                            pyautogui_code.append(f"pyautogui.moveTo({adj_x}, {adj_y})")
                        else:
                            pyautogui_code.append("pyautogui.moveTo(0, 0)")

                    elif action == "left_click_drag":
                        if "coordinate" in args:
                            x, y = args["coordinate"]
                            adj_x, adj_y = adjust_coordinates(x, y)
                            duration = args.get("duration", 0.5)
                            pyautogui_code.append(f"pyautogui.dragTo({adj_x}, {adj_y}, duration={duration})")
                        else:
                            pyautogui_code.append("pyautogui.dragTo(0, 0)")
            except (json.JSONDecodeError, KeyError):
                pass

        lines = response.split("\n")
        inside_tool_call = False
        current_tool_call: List[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.lower().startswith(("action:")):
                if not low_level_instruction:
                    low_level_instruction = line.split("Action:")[-1].strip()
                continue

            if line.startswith("<tool_call>"):
                inside_tool_call = True
                continue
            elif line.startswith("</tool_call>"):
                if current_tool_call:
                    process_tool_call("\n".join(current_tool_call))
                    current_tool_call = []
                inside_tool_call = False
                continue

            if inside_tool_call:
                current_tool_call.append(line)
                continue

            if line.startswith("{") and line.endswith("}"):
                try:
                    json_obj = json.loads(line)
                    if "name" in json_obj and "arguments" in json_obj:
                        process_tool_call(line)
                except json.JSONDecodeError:
                    pass

        if current_tool_call:
            process_tool_call("\n".join(current_tool_call))

        if not low_level_instruction and len(pyautogui_code) > 0:
            action_type = pyautogui_code[0].split(".", 1)[1].split("(", 1)[0]
            low_level_instruction = f"Performing {action_type} action"

        return low_level_instruction, pyautogui_code, wait_time

    def forward(self, instruction: str) -> str:
        return self.__call__(instruction)

    def __call__(self, instruction: str) -> str:
        if not self.eval_pic_dir:
            self.set_output_dir(os.getcwd())

        try:
            obs = self.env._get_obs() if hasattr(self.env, "_get_obs") else {"screenshot": self.env.controller.get_screenshot()}
            screenshot_bytes = obs["screenshot"]
        except Exception as e:
            return json.dumps({"error": f"Failed to capture screenshot: {e}"})

        pre_path = os.path.join(self.eval_pic_dir, f"step_{self.step_counter:04d}_pre.png")
        try:
            self._save_png(screenshot_bytes, pre_path)
        except Exception as e:
            return json.dumps({"error": f"Failed to save pre screenshot: {e}"})

        try:
            processed_b64, (proc_w, proc_h) = process_image(screenshot_bytes)
        except Exception as e:
            return json.dumps({"error": f"Failed to process screenshot: {e}"})

        self._screenshots_b64.append(processed_b64)

        try:
            messages = self._build_messages(instruction=instruction, processed_image_b64=processed_b64)
            qwen = self._get_qwen()
            response_text = qwen.call(messages)
        except Exception as e:
            action = "WAIT"
            observation, _, _, _ = self.env.step(action, pause=0)
            post_bytes = observation["screenshot"]
            post_path = os.path.join(self.eval_pic_dir, f"step_{self.step_counter:04d}_post.png")
            try:
                self._save_png(post_bytes, post_path)
            except Exception:
                post_path = ""
            result = {
                "action": action,
                "pyautogui": action,
                "done": False,
                "screenshot_pre": pre_path,
                "screenshot_post": post_path,
                "response_text": f"Model call failed: {e}",
            }
            self.step_counter += 1
            return json.dumps(result)

        try:
            _ensure_pillow()
            img = Image.open(BytesIO(screenshot_bytes))
            orig_w, orig_h = img.size
        except Exception:
            if hasattr(self.env, "screen_width") and hasattr(self.env, "screen_height"):
                orig_w, orig_h = self.env.screen_width, self.env.screen_height
            else:
                orig_w, orig_h = 1920, 1080

        low_level_instruction, pyautogui_code, wait_time = self._parse_response(response_text, original_width=orig_w, original_height=orig_h)

        done = False
        executed = "WAIT"
        if pyautogui_code:
            if "DONE" in pyautogui_code:
                executed = "DONE"
                done = True
            elif "WAIT" in pyautogui_code and len(pyautogui_code) == 1:
                executed = "WAIT"
            else:
                for cmd in pyautogui_code:
                    if cmd not in ("WAIT", "DONE"):
                        executed = cmd
                        break

        if executed == "WAIT":
            pause_val = float(wait_time) if wait_time is not None else 0.0
            observation, _, _, _ = self.env.step(executed, pause=pause_val)
        else:
            observation, _, _, _ = self.env.step(executed, pause=0)

        post_bytes = observation["screenshot"]
        post_path = os.path.join(self.eval_pic_dir, f"step_{self.step_counter:04d}_post.png")
        try:
            self._save_png(post_bytes, post_path)
        except Exception:
            post_path = ""

        self._responses.append(response_text)

        result = {
            "action": executed,
            "pyautogui": executed,
            "done": done,
            "screenshot_pre": pre_path,
            "screenshot_post": post_path,
            "response_text": response_text,
        }

        self.step_counter += 1
        return json.dumps(result)
