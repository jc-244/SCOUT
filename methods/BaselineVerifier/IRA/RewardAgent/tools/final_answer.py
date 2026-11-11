from smolagents import Tool
import json


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Finish the evaluation by returning the final reward judgment as JSON. Call this when you are ready to conclude."
    inputs = {
        "reward": {
            "description": "Final reward score in [0, 1].",
            "type": "number",
        },
        "verdict": {
            "description": "Final verdict. One of: 'Success', 'Partial Success', 'Failure'.",
            "type": "string",
        },
        "reasoning": {
            "description": "Short explanation for the final decision.",
            "type": "string",
        },
    }
    output_type = "string"

    def forward(self, reward: float, verdict: str, reasoning: str) -> str:
        return self.__call__(reward, verdict, reasoning)

    def __call__(self, reward=None, verdict=None, reasoning=None):
        if isinstance(reward, dict) and verdict is None and reasoning is None:
            data = reward
            reward = data.get("reward")
            verdict = data.get("verdict")
            reasoning = data.get("reasoning")

        if reward is None or verdict is None or reasoning is None:
            raise ValueError("final_answer requires reward, verdict, and reasoning.")

        try:
            reward_f = float(reward)
        except Exception:
            raise ValueError(f"reward must be a float in [0, 1], got: {reward!r}")
        if not (0.0 <= reward_f <= 1.0):
            raise ValueError(f"reward must be in [0, 1], got: {reward_f}")

        allowed = {"Success", "Partial Success", "Failure"}
        if verdict not in allowed:
            raise ValueError(f"verdict must be one of {sorted(allowed)}, got: {verdict!r}")

        reasoning_s = str(reasoning)

        result = {
            "reward": reward_f,
            "verdict": verdict,
            "reasoning": reasoning_s,
        }
        return json.dumps(result, ensure_ascii=False)

    def to_code_prompt(self):
        return (
            "def final_answer(reward: float, verdict: str, reasoning: str) -> str:\n"
            "    \"\"\"Finish the evaluation and return the final JSON string.\n"
            "    - reward: float in [0, 1]\n"
            "    - verdict: one of [\"Success\", \"Partial Success\", \"Failure\"]\n"
            "    - reasoning: short explanation\n"
            "    \"\"\"\n"
        )
