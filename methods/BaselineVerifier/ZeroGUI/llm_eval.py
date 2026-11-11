import os
import argparse
import json
import time
import re
import random
import base64
import requests
import anthropic
from openai import OpenAI
from typing import Dict, List, Tuple, Optional

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt')


class LLMEvaluator:
    def __init__(
            self,
            api_type: str = "openai",
            model: str = "gpt-4o-2024-11-20",
            base_url: str = None,
            api_key: Optional[str] = None,
            prompt_file: str = None,
            prompt_dir: str = None,
            temperature: float = 0.0,
            voting_type: Optional[str] = None,
            voting_num: int = 1,
    ):
        self.api_type = api_type
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.voting_type = voting_type
        self.voting_num = voting_num
        print(f"Initializing LLMEvaluator with {api_type}")
        self._setup_client()

        prompt_dir = prompt_dir if prompt_dir is not None else PROMPT_DIR
        if prompt_file is not None:
            prompt_path = os.path.join(prompt_dir, prompt_file)
            with open(prompt_path, 'r') as f:
                self.prompt = json.load(f)

    def _setup_client(self):
        try:
            if self.api_type == "openai":
                self.client = OpenAI(api_key=self.api_key)
            elif self.api_type == "claude":
                self.client = anthropic.Anthropic(api_key=self.api_key)
            elif self.api_type == "3rd_openai":
                self.client = None
            elif "qwen" in self.api_type.lower():
                self.client = None
            else:
                print(f"Unsupported LLM type: {self.api_type}")
                raise NotImplementedError(f"Unsupported LLM type: {self.api_type}")
        except Exception as e:
            print(f"Failed to initialize client: {str(e)}")
            raise

    def _encode_image(self, image_bytes: bytes=None, image_path: str=None) -> str:
        if image_bytes is not None:
            return base64.b64encode(image_bytes).decode("utf-8")
        elif image_path is not None:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            raise NotImplementedError

    def call_llm(self, prompt: str, n_samples: int = 1, max_tokens: int = 2048, temperature: float = 0.0, 
                 retry_times: int = 5, retry_delay: int = 5, timeout: int = 360, return_list: bool = False):
        for attempt in range(retry_times):
            try:
                print(f"Calling {self.api_type} API (attempt {attempt+1}/{retry_times})")
                if self.api_type == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        n=n_samples,
                    )
                    if return_list:
                        return [choice.message.content for choice in response.choices]
                    else:
                        assert n_samples == 1
                        return response.choices[0].message.content
                    
                elif self.api_type == "claude":
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=prompt
                    )
                    return response.content
                    
                elif self.api_type == "3rd_openai" or "qwen" in self.api_type.lower():
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    response = requests.post(
                        os.path.join(self.base_url, "v1/chat/completions"),
                        headers=headers,
                        json={
                            "model": self.model,
                            "messages": prompt,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "n": n_samples,
                        },
                        timeout=timeout,
                    )
                    if response.status_code != 200:
                        print(f"API request failed: {response.text}")
                        if attempt < retry_times - 1:
                            print(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            continue
                        return [] if return_list else ""

                    if return_list:
                        return [choice['message']['content'] for choice in response.json()['choices']]
                    else:
                        assert n_samples == 1
                        return response.json()['choices'][0]['message']['content']
                else:
                    print(f"Unsupported LLM type: {self.api_type}")
                    raise ValueError(f"Unsupported LLM type: {self.api_type}")
                    
            except Exception as e:
                print(f"Error calling {self.api_type} API (attempt {attempt+1}/{retry_times}): {str(e)}")
                if attempt < retry_times - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed after {retry_times} attempts")
                    raise

        return [] if return_list else ""

    def parse_from_response(self, response: str) -> float:
        try:
            completion_pattern = r'(?:SCORE|[Ss]core)(?:\*\*)?[:]\s*(?:\*\*)?\s*(?:\*\*)?([0-9]+)'
            completion_matches = list(re.finditer(completion_pattern, response, re.IGNORECASE))
            
            if completion_matches:
                last_match = completion_matches[-1]
                success_text = float(last_match.group(1))
                success = (success_text == 1.0)
                reward = 1.0 if success else 0.0
            else:
                raise ValueError("Completion value not found in analysis")
        except (ValueError, AttributeError) as e:
            print(f"Error extracting completion: {e}")
            reward = -1.0

        return reward

    def evaluate_task(self, task_config: dict, trajectory: dict, save_dir=None):
        try:
            
            instruction = task_config["instruction"]
            system_prompt = self.prompt["system_prompt"]
            user_prompt = self.prompt["user_prompt"]
            user_prompt = user_prompt.format(instruction=instruction)
            screenshots = trajectory["screenshots"]

            if self.api_type == "openai" or self.api_type == "3rd_openai" or "qwen" in self.api_type.lower():
                messages = [{"role": "system", "content": system_prompt}]
                user_content = [{"type": "text", "text": user_prompt}]
                
                for image_bytes in screenshots:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self._encode_image(image_bytes=image_bytes)}",
                            "detail": "high",
                        }
                    })

                messages.append({"role": "user", "content": user_content})
                formatted_prompt = messages
                
            elif self.api_type == "claude":
                content = [{"type": "text", "text": user_prompt}]
                for image_bytes in screenshots:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": self._encode_image(image_bytes=image_bytes)
                        }
                    })
                formatted_prompt = [{"role": "user", "content": content}]
                
            else:
                print(f"Unsupported LLM type: {self.api_type}")
                raise NotImplementedError(f"Unsupported LLM type: {self.api_type}")

            if self.voting_type is None:
                analysis = self.call_llm(formatted_prompt, temperature=self.temperature)
                if save_dir:
                    with open(os.path.join(save_dir, "llm_analysis.txt"), "a") as f:
                        f.write(analysis + "\n\n===========================================\n\n")
                
                reward = self.parse_from_response(analysis)
                
                eval_outputs = {
                    "reward": reward,
                    "llm_output": analysis,
                }
            else:
                analysis = self.call_llm(formatted_prompt, n_samples=self.voting_num, 
                                         temperature=self.temperature, return_list=True)
                voting_rewards = []
                for response in analysis:
                    reward = self.parse_from_response(response)
                    if reward >=0:
                        voting_rewards.append(reward)

                if len(voting_rewards) == 0:
                    reward = -1
                else:
                    voting_sum = sum(voting_rewards)
                    if self.voting_type == "majority":
                        reward = 1 if voting_sum > len(voting_rewards) // 2 else 0
                    elif self.voting_type == "all":
                        reward = 1 if voting_sum == len(voting_rewards) else 0
                    elif self.voting_type == "any":
                        reward if voting_sum > 1 else 0
                    else:
                        raise NotImplementedError

                eval_outputs = {
                    "reward": reward,
                    "voting_rewards": voting_rewards,
                    "llm_output": analysis,
                }

            return eval_outputs
            
        except Exception as e:
            print(f"Error in evaluate_task: {str(e)}")
            raise
