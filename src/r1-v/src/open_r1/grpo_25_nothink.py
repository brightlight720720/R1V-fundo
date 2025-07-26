# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import re
import logging
import sys
from pathlib import Path

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration 

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "reasoning_steps"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )

    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


from trl.trainer.grpo_config import GRPOConfig as _GRPOConfig

@dataclass
class LocalGRPOConfig(_GRPOConfig): # add after nothing ck100-r130
    """
    Adds a `target_kl` parameter so we can steer the adaptive‑KL controller.
    """
    target_kl: float = field(
        default=0.05,
        metadata={
            "help": (
                "Target KL divergence for the adaptive‑KL controller. "
                "β is increased whenever the measured KL exceeds this value."
            )
        },
    )


def accuracy_reward_old(completions, solution, **kwargs): # before checkpoint 100+30
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
                else:
                    try:
                        diff = abs(float(student_answer) - float(ground_truth))
                        reward = max(0.0, 1 - diff / 2)          # 0.75 if off by 1
                    except Exception:
                        pass
                if reward < 0.2:
                    reward = -0.1                            # discourage wrong answers

            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a",encoding='utf-8', errors='replace') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

def accuracy_reward(completions, solution, **kwargs):
    """
    Reward function that checks if the completion is correct using either symbolic verification or exact string matching.
    - Full reward (1.0) for correct answers.
    - Partial reward for numerically close answers.
    - Penalty (-0.2) for missing answers.
    - Penalty (-0.1) for wrong answers.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = -0.2  # Default penalty for missing or wrong answers

        # 1. Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # If symbolic verification fails, continue to string matching

        # 2. If symbolic verification failed, try string matching
        if reward == -0.2:
            try:
                # Extract answer from solution if it has <answer>...</answer> tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has <answer>...</answer> tags
                content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # 2a. Explicitly penalize missing answers
                if not student_answer:
                    reward = -0.2  # Strong penalty for no answer
                else:
                    # 2b. Compare the extracted answers
                    if student_answer == "3" and ground_truth == "3":
                        reward = 1.2
                    elif student_answer == "4" and ground_truth == "4":
                        reward = 1.2
                    elif student_answer == ground_truth:
                        reward = 1.0
                    else:
                        try:
                            # Try to compare as numbers for partial credit
                            diff = abs(float(student_answer) - float(ground_truth))
                            reward = max(0.0, 1 - diff / 2)  # 0.75 if off by 1, etc.
                        except Exception:
                            reward = -0.2  # after ck100 +100+30, change to this 
                            #pass
                    # 2c. Penalize wrong answers
                    if reward ==0:
                        reward = -0.1  # Discourage wrong answers

            except Exception:
                pass  # Keep reward as -0.2 if all else fails
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8', errors='replace') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [0.1 if match else 0.0 for match in matches]
    #[0.2 if match else 0.0 for match in matches]....before checkpoint 100

def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|observe,|notice,|find,|quality,|good,|poor)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(0.1, count / 10) for count in matches]
    #[min(0.2, count / 10) for count in matches]...before checkpoint 100

def weighted_reward(completions, solution, **kwargs):
    acc = accuracy_reward(completions, solution, **kwargs)
    fmt = format_reward(completions, **kwargs)          #  ignores extra kwargs
    rsn = reasoning_steps_reward(completions, **kwargs)

    return [2.0*a + 0.3*fmt_i + 1.0*rsn_i
            for a, fmt_i, rsn_i in zip(acc, fmt, rsn)]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "reasoning_steps": reasoning_steps_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

SYSTEM_PROMPT_IMAGE = '''
    A conversation between User and Assistant. You are an advanced vision-language model expert in medical image analysis. You can analyze images at both the overall and pixel levels, compare feature extractions, and discern subtle differences between images.
You are provided with a synthetic fundoscopy image. Your task is threefold:
1. please assess the quality of the fundoscopy by evaluating the clarity of anatomical structures, the visibility of key landmarks, and the absence of artifacts. Good quality images allow for clear visualization of the optic disc, macula, and vessels, while poor quality images are characterized by blurriness and obscuration , which artefacts or eye abnormalities cover more than 1/4 of images or  vascular arches or larger vascular arches are invisible.** evaluate the quality first and describe the quality assessment and before make the diagnosis**
2.	Analysis: Examine the synthetic fundoscopy image and describe, step by step, how it differs from a normal fundoscopy image.
3.	Diagnosis: Based on your analysis, determine whether the synthetic fundoscopy image show sign of disease.
** 0 =No diabetic retinopathy, 1= Mild non-proliferative diabetic retinopathy, 2 =Moderate non-proliferative diabetic retinopathy,3 = Severe non-proliferative diabetic retinopathy,4 =Proliferative diabetic retinopathy**
Please structure your response into two main sections: think and answer.
**you have to reply in english only, any other language or special words is not allowed, and focus on diagnosis**
•	think:
<think>
**Detailed Comparative Analysis:**
    *   Perform a meticulous comparison. For each anatomical structure, clearly state what is observed in the image and then describe any **differences or abnormalities** compare to a normal fundoscopy image.
    *   Focus on the following key areas and specific findings:
    *   **Optic Disc:**
        *   **Color:** (e.g., pinkish-orange, pale, hyperemic)
        *   **Margins:** (e.g., sharp, blurred, swollen, elevated)
        *   **Cup-to-Disc Ratio (CDR):** (e.g., normal 0.3-0.5, enlarged, obliterated)
        *   **Presence of:** (e.g., papilledema, drusen, neovascularization)
    *   **Retinal Vessels:**
        *   **Caliber:** (e.g., normal, attenuated, dilated)
        *   **Tortuosity:** (e.g., straight, tortuous)
        *   **A-V Ratio:** (e.g., normal 2:3, altered)
        *   **Presence of:** (e.g., hemorrhages [flame, dot-blot, boat-shaped], exudates [hard, soft/cotton wool spots], sheathing, neovascularization, venous engorgement, nicking)
    *   **Macula and Fovea:**
        *   **Foveal Reflex:** (e.g., present, absent)
        *   **Pigmentary Changes:** (e.g., mottling, hyperpigmentation, hypopigmentation)
        *   **Presence of:** (e.g., drusen, exudates, hemorrhages, edema, cysts, holes)

    *   **Retinal Background (Peripheral Retina):**
        *   **Color/Clarity:** (e.g., even, discolored, cloudy)
        *   **Presence of:** (e.g., hemorrhages, exudates, drusen, cotton wool spots, retinal tears/detachment, laser scars, pigmentary changes)
    *   **Vitreous:**
        *   **Clarity:** (e.g., clear, hazy, presence of floaters/hemorrhage)
**Summary of Key Abnormal Findings:**
    *   Based on your detailed comparison, list the most prominent or concerning abnormalities observed in the current fundoscopy image that deviate from the normal reference.
     ** provide the quality conclusion of this fundoscopy at the end**
</think>

•	Answer:
<answer>
Provide your final diagnosis after consider about the quality and what you observe, as a single number:
0 =No diabetic retinopathy, 1= Mild non-proliferative diabetic retinopathy, 2 =Moderate non-proliferative diabetic retinopathy,3 = Severe non-proliferative diabetic retinopathy,4 =Proliferative diabetic retinopathy, 5=diabetic macular edema**. 
</answer>

The output format must strictly follow these tags:
<think>
... [your detailed observation] ...
</think>
<answer>
[final answer: "0" or "1" or "2" or "3" or "4"]
</answer>
Please ensure you adhere strictly to this format and that your final answer is only : "0" or "1" or "2" or "3" or "4"

'''


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    #reward_funcs = [weighted_reward]

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config,cache_dir="/workspace/hf_cache")            
                          


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
    QUESTION_TEMPLATE_IMAGE = """{Question} A conversation between User and Assistant. You are an advanced vision-language model expert in medical image analysis. You can analyze images at both the overall and pixel levels, compare feature extractions, and discern subtle differences between images.
You are provided with a synthetic fundoscopy image. Your task is threefold:
1. please assess the quality of the fundoscopy by evaluating the clarity of anatomical structures, the visibility of key landmarks, and the absence of artifacts. Good quality images allow for clear visualization of the optic disc, macula, and vessels, while poor quality images are characterized by blurriness and obscuration , which artefacts or eye abnormalities cover more than 1/4 of images or  vascular arches or larger vascular arches are invisible.** evaluate the quality first and describe the quality assessment and before make the diagnosis **
2.	Analysis: Examine the synthetic fundoscopy image and describe, step by step, how it differs from a normal fundoscopy image.
3.	Diagnosis: Based on your analysis, determine whether the synthetic fundoscopy image show sign of disease.
** 0 =No diabetic retinopathy, 1= Mild non-proliferative diabetic retinopathy, 2 =Moderate non-proliferative diabetic retinopathy,3 = Severe non-proliferative diabetic retinopathy,4 =Proliferative diabetic retinopathy**. 
Your final diagnosis must be a single number. No other responses are allowed.
Please structure your response into two main sections: think and answer.
**you have to reply in english only, any other language or special words is not allowed, and focus on diagnosis**
•	think: <think>
**Detailed Comparative Analysis:**
    *   Perform a meticulous comparison. For each anatomical structure, clearly state what is observed in the image and then describe any **differences or abnormalities** compare to a normal fundoscopy image.
    *   Focus on the following key areas and specific findings:
    *   **Optic Disc:**
        *   **Color:** (e.g., pinkish-orange, pale, hyperemic)
        *   **Margins:** (e.g., sharp, blurred, swollen, elevated)
        *   **Cup-to-Disc Ratio (CDR):** (e.g., normal 0.3-0.5, enlarged, obliterated)
        *   **Presence of:** (e.g., papilledema, drusen, neovascularization)
    *   **Retinal Vessels:**
        *   **Caliber:** (e.g., normal, attenuated, dilated)
        *   **Tortuosity:** (e.g., straight, tortuous)
        *   **A-V Ratio:** (e.g., normal 2:3, altered)
        *   **Presence of:** (e.g., hemorrhages [flame, dot-blot, boat-shaped], exudates [hard, soft/cotton wool spots], sheathing, neovascularization, venous engorgement, nicking)
    *   **Macula and Fovea:**
        *   **Foveal Reflex:** (e.g., present, absent)
        *   **Pigmentary Changes:** (e.g., mottling, hyperpigmentation, hypopigmentation)
        *   **Presence of:** (e.g., drusen, exudates, hemorrhages, edema, cysts, holes)
    *   **Retinal Background (Peripheral Retina):**
        *   **Color/Clarity:** (e.g., even, discolored, cloudy)
        *   **Presence of:** (e.g., hemorrhages, exudates, drusen, cotton wool spots, retinal tears/detachment, laser scars, pigmentary changes)
    *   **Vitreous:**
        *   **Clarity:** (e.g., clear, hazy, presence of floaters/hemorrhage)
**Summary of Key Abnormal Findings:**
    *   Based on your detailed comparison, list the most prominent or concerning abnormalities observed in the current fundoscopy image that deviate from the normal reference.
    ** provide the quality conclusion of this fundoscopy at the end**
</think>
•	Answer: 
the criteria of Stages of Diabetic Retinopathy and their Fundoscopic Findings:
No diabetic retinopathy:
No visible retinal changes.
Non-Proliferative Diabetic Retinopathy (NPDR):
Mild NPDR: Presence of microaneurysms (tiny, balloon-like dilations of retinal capillaries).
Moderate NPDR: Microaneurysms, hemorrhages (small bleeds in the retina), and hard exudates (yellowish deposits of lipid and protein).
Severe NPDR: 4:2:1 rule: hemorrhages in all four quadrants, venous beading in two or more quadrants, or intraretinal microvascular abnormalities (IRMA) in one or more quadrants.

Proliferative Diabetic Retinopathy (PDR):
Neovascularization (new blood vessel growth) on the optic disc or retina, often accompanied by vitreous hemorrhage or tractional retinal detachment.
Diabetic Macular Edema (DME):
Fluid and thickening of the macula (central part of the retina), which can occur at any stage of DR. 
<answer>
Provide your final diagnosis after consider about the quality and what you observe, as a single number:
**0 =No diabetic retinopathy, 1= Mild non-proliferative diabetic retinopathy, 2 =Moderate non-proliferative diabetic retinopathy,3 = Severe non-proliferative diabetic retinopathy,4 =Proliferative diabetic retinopathy**. 
</answer>

**The output format must strictly follow these tags**:
<think>
... [your detailed observation] ...
</think>
<answer>
[final answer: "0" or "1" or "2" or "3" or "4"]
</answer>
Please ensure you adhere strictly to this format and that your final answer is only : "0" or "1" or "2" or "3" or "4"

"""
    
    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE_IMAGE.format(Question=example["problem"])},
                    ],
                },
            ],
        }


    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, LocalGRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
