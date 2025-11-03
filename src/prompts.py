# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the prompts for the experiment."""

from lm_act.src import config as config_lib
from typing import Any


def build_demonstration_prompt(
    heuristics_list: str,
) -> str:
  """Returns the prompt for the demonstrations."""
  if not heuristics_list:
    return (
        'You are an intelligent agent operating in a dynamic environment. Based'
        ' on the series of observations provided, you need to determine the'
        ' optimal action that maximizes the expected reward or achieves the'
        ' desired goal. Carefully consider all the given observations, infer'
        ' the current state of the environment, and select the most appropriate'
        ' action.\n\n'
    )
  
  # MODIFICATION: Join all the individual heuristics into one big string
  all_heuristics_str = '\n\n'.join(heuristics_list)

  # MODIFICATION: This is the new "verbal reinforcement" prompt
  return (
      'You are a powerful reinforcement learning agent. You must learn from the'
      ' following grounded examples of expert policy and reasoning. Apply this'
      ' logic to the new situation.\n\n'
      '--- START OF EXAMPLES ---\n\n'
      f'{all_heuristics_str}\n\n'
      '--- END OF EXAMPLES ---\n\n'
  )

def _format_raw_observation(  # NEW HELPER
    config: config_lib.Experiment,
    observation: Any,
    demo_idx: int,
    step_idx: int,
) -> tuple[str, dict[str, Any]]:
  """Formats a single raw observation into a string."""
  content_by_tag = dict()
  match config.environment.observation_type:
    case 'fen' | 'coords':
      obs_prompt = f'Observation: {observation}'
    case 'dict':
      obs_prompt = f'Observation: {observation}'
    case 'pgn' | 'txt':
      obs_prompt = f'Observation:\n{observation}'
    case 'rgb' | 'png':
      # This prompt generator does not support images
      # It will just show a placeholder.
      tag = f'<IMG_{demo_idx}_{step_idx}>'
      obs_prompt = f'Observation: {tag}'
    case _:
      raise ValueError(
          'Unsupported observation type:'
          f' {config.environment.observation_type}'
      )
  return obs_prompt, content_by_tag

def build_contrastive_prompt(  # NEW FUNCTION 
    game_name: str,
    formatted_observation: str,
    expert_action: str,
    example_index: int,
) -> str:
  """Returns the prompt for generating a contrastive heuristic."""

  return (
      'You are an expert AI policy analyst. Your task is to provide verbal'
      ' reinforcement by explaining *why* an expert\'s action is superior to a'
      ' common suboptimal action.\n\n'
      '--- CONTEXT ---\n'
      f'Game: {game_name}\n'
      f'{formatted_observation}\n'
      f'Expert Action: {expert_action}\n\n'
      '--- YOUR TASK ---\n'
      '1.  First, think of a plausible **Suboptimal Action** a novice might'
      '    take in this *exact* situation.\n'
      '2.  Write an **Expert Rationale** explaining the goal and long-term'
      '    benefit of the expert\'s action.\n'
      '3.  Write a **Suboptimal Rationale** explaining the flaw or missed'
      '    opportunity in the novice move.\n'
      '4.  Finally, write a concise **Distilled Heuristic** that captures the'
      '    general rule from this example.\n\n'
      'Provide your analysis *only* in the following format:\n\n'
      f'[Example {example_index}]\n'
      f'{formatted_observation}\n'
      f'Expert Action: {expert_action}\n'
      'Suboptimal Action: [Your answer here]\n'
      'Expert Rationale: [Your answer here]\n'
      'Suboptimal Rationale: [Your answer here]\n'
      'Distilled Heuristic: [Your answer here]'
  )

def build_trajectory_prompt(
    trajectory: str,
    legal_actions: list[str],
    config: config_lib.Experiment,
) -> str:
  """Returns the prompt for the current trajectory."""
  prompt = f'\nThis is the current trajectory:\n\n{trajectory}\n'

  if config.prompt.show_legal_actions:
    prompt += (
        '\nIn this situation, this is the list of all the actions that are'
        f' legal:\n\n{", ".join(legal_actions)}\n'
    )

  prompt += '\nGiven the '
  if 0 < config.num_demonstrations:
    prompt += 'grounded examples and the '
  prompt += 'current trajectory, you should infer the next logical action.'

  if config.prompt.show_legal_actions:
    prompt += '\nCheck that the chosen action is in the set of legal actions.'

  if config.prompt.use_chain_of_thought:
    prompt += (
        '\nThink step by step and very briefly explain your reasoning for'
        ' choosing this action.\nYou must answer with the reasoning followed by'
        ' the action in the following format:\nReasoning: ...\nAction: ...'
    )
  else:
    if config.prompt.show_legal_actions:
      prompt += (
          '\nYou must answer with one of the legal actions only, without any'
          ' other text.'
      )
    else:
      prompt += (
          '\nYou must answer with the action only, without any other text,'
          ' following exactly the same format as the previous actions.'
      )

  return prompt.strip()
