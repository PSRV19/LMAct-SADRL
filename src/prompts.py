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
    heuristic: str, # MODIFICATION: Try natural language heuristic instead of raw demonstrations
) -> str:
  """Returns the prompt for the demonstrations."""
  if not heuristic:
    return (
        'You are an intelligent agent operating in a dynamic environment. Based'
        ' on the series of observations provided, you need to determine the'
        ' optimal action that maximizes the expected reward or achieves the'
        ' desired goal. Carefully consider all the given observations, infer'
        ' the current state of the environment, and select the most appropriate'
        ' action.\n\n'
    )

  return ( # MODIFICATION: New prompt format for heuristic
      'You are a powerful reinforcement learning agent. You must follow the'
      ' policy heuristic provided to you to solve the task. The heuristic was'
      ' distilled from expert behavior, and you should use it as your primary'
      ' guide.\n\nHere is the expert policy heuristic:\n'
      f'--- HEURISTIC START ---\n{heuristic}\n--- HEURISTIC END ---\n\n'
  )

def _format_raw_trajectory(  # NEW HELPER FUNCTION
    config: config_lib.Experiment,
    observations: list[Any],
    actions: list[Any],
    demo_idx: int,
) -> tuple[str, dict[str, Any]]:
  """Formats a single raw trajectory into a string for heuristic generation."""
  content_by_tag = dict()
  demo_prompts = list()
  for step_idx, (observation, action) in enumerate(zip(observations, actions)):
    match config.environment.observation_type:
      case 'fen' | 'coords':
        demo_prompt = f'Observation: {observation} '
      case 'dict':
        demo_prompt = f'Observation: {observation}\n'
      case 'pgn' | 'txt':
        demo_prompt = f'Observation:\n{observation}\n'
      case 'rgb' | 'png':
        # The heuristic generator does not support images
        # We will just use a placeholder
        tag = f'<IMG_{demo_idx}_{step_idx}>'
        demo_prompt = f'Observation: {tag} '
      case _:
        raise ValueError(
            'Unsupported observation type:'
            f' {config.environment.observation_type}'
        )

    demo_prompt += f'Action: {action}'
    demo_prompts.append(demo_prompt.strip())
    demo_prompts.append('\n')
  return ''.join(demo_prompts), content_by_tag


def build_heuristic_prompt(  # NEW FUNCTION
    game_name: str,
    formatted_trajectories: list[str],
) -> str:
  """Returns the prompt for generating a policy heuristic (Experiment 1)."""

  trajectories_str = '\n\n'.join(formatted_trajectories)

  return (
      'You are an expert AI policy analyst. Your goal is to observe'
      ' "trajectories" (sequences of observations and actions) from an'
      ' expert agent playing a game.\n\n'
      'Your task is to reflect on the expert\'s behavior and distill its'
      ' implicit strategy into a single, concise, and generalizable'
      ' "policy heuristic." This heuristic should be a natural language rule'
      ' that explains *what* the expert is trying to do, what it'
      ' prioritizes, and how it achieves its goals.\n\n'
      '- **Analyze:** Look at the full trajectory. What patterns do you see?\n'
      '- **Identify:** Focus on critical moments. What does the expert consistently prioritize (e.g., safety, offense, resource gathering)?\n'
      '- **Formulate:** Summarize this policy as a simple rule or "heuristic."\n\n'
      f'--- GAME ---\n{game_name}\n\n'
      f'--- EXPERT TRAJECTORIES ---\n{trajectories_str}\n\n'
      '--- TASK ---\n'
      'Based *only* on the trajectories provided, what is the expert\'s'
      ' distilled heuristic? Provide *only* the heuristic, with no preamble.'
      '\n\nDistilled Heuristic:'
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
    prompt += 'policy heuristic and the ' # MODIFIED: changed demonstrations to policy heuristic in wording
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
