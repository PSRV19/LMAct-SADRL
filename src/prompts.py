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


def build_demonstration_prompt(
    demonstrations: str,
) -> str:
  """Returns the prompt for the demonstrations."""
  if not demonstrations:
    return (
        'You are an intelligent agent operating in a dynamic environment. Based'
        ' on the series of observations provided, you need to determine the'
        ' optimal action that maximizes the expected reward or achieves the'
        ' desired goal. Carefully consider all the given observations, infer'
        ' the current state of the environment, and select the most appropriate'
        ' action.\n\n'
    )

  # MODIFICATION: Updated the prompt to acknowledge the demo labels
  return (
      'You are a powerful reinforcement learning agent. You will be shown a'
      ' curriculum of expert demonstrations sorted from easiest to hardest (by'
      ' game phase).\n\n'
      'Pay close attention to the labels to guide your learning:\n\n'
      '1.  `--- END-GAME DEMOS ---`\n'
      '    These are short, solved scenarios. Your Goal: Learn to identify'
      '    and execute immediate, forced wins or optimal final moves.\n\n'
      '2.  `--- MID-GAME DEMOS ---`\n'
      '    These demos show complex, multi-step problem solving. Your Goal:'
      '    Learn the main heuristics and sub-policies needed to gain an'
      '    advantage or navigate obstacles.\n\n'
      '3.  `--- OPENING-GAME DEMOS ---`\n'
      '    These are full, long games. Your Goal: Learn how to connect the'
      '    opening moves to the mid-game strategies you have learned.\n\n'
      'Apply this knowledge to the current situation.\n\n'
      f'{demonstrations}\n'
  )

def get_game_phase_label(demo_length: int, max_steps: int) -> str:
    """Categorizes a demo based on its length, which is representative of game phase."""
   # Ensure max_steps is not zero to avoid division by zero
    if max_steps <= 0:
        max_steps = 1  # Avoid division by zero
        
    ratio = demo_length / max_steps

    # Shortest demos = END-GAME (e.g., a 2-step forced win)
    if ratio < 0.5:
        return "END-GAME"
    # Medium-length demos = MID-GAME
    elif ratio < 0.8:
        return "MID-GAME"
    # Longest demos = OPENING-GAME
    else:
        return "OPENING-GAME"


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
    prompt += 'demonstrations and the '
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
