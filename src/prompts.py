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
      ' curriculum of expert demonstrations, categorized by game phase.'
      ' Your goal is to learn the specific policy for each phase.\n\n'
      '--- 1. END-GAME DEMOS ---'
      ' (These are short, 3-5 step games)'
      ' **Your Goal:** Learn to identify and execute *immediate, forced wins*.'
      ' Focus on recognizing 1-move or 2-move winning patterns.'
      ' This is your most important skill.\n\n'

      '--- 2. MID-GAME DEMOS ---'
      ' (These are medium-length, 5-7 step games)'
      ' **Your Goal:** Learn how to *create* the winning end-game scenarios you'
      ' learned above. Focus on blocking opponent threats and setting up'
      ' 2-way forks or unavoidable wins.\n\n'

      '--- 3. OPENING/FULL-GAME DEMOS ---'
      ' (These are long, 8-9 step games)'
      ' **Your Goal:** Learn the safest and most effective opening moves.'
      ' Focus on moves that lead to the strong mid-game positions you'
      ' have already seen.\n\n'
      
      'Pay attention to these labels. When you are in a new game, first'
      ' identify the current game phase, then apply the specific policy'
      ' you have learned for that phase.\n\n'
      f'{demonstrations}\n'
    )

def get_game_phase_label(demo_length: int, total_steps: int) -> str:
    """Categorizes a demo based on its length."""
    ratio = demo_length / total_steps
    if ratio < 0.33:
        return "--- END-GAME DEMO ---"
    elif ratio < 0.66:
        return "--- MID-GAME DEMO ---"
    else:
        return "--- OPENING DEMO ---"

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
