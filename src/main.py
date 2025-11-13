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

"""Evaluates an agent on the LMAct benchmark."""

from collections.abc import Sequence
import logging

from absl import app
from absl import flags
import immutabledict
import numpy as np
import tqdm
# wandb
import wandb
import dataclasses

from lm_act.src import config as config_lib
from lm_act.src import evaluate
from lm_act.src.agents import chess as chess_agent
from lm_act.src.agents import crossword as crossword_agent
from lm_act.src.agents import grid_world as grid_world_agent
from lm_act.src.agents import random as random_agent
from lm_act.src.agents import tic_tac_toe as tic_tac_toe_agent
from lm_act.src.agents import gpt4o_agent as gpt4o_agent
from lm_act.src.agents import o1mini_agent as o1mini_agent
from lm_act.src.environments import chess
from lm_act.src.environments import crossword
from lm_act.src.environments import dm_control
from lm_act.src.environments import grid_world
from lm_act.src.environments import tic_tac_toe


_ENVIRONMENT = flags.DEFINE_enum(
    name='environment',
    default='tic_tac_toe',
    enum_values=[
        'chess',
        'crossword',
        'dm_control',
        'grid_world',
        'tic_tac_toe',
    ],
    help='The environment to evaluate.',
)
_OBSERVATION_TYPE = flags.DEFINE_enum(
    name='observation_type',
    default='txt',
    enum_values=['coords', 'dict', 'fen', 'pgn', 'png', 'rgb', 'txt'],
    help='The observation representation to evaluate.',
)
_ACTION_TYPE = flags.DEFINE_enum(
    name='action_type',
    default='txt',
    enum_values=['txt', 'san'],
    help='The action representation to evaluate.',
)
_AGENT = flags.DEFINE_enum(
    name='agent',
    default='random',
    enum_values=[
        'random',
        'chess_stockfish',
        'crossword_oracle',
        'grid_world_shortest_path',
        'tic_tac_toe_minimax',
        'gpt4o_agent',
        'o1mini_agent',
    ],
    help='The agent to evaluate.',
)
_NUM_DEMONSTRATIONS = flags.DEFINE_integer(
    name='num_demonstrations',
    default=0,
    help='The number of demonstrations to use.',
)
_NUM_EVALUTION_EPISODES = flags.DEFINE_integer(
    name='num_evaluation_episodes',
    default=100,
    help='The number of episodes to evaluate.',
)
_NUM_EVALUATION_STEPS = flags.DEFINE_integer(
    name='num_evaluation_steps',
    default=100,
    help='The number of steps to evaluate.',
)
_MODEL_NAME = flags.DEFINE_string(
    name='model_name',
    default=None,
    help='The name of the model to use for API agents.'
)

_WANDB_PROJECT = flags.DEFINE_string(
    name='wandb_project',
    default='lm-act',
    help='The Weights & Biases project name to log to.'
)

_RUN_NAME_PREFIX = flags.DEFINE_string(
    name='run_name_prefix',
    default='demos_',
    help='The prefix for the wandb run name.'
)

_CONFIG_BY_ENVIRONMENT = immutabledict.immutabledict({
    'chess': chess.EnvironmentConfig,
    'crossword': crossword.EnvironmentConfig,
    'dm_control': dm_control.EnvironmentConfig,
    'grid_world': grid_world.EnvironmentConfig,
    'tic_tac_toe': tic_tac_toe.EnvironmentConfig,
})
_CONFIG_BY_AGENT = immutabledict.immutabledict({
    'random': random_agent.RandomAgentConfig,
    'chess_stockfish': chess_agent.StockfishAgentConfig,
    'crossword_oracle': crossword_agent.OracleAgentConfig,
    'grid_world_shortest_path': grid_world_agent.ShortestPathAgentConfig,
    'tic_tac_toe_minimax': tic_tac_toe_agent.MinimaxAgentConfig,
    'gpt4o_agent': gpt4o_agent.GPT4oAgentConfig, # Added agent
    'o1mini_agent': o1mini_agent.o1MiniAgentConfig # Added agent
})


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.getLogger().setLevel(logging.WARNING)

  agent_config_kwargs = {'action_type': _ACTION_TYPE.value}
  
  if _MODEL_NAME.value:
    agent_config_kwargs['model_name'] = _MODEL_NAME.value
  
  agent_config = _CONFIG_BY_AGENT[_AGENT.value](**agent_config_kwargs)
  
  experiment_config = config_lib.Experiment(
      num_demonstrations=_NUM_DEMONSTRATIONS.value,
      num_evaluation_steps=_NUM_EVALUATION_STEPS.value,
      agent=agent_config,  
      environment=_CONFIG_BY_ENVIRONMENT[_ENVIRONMENT.value](
          observation_type=_OBSERVATION_TYPE.value,
          action_type=_ACTION_TYPE.value,
      ),
      prompt=config_lib.Prompt(
          show_legal_actions=False, 
          use_chain_of_thought=False,
      ),
  )

  # --- MODIFICATION: Initialize wandb logging ---
  group_name = (
    f"{experiment_config.environment.name}_{experiment_config.agent.name}"
  )

  run_name = (
    f"{_RUN_NAME_PREFIX.value}{experiment_config.num_demonstrations}"
  )

  wandb.init(
    project=_WANDB_PROJECT.value,  # <-- THIS IS THE CRITICAL FIX
    config=dataclasses.asdict(experiment_config),
    group=group_name,
    name=run_name
  )

  print(f'Environment: {experiment_config.environment.name}')
  print(f'Observation type: {experiment_config.environment.observation_type}')
  print(f'Agent: {experiment_config.agent.name}')
  print(f'Num evaluation episodes: {_NUM_EVALUTION_EPISODES.value}')

  scores = list()
#   all_scores = list() # For current episode data
  num_steps = list()
  num_invalid_actions = list()
  num_illegal_actions = list()
  num_empty_actions = list()

  for episode in tqdm.trange(_NUM_EVALUTION_EPISODES.value):
    (
        episode_score,
        episode_num_steps,
        episode_num_invalid_actions,
        episode_num_illegal_actions,
        episode_num_empty_actions,
        current_episode_data,
        demonstration_prompt
    ) = evaluate.evaluate_episode(
        episode_idx=episode,
        config=experiment_config, 
    )

    scores.append(episode_score)
    # all_scores.append(current_episode_data)
    num_steps.append(episode_num_steps)
    num_invalid_actions.append(episode_num_invalid_actions)
    num_illegal_actions.append(episode_num_illegal_actions)
    num_empty_actions.append(episode_num_empty_actions)

    if episode == 1 or (episode + 1) % 50 == 0 or episode == experiment_config.num_evaluation_episodes - 1:
        print(f"\n--- DEMONSTRATION PROMPT (End of Episode {episode}) ---")
        print(demonstration_prompt)
        print(f"--- END DEMONSTRATION PROMPT (End of Episode {episode}) ---\n")

    wandb.log({
        'episode': episode,
        'score': episode_score,
        'num_steps': episode_num_steps,
        'num_invalid_actions': episode_num_invalid_actions,
        'num_illegal_actions': episode_num_illegal_actions,
        'num_empty_actions': episode_num_empty_actions,
    }, step=episode)

  wandb.summary['average_score'] = np.mean(scores)
  wandb.summary['average_num_steps'] = np.mean(num_steps)
  wandb.summary['average_num_invalid_actions'] = np.mean(num_invalid_actions)
  wandb.summary['average_num_illegal_actions'] = np.mean(num_illegal_actions)
  wandb.summary['average_num_empty_actions'] = np.mean(num_empty_actions)

  print(f'Average score: {np.mean(scores):.2f}')
  print('Run complete. View results on Weights & Biases.')

  wandb.finish()

  # --- END MODIFICATION ---

if __name__ == '__main__':
  app.run(main)
