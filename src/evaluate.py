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

"""Evaluates a single episode."""

import copy
import os
import pathlib
from typing import Any

from absl import logging
import numpy as np

from lm_act.src import bagz
from lm_act.src import config as config_lib
from lm_act.src import constants
from lm_act.src import prompts


_BASE_DIR_PATH = pathlib.Path(
    os.path.join(
        os.getcwd(),
        'data/lm_act/',
    )
)

# --- NEW HELPER FOR STRATIFIED SAMPLING ---
def _get_game_phase(demo_length: int, max_steps: int) -> str:
    """Categorizes a demo based on its length."""
    ratio = demo_length / max_steps
    if ratio < 0.33:  # Shorter games are "Opening"
        return "end"
    elif ratio < 0.66: # Medium games are "Mid-Game"
        return "mid"
    else: # Longer games are "End-Game"
        return "opening"


def _load_demonstrations_and_opening_path(
    rng: np.random.Generator,
    config: config_lib.Experiment,
) -> tuple[list[list[Any]], list[list[Any]], pathlib.Path]:
  """Loads the demonstrations (observations & actions) and the opening path.
     Modified to load demonstrations in order for horizon-curriculum learning. 

  Args:
    rng: The random number generator.
    config: The experiment configuration.

  Returns:
    - The demonstrations episodes, consisting of observations and actions.
    - The opening path, i.e., the path to the opening (which is used to set the
      initial state of the environment for the evaluation episode).

  Raises:
    ValueError: If there are insufficient demonstrations in the directory.
  """
  base_dir_path = _BASE_DIR_PATH / config.environment.name
  all_demo_names = [
      file_name
      for file_name in os.listdir(base_dir_path)
      if file_name.startswith('demonstration')
  ]

  if not all_demo_names:
    raise ValueError(f'No demonstrations found in {base_dir_path}.')
  
  # Select one demo to be the evaluation opening and remove it from the pool 
  eval_opening_idx = rng.choice(len(all_demo_names))
  eval_opening_name = all_demo_names.pop(eval_opening_idx)
  opening_path = base_dir_path / eval_opening_name
  logging.info(f"Selected {eval_opening_name} for evaluation.")
  
  if config.num_demonstrations == 0:
    # Zero-shot, no demos needed.
    return [], [], opening_path
  
  # Define decoder functions (as before)
  match config.environment.observation_type:
    case 'rgb':
      rgb_shape = constants.get_rgb_shape(config.environment.name)
      observation_decode_fn = lambda x: np.frombuffer(x, dtype=np.uint8).reshape(rgb_shape)
    case 'png':
      observation_decode_fn = lambda x: x
    case _:
      observation_decode_fn = lambda x: x.decode('utf-8')
  action_decode_fn = lambda x: x.decode('utf-8')

  # Load ALL remaining demos into memory and categorize them
  # NOTE: This assumes a max game length, and only works for tic_tac_toe for initial experimentation.

  if config.environment.name == 'tic_tac_toe':
    MAX_GAME_STEPS = 9
  else:
    # As a fallback, we'll just find the max length in the dataset
    # This is less ideal than a true game-defined max.
    all_lengths = []
    for demonstration_name in all_demo_names:
        actions_path = base_dir_path / demonstration_name / f'actions_{config.environment.action_type}.bag'
        all_lengths.append(len(bagz.BagReader(actions_path.as_posix())))
    MAX_GAME_STEPS = max(all_lengths) if all_lengths else 100
    logging.warning(f"No max steps for {config.environment.name}, defaulting to {MAX_GAME_STEPS}")

  
  categorized_demos = {"opening": [], "mid": [], "end": []}
  
  for demonstration_name in all_demo_names:
    demo_dir_path = base_dir_path / demonstration_name
    observations_path = demo_dir_path / f'observations_{config.environment.observation_type}.bag'
    actions_path = demo_dir_path / f'actions_{config.environment.action_type}.bag'
    
    observations = list(map(observation_decode_fn, bagz.BagReader(observations_path.as_posix())))
    actions = list(map(action_decode_fn, bagz.BagReader(actions_path.as_posix())))
    
    if not observations or not actions or len(observations) != len(actions):
        logging.warning(f"Skipping malformed demo: {demonstration_name}")
        continue
        
    demo_length = len(actions)
    phase = _get_game_phase(demo_length, MAX_GAME_STEPS)
    categorized_demos[phase].append((observations, actions))

  logging.info(
      f"Categorized all demos: {len(categorized_demos['opening'])} open, "
      f"{len(categorized_demos['mid'])} mid, {len(categorized_demos['end'])} end."
  )

  # Shuffle each bucket
  for phase in categorized_demos:
    rng.shuffle(categorized_demos[phase])

  # Perform Stratified Sampling (Round-Robin) ---
  selected_demos = []
  num_to_sample = config.num_demonstrations
  
  # Use round-robin sampling, prioritizing extremes (open -> end -> mid)
  # This ensures a good spread even for very small N
  sample_order = ["opening", "end", "mid"]
  
  while len(selected_demos) < num_to_sample:
    something_added = False
    for phase in sample_order:
      if categorized_demos[phase]:
        selected_demos.append(categorized_demos[phase].pop())
        something_added = True
        if len(selected_demos) == num_to_sample:
          break
    if not something_added or len(selected_demos) == num_to_sample:
      # Stop if we've hit our target or if all lists are empty
      break

  logging.info(f"Selected {len(selected_demos)} demos via stratified sampling.")

  # Split back into two lists ---
  demo_observations = [demo[0] for demo in selected_demos]
  demo_actions = [demo[1] for demo in selected_demos]

  return demo_observations, demo_actions, opening_path

  # if len(demonstration_names) < config.num_demonstrations + 1:
  #   raise ValueError(
  #       f'Insufficient demonstrations in {base_dir_path}: Need at least'
  #       f' {config.num_demonstrations + 1} but only found'
  #       f' {len(demonstration_names)}.'
  #   )

  # if config.replay_episode:
  #   assert config.num_demonstrations == 1
  #   num_openings = config.num_demonstrations
  # else:
  #   # We need to add 1 to account for the opening that that will be evaluated.
  #   num_openings = config.num_demonstrations + 1
  # demonstration_names = rng.choice(
  #     demonstration_names,
  #     size=num_openings,
  #     replace=False,
  #     shuffle=False,
  # )
  # opening_name = demonstration_names[-1]
  # demonstration_names = demonstration_names[: config.num_demonstrations]

  # demo_observations = list()
  # demo_actions = list()

  # match config.environment.observation_type:
  #   case 'rgb':
  #     rgb_shape = constants.get_rgb_shape(config.environment.name)
  #     observation_decode_fn = lambda x: np.frombuffer(
  #         x,
  #         dtype=np.uint8,
  #     ).reshape(rgb_shape)
  #   case 'png':
  #     # PNG data does not need to be decoded.
  #     observation_decode_fn = lambda x: x
  #   case _:
  #     observation_decode_fn = lambda x: x.decode('utf-8')
  # action_decode_fn = lambda x: x.decode('utf-8')

  # for demonstration_name in demonstration_names:
  #   demo_dir_path = base_dir_path / demonstration_name
  #   observations_path = (
  #       demo_dir_path
  #       / f'observations_{config.environment.observation_type}.bag'
  #   )
  #   actions_path = (
  #       demo_dir_path / f'actions_{config.environment.action_type}.bag'
  #   )
  #   observations = bagz.BagReader(observations_path.as_posix())
  #   actions = bagz.BagReader(actions_path.as_posix())
  #   assert len(observations) == len(actions) 
  #   demo_observations.append(list(map(observation_decode_fn, observations)))
  #   demo_actions.append(list(map(action_decode_fn, actions)))

  # return demo_observations, demo_actions, base_dir_path / opening_name

MAX_GAME_STEPS = 9 # For Tic-Tac-Toe

def _create_demonstration_prompt(
    config: config_lib.Experiment,
    demo_observations: list[list[Any]],
    demo_actions: list[list[Any]],
) -> tuple[str, dict[str, Any]]:
  """Returns the demonstration prompt and content for the given config."""
  content_by_tag = dict()
  demo_prompts = list()

  # MODIFICATION: Sort and Label
  
  # Determine MAX_GAME_STEPS again for labeling
  if config.environment.name == 'tic_tac_toe':
    MAX_GAME_STEPS = 9
  else:
    # Find max length *in the sample* as a fallback
    MAX_GAME_STEPS = max(len(o) for o in demo_observations) if demo_observations else 100

  # 2. Combine, SORT, and iterate to build prompt
  
  demos = list(zip(demo_observations, demo_actions))
  # Sort by length of observations (shortest to longest)
  demos.sort(key=lambda x: len(x[0])) 
  
  logging.info(f"Building prompt with {len(demos)} sorted, stratified demos.")

  current_phase = None
  for demo_idx, (observations, actions) in enumerate(demos):
    
    demo_length = len(observations)
    
    # Get the label
    phase = _get_game_phase(demo_length, MAX_GAME_STEPS)
    
    # Add the label only when the phase changes
    if phase != current_phase:
        current_phase = phase
        demo_prompts.append(f"\n--- {current_phase.upper()}-GAME DEMOS ---\n\n")

    # Format the demo as before
    for step_idx, (observation, action) in enumerate(
        zip(observations, actions)
    ):
      match config.environment.observation_type:
        case 'fen' | 'coords':
          demo_prompt = f'Observation: {observation} '
        case 'dict':
          demo_prompt = f'Observation: {observation}\n'
        case 'pgn' | 'txt':
          demo_prompt = f'Observation:\n{observation}\n'
        case 'rgb' | 'png':
          tag = f'<IMG_{demo_idx}_{step_idx}>'
          content_by_tag[tag] = observation
          demo_prompt = f'Observation: {tag} '
        case _:
          raise ValueError(...)

      demo_prompt += f'Action: {action}' 
      demo_prompts.append(demo_prompt.strip())
      demo_prompts.append('\n')
    demo_prompts.append('\n')

  # END MODIFICATION

  demonstration_prompt = prompts.build_demonstration_prompt(
      demonstrations=''.join(demo_prompts),
  )

  logging.info('Demonstration prompt with horizon-curriculum: %s', demonstration_prompt)
  return demonstration_prompt, content_by_tag


def _create_trajectory_prompt(
    config: config_lib.Experiment,
    observations: list[Any],
    actions: list[Any],
    rewards: list[float | None],
    legal_actions: list[str],
) -> tuple[str, dict[str, Any]]:
  """Returns the trajectory prompt and content for the given config."""
  content_by_tag = dict()
  trajectory_prompts = list()

  # The first action is a dummy action so we place it at the end of the list.
  actions = np.roll(copy.deepcopy(actions), -1)

  for step_idx, current_obs in enumerate(observations):
    match config.environment.observation_type:
      case 'fen' | 'coords':
        step_prompt = f'Observation: {current_obs} '
      case 'dict':
        step_prompt = f'Observation: {current_obs}\n'
      case 'pgn' | 'txt':
        step_prompt = f'Observation:\n{current_obs}\n'
      case 'rgb' | 'png':
        tag = f'<IMG_{config.num_demonstrations}_{step_idx}>'
        content_by_tag[tag] = current_obs
        step_prompt = f'Observation: {tag} '
      case _:
        raise ValueError(...)
  
    if step_idx > 0:
      prev_action = actions[step_idx] # Action taken after obs_{step_idx-1}
      current_reward = rewards[step_idx] # Reward received at obs_{step_idx}

      if config.prompt.include_past_actions and prev_action is not None:
          step_prompt += f'Action: {prev_action} '

    trajectory_prompts.append(step_prompt)
    trajectory_prompts.append('\n')

  trajectory_prompt = prompts.build_trajectory_prompt(
      config=config,
      trajectory=''.join(trajectory_prompts),
      legal_actions=legal_actions,
  )
  logging.info('Current trajectory prompt: %s', trajectory_prompt)

  return trajectory_prompt, content_by_tag


def evaluate_episode_replay(
    episode_idx: int,
    config: config_lib.Experiment,
) -> int:
  """Returns the number of correctly replayed actions for a single episode."""

  # Every episode has to initialize the RNG with a different seed.
  rng = np.random.default_rng(seed=episode_idx)

  logging.info('Setting up the agent: %s.', config.agent.name)
  agent = constants.get_agent_builder(config.agent.name)(config=config.agent)

  logging.info('Loading the demonstrations and the evaluation opening name.')
  demo_observations, demo_actions, opening_path = (
      _load_demonstrations_and_opening_path(rng=rng, config=config)
  )
  assert len(demo_observations) == 1
  assert len(demo_actions) == 1

  logging.info('Replaying episode %d (opening %s).', episode_idx, opening_path)

  logging.info('Creating the demonstration chunks.')
  demonstration_prompt, demonstration_prompt_data = (
      _create_demonstration_prompt(
          config=config,
          demo_observations=demo_observations,
          demo_actions=demo_actions
      )
  )

  num_correctly_replayed_actions = 0

  for step, (demo_observation, demo_action) in enumerate(
      zip(demo_observations[0], demo_actions[0])
  ):
    trajectory_prompt, trajectory_prompt_data = _create_trajectory_prompt(
        config=config,
        observations=demo_observations[0][: step + 1],
        actions=[None] + demo_actions[0][:step],  # Dummy initial action.
        rewards=[None] + [0.0] * step,  # Dummy rewards.
        legal_actions=list(),  # We cannot compute the legal actions.
    )
    sample = agent.step(
        observation={
            'prompt': demonstration_prompt + trajectory_prompt,
            'prompt_data': demonstration_prompt_data | trajectory_prompt_data,
        },
        environment=None,
        rng=rng,
    )
    replayed_action_is_correct = sample == demo_action
    num_correctly_replayed_actions += replayed_action_is_correct

    logging.info({
        'demo_observation': demo_observation,
        'demo_action': demo_action,
        'sample': sample,
        'replayed_action_is_correct': replayed_action_is_correct,
    })

  return num_correctly_replayed_actions


def evaluate_episode(
    episode_idx: int,
    config: config_lib.Experiment
) -> tuple[float, int, int, int, int, str]:
  """Evaluates a single episode."""

  # Every episode has to initialize the RNG with a different seed.
  rng = np.random.default_rng(seed=episode_idx)

  logging.info('Setting up the agent: %s.', config.agent.name)
  agent = constants.get_agent_builder(config.agent.name)(config=config.agent)

  logging.info('Loading the demonstrations and the evaluation opening name.')
  demo_observations, demo_actions, opening_path = (
      _load_demonstrations_and_opening_path(rng=rng, config=config)
  )

  logging.info(
      'Evaluating episode %d with opening %s.', episode_idx, opening_path
  )

  logging.info('Creating the demonstration chunks.')
  demonstration_prompt, demonstration_prompt_data = (
      _create_demonstration_prompt(
          config=config,
          demo_observations=demo_observations,
          demo_actions=demo_actions
      )
  )

  logging.info('Setting up the environment: %s.', config.environment.name)
  env = constants.get_environment_builder(config.environment.name)(
      config=config.environment,
      opening_paths=[opening_path],
  )
  time_step = env.reset()

  observations = [time_step.observation[config.environment.observation_type]]
  rewards = [time_step.reward]
  actions = [None]  # Dummy action for the initial observation.

  num_illegal_actions = num_invalid_actions = num_empty_actions = 0

  for _ in range(config.num_evaluation_steps):
    if time_step.last():
      break

    trajectory_prompt, trajectory_prompt_data = _create_trajectory_prompt(
        config=config,
        observations=observations,
        actions=actions,
        rewards=rewards,
        legal_actions=env.legal_actions,
    )
    sample = agent.step(
        observation=time_step.observation
        | {
            'prompt': demonstration_prompt + trajectory_prompt,
            'prompt_data': demonstration_prompt_data | trajectory_prompt_data,
        },
        environment=env,
        rng=rng,
    )

    sample_is_empty = not sample
    num_empty_actions += sample_is_empty

    if sample_is_invalid := env.action_is_invalid(sample):
      num_invalid_actions += 1
      # If the sample is invalid, we also always consider it illegal.
      sample_is_illegal = True
      num_illegal_actions += 1
    elif sample_is_illegal := env.action_is_illegal(sample):
      num_illegal_actions += 1

    action = env.sample_legal_action(rng) if sample_is_illegal else sample

    logging.info({
        'observation': time_step.observation[
            config.environment.observation_type
        ],
        'reward': time_step.reward,
        'action': action,
        'sample': sample,
        'sample_is_invalid': sample_is_invalid,
        'sample_is_illegal': sample_is_illegal,
        'sample_is_empty': sample_is_empty,
        'step_type': int(time_step.step_type),
    })

    time_step = env.step(action)
    observations.append(
        time_step.observation[config.environment.observation_type]
    )
    rewards.append(time_step.reward)
    actions.append(action)

  logging.info({
      'rgb': (
          time_step.observation['rgb']
          if 'rgb' in time_step.observation
          else None
      ),
      'observation': time_step.observation[config.environment.observation_type],
      'reward': time_step.reward,
      'action': None,
      'sample': None,
      'sample_is_invalid': None,
      'sample_is_illegal': None,
      'sample_is_empty': None,
      'step_type': int(time_step.step_type),
  })

  score = sum(rewards[1:])  # Skip the first reward since it is always None.
  num_steps = len(rewards) - 1

  return (
      score,
      num_steps,
      num_invalid_actions,
      num_illegal_actions,
      num_empty_actions,
      demonstration_prompt
  )
