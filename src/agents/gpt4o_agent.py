import os
import dataclasses
import re
from typing import Any, Mapping

import openai
import numpy as np

from lm_act.src import config as config_lib
from lm_act.src import interfaces

@dataclasses.dataclass(frozen=True, kw_only=True)
class GPT4oAgentConfig(config_lib.Agent):
  name: str = 'gpt4o_agent'
  model_name: str = 'gpt-4o' 

class GPT4oAgent(interfaces.Agent):
  """An agent that calls an external LLM API."""

  def __init__(self, config: GPT4oAgentConfig) -> None:
    self.client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    self.model_name = config.model_name

  def step(
      self,
      observation: Mapping[str, Any],
      environment: interfaces.Environment,
      rng: np.random.Generator,
  ) -> str:
    """Constructs the prompt and calls the LLM API."""
    
    prompt_text = observation['prompt']
    
    try:
      response = self.client.chat.completions.create(
          model=self.model_name,
          messages=[
              {"role": "user", "content": prompt_text}
          ]
      )
      
      action = response.choices[0].message.content.strip()
      
      if 'Action:' in action:
          action = action.split('Action:')[1].strip()
          
      return action

    except Exception as e:
      print(f"An API error occurred: {e}")
      return "" # Return an empty string on failure
     
        
    