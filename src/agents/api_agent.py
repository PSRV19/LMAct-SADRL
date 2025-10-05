# In src/agents/api_agent.py

import os
import dataclasses
import re
from typing import Any, Mapping

import openai
import numpy as np

from lm_act.src import config as config_lib
from lm_act.src import interfaces

@dataclasses.dataclass(frozen=True, kw_only=True)
class ApiAgentConfig(config_lib.Agent):
  name: str = 'api_agent'
  model_name: str = 'google/gemini-2.5-flash-lite-preview-09-2025' # Or other model

class ApiAgent(interfaces.Agent):
  """An agent that calls an external LLM API."""

  def __init__(self, config: ApiAgentConfig) -> None:
    self.client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        default_headers={
            # Recommended by OpenRouter to identify the app
            "HTTP-Referer": "https://github.com/google-deepmind/lm_act", 
            "X-Title": "LMAct Benchmark",
        },
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
      
      # Post-process the output to find the "Action:" keyword
      if 'Action:' in action:
          action = action.split('Action:')[1].strip()
          
      return action

    except Exception as e:
      print(f"An API error occurred: {e}")
      return "" # Return an empty string on failure
     
        
    