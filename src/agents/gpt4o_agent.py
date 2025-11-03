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
      
      raw_output = response.choices[0].message.content.strip()

      # Check for the unique keywords in the heuristic generation prompt
      if "Suboptimal Action:" in prompt_text or "Distilled Heuristic:" in prompt_text:
        # This is a heuristic generation call. Return the raw text.
        return raw_output
      else:
        # This is a game-playing call. Parse for the *last* "Action:"
        # This is more robust for Chain of Thought (CoT)
        if 'Action:' in raw_output:
            parts = raw_output.rsplit('Action:', 1)
            if len(parts) > 1:
                return parts[1].strip() # Return the final action
            else:
                # "Action:" was in the text, but not as a final answer
                # This can happen if CoT is off.
                return raw_output # Return the raw text as a fallback
        else:
            # No "Action:" found (e.g., CoT is off), return the whole string
            return raw_output
      # --- END MODIFICATION ---

    except Exception as e:
      print(f"An API error occurred: {e}")
      return "" # Return an empty string on failure
     
        
    