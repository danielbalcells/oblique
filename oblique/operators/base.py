from dataclasses import dataclass
from typing import List, Any

from langchain.prompts import load_prompt
from langchain.chains import LLMChain
from langchain.llms import OpenAI

from oblique.idea import Idea

DEF_TEMPLATE_PATH = 'templates/base_operator.yaml'
DEF_TEMPERATURE = 0.2
DEF_N_OUTS = 4

class BaseOperator:
    def __init__(self, temperature: float = DEF_TEMPERATURE, template_path: str = DEF_TEMPLATE_PATH) -> None:
        self.temperature = temperature
        self.template = load_prompt(template_path)
        self.output_parser = None
        self.llm = self.create_chain(self.temperature)
        self.module_name = 'Base Operator'
        self.input_description = ''
        self.output_description = ''
        self.instruction = ''

    def create_chain(self, temperature: float) -> OpenAI:
        llm = OpenAI(temperature=temperature)

    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature
        self.llm = self.create_llm(temperature)

    def run(self, idea: Idea) -> Any:
        raise NotImplementedError

    def format_prompt(self, idea: Idea) -> str:
        return self.template.format(idea=idea)

        
class OneToManyOperator(BaseOperator):
    def __init__(self, n_outs: int = DEF_N_OUTS, temperature: float = DEF_TEMPERATURE) -> None:
        self.temperature = temperature
        self.template = load_prompt(DEF_TEMPLATE_PATH)
        self.input_description = 'a single idea'
        self.output_description = f'a list of {n_outs} ideas'
