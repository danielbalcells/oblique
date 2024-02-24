from dataclasses import dataclass, field
from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain_core.output_parsers import JsonOutputParser

from oblique.idea import Idea

# DEF_TEMPLATE_PATH = 'templates/base_operator.yaml'
DEF_TEMP = 0.2
        
@dataclass
class OneToOneOperator:
    temperature: float = DEF_TEMP
    instruction: str = field(default=None, init=False)
    prompt: PromptTemplate = field(init=False)
    parser: JsonOutputParser = field(default_factory=JsonOutputParser, init=False)
    chain: LLMChain = field(init=False)
    def __post_init__(self) -> None:
        self.prompt = PromptTemplate.from_template(
            '''
            You are running a small component of a subversive modular system for creative idea manipulation.
            Your particular module is called {module_name}.
            It accepts a single idea as input, and generates a single idea as output.
            The instructions to generate the output are as follows:
            {instruction}
            
            Respond only with the single output idea string in clean JSON and nothing more.
            Do not include bullets, numbering or any other formatting in the individual output strings.
            The input is:
            {input}
            Output the single idea as a string in valid JSON.
            ''' 
        )
        self.chain = self.create_chain(self.temperature)

    def create_chain(self, temperature: float) -> PromptTemplate:
        llm = OpenAI(temperature=temperature)
        return self.prompt | llm | self.parser

    def run(self, input: Idea) -> Idea:
        if not self.instruction:
            raise ValueError()
        args = {
            "input": input.get_text(),
            "module_name": self.__class__.__name__,
            "instruction": self.instruction
        }
        result = self.chain.invoke(args)
        return Idea(result)

@dataclass
class CounterArgument(OneToOneOperator):
    instruction: str = 'Generate a counter-argument to the input idea.'

@dataclass
class SupportArgument(OneToOneOperator):
    instruction: str = 'Provide a reason why the input idea might be true, based on factors besides the input idea itself.'

@dataclass
class Answer(OneToOneOperator):
    instruction: str = 'Provide an answer to the input question.'