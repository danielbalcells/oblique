from dataclasses import dataclass, field
from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain_core.output_parsers import JsonOutputParser

from oblique.idea import Idea, IdeaList

# DEF_TEMPLATE_PATH = 'templates/base_operator.yaml'
DEF_TEMP = 0.2
        
@dataclass
class ManyToOneOperator:
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
            It accepts multiple ideas as input, and generates a single idea as output.
            The instructions to generate the output are as follows:
            {instruction}
            
            Respond only with the single output idea string in clean JSON and nothing more.
            Do not include bullets, numbering or any other formatting in the individual output strings.
            The inputs are:
            {input}
            Output the single idea as a string in valid JSON.
            ''' 
        )
        self.chain = self.create_chain(self.temperature)

    def create_chain(self, temperature: float) -> PromptTemplate:
        llm = OpenAI(temperature=temperature)
        return self.prompt | llm | self.parser

    def run(self, input: IdeaList) -> Idea:
        if not self.instruction:
            raise ValueError()
        input_str = '\n'.join([idea.get_text() for idea in input])
        args = {
            "input": input_str,
            "module_name": self.__class__.__name__,
            "instruction": self.instruction
        }
        result = self.chain.invoke(args)
        return Idea(result)

@dataclass
class CommonDenominator(ManyToOneOperator):
    instruction: str = 'Find the common denominator between the input ideas.'

@dataclass
class ParentCategory(ManyToOneOperator):
    instruction: str = 'The input ideas all belong to a category. Find that category.'