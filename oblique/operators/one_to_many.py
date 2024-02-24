from dataclasses import dataclass, field
from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain_core.output_parsers import JsonOutputParser

from oblique.idea import Idea, IdeaList

# DEF_TEMPLATE_PATH = 'templates/base_operator.yaml'
DEF_TEMP = 0.2
DEF_N_OUTS = 4
        
@dataclass
class OneToManyOperator:
    N: int = DEF_N_OUTS
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
            It accepts a single idea as input, and generates {N} ideas as outputs.
            The instructions to generate the output are as follows:
            {instruction}
            
            Respond only with the output list of {N} ideas in clean JSON and nothing more.
            Do not include bullets, numbering or any other formatting in the individual output strings.
            The input is: {input}
            Output the {N} output ideas as a list of strings in valid JSON.
            ''' 
        )
        self.chain = self.create_chain(self.temperature)

    def create_chain(self, temperature: float) -> PromptTemplate:
        llm = OpenAI(temperature=temperature)
        return self.prompt | llm | self.parser

    def run(self, input: Idea, N: int = DEF_N_OUTS) -> IdeaList:
        if not self.instruction:
            raise ValueError()
        args = {
            "input": input.get_text(),
            "N": N,
            "module_name": self.__class__.__name__,
            "instruction": self.instruction
        }
        results = self.chain.invoke(args)
        return IdeaList([Idea(result) for result in results])

@dataclass
class BranchOut(OneToManyOperator):
    instruction: str = 'Branch out from the input idea, ensuring that each of the output ideas covers a different aspect, property or quality of the input, taking it into different directions.'

@dataclass
class GetExamples(OneToManyOperator):
    instruction: str = 'The input term defines a category. Provide examples of things that belong to that category.'

@dataclass
class GetSupportingArguments(OneToManyOperator):
    instruction: str = 'Provide supporting arguments that provide reasons why the input idea might be true or valid.'

@dataclass
class GetCounterArguments(OneToManyOperator):
    instruction: str = 'Provide counter-arguments that provide reasons why the input idea might be false or invalid.'

@dataclass
class GetBreakdownQuestions(OneToManyOperator):
    instruction: str = 'Generate questions that help break down the input idea into smaller, more actionable questions that are more limited in scope and are easier to answer.'

