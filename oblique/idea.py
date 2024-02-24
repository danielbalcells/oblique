from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Type, Union

if TYPE_CHECKING:
    from oblique.operators.one_to_one import OneToOneOperator
    from oblique.operators.one_to_many import OneToManyOperator
    from oblique.operators.many_to_one import ManyToOneOperator

@dataclass
class Idea:
    text: str
     
    def __init__(self, text: str) -> None:
        self.text = text
    
    def get_text(self) -> str:
        return self.text

    def apply(self, operator: Union['OneToOneOperator', Type['OneToOneOperator']]) -> 'Idea':
        if isinstance(operator, type):
            operator = operator()
        return operator.run(self)

    def expand(self, operator: Union['OneToManyOperator', Type['OneToManyOperator']]) -> 'IdeaList':
        if isinstance(operator, type):
            operator = operator()
        return operator.run(self)


class IdeaList(list[Idea]):

    def apply(self, operator: Union['OneToOneOperator', Type['OneToOneOperator']]) -> 'IdeaList':
        if isinstance(operator, type):
            operator = operator()
        return IdeaList([operator.run(idea) for idea in self])

    def reduce(self, operator: Union['ManyToOneOperator', Type['ManyToOneOperator']]) -> 'Idea':
        if isinstance(operator, type):
            operator = operator()
        return operator.run(self)

