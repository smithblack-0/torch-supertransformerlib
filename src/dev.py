import torch
from typing import Generic, TypeVar, List

T = TypeVar('T')

from typing import TypeVar, List, Generic

T = TypeVar('T')


class MyList(Generic[T]):
    def __init__(self, items: List[T]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> T:
        return self.items[index]

    def append(self, item: T) -> None:
        self.items.append(item)


def my_list(x: List[T]) -> MyList[T]:
    return MyList(x)

def my_list(x: List[T]) -> MyList[T]:
    return MyList(x)

# Example usage
data = [1, 2, 3]
mylist = my_list(data)
print(mylist)  # MyList([1, 2, 3])


my_list_scripted = torch.jit.script(my_list)

x = torch.tensor([1, 2, 3])
mylist = my_list_scripted(x.tolist())
print(mylist)  # MyList([1, 2, 3])



