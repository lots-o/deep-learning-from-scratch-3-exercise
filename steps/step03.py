"""
제 1 고지 : 미분 자동 계산 
    step 3 : 함수 연결 
        - 여러 함수를 연결 (합성함수)하여 복잡한 함수식이더라도 계산그래프를 통해 각 변수에 대한 미분을 효율적으로 계산
        - 즉, 변수별 미분을 계산하는 역전파를 구현할수 있도록 한다 
"""
from turtle import forward
import numpy as np


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class Function:
    """
    Function Base Class

    """

    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        return Variable(y)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        구체적인 함수 계산 담당
        """
        raise NotImplementedError


class Square(Function):
    """
    y= x ** 2
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x**2


class Exp(Function):
    """
    y=e**x
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
c = C(b)

print(c.data)
