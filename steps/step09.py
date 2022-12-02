"""
제 1 고지 : 미분 자동 계산 
    step 9 : 함수를 더 편리하게 
        - 1. 파이썬 함수로 이용하기 
        - 2. y.grad = 1.0 생략 -> backward 간소화 
        - 3. np.ndarray 만 취급하기 ( 주의 해야 할것은 0차원 ndarray 인스턴스를 사용하여 계산하면 결과타입이 np.float32,64 가 되는 경우가 나온다)
        

"""
import numpy as np


def as_array(x):

    """
    0차원 ndarray / ndarray가 아닌 경우
    """
    if np.isscalar(x):
        return np.array(x)
    return x


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)}은(는) 지원하지 않습니다.")
        self.data = data
        self.grad = None  # gradient
        self.creator = None  # creator

    def set_creator(self, func) -> None:
        self.creator = func

    def backward(self):
        """
        자동 역전파 (반복)
        """
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()  # 1. 함수를 가져온다
            x, y = f.input, f.output  # 2. 함수의 입력 / 출력을 가져온다
            x.grad = f.backward(y.grad)  # 3. 역전파를 계산한다

            if x.creator is not None:
                funcs.append(x.creator)  # 하나 앞의 함수를 리스트에 추가한다.


class Function:
    """
    Function Base Class

    """

    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        self.input = input  # 역전파 계산을 위해 입력변수 보관
        output = Variable(as_array(y))
        output.set_creator(self)  # 출력 변수에 creator 설정 ( 연결을 동적으로 만드는 핵심)
        self.output = output  # 출력도 저장

        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        구체적인 함수 계산 담당
        """
        raise NotImplementedError()

    def backward(self, gy: np.ndarray) -> np.ndarray:
        """
        역전파
        """
        raise NotImplementedError()


class Square(Function):
    """
    y= x ** 2
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x**2

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = 2 * x * gy
        return gx


def square(x):
    f = Square()
    return f(x)


class Exp(Function):
    """
    y=e**x
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def exp(x):
    f = Exp()
    return f(x)


x = Variable(np.array(0.5))
A = Square()
B = Exp()
C = Square()

a = A(x)
b = B(a)
y = C(b)

## 자동 역전파
y.backward()
print(x.grad)
