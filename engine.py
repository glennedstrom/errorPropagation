import math

"""
# ErrorGrad

The motivation for this autograd update is to:
1. Proprely handle backpropagation in all edge cases by computing the partial derivatives properly
2. I want to see how error propagates through neural networks to find out if the result still makes sense
- Logically, it is just using the error with the rate of change of the prediction with respect to the input parameters, so it should work.
"""

#NOTE: To know the error of your predicted values, you need to do y_pred.backward() instead of loss.backward()
# This is because you most likely want to know the error of your predicted values rather than of the loss function
#WARN: If you compute both the loss.backward() to update gradients and y_pred.backward(), make sure to zero_grad before every call to .backward()

class err():
    """
    Error Propagation Calculating type

    err(number, ±error)

    Overloaded math functions to auto-calculate error:
    * / + - **
    """

    def __init__(self, val, err=0.0, _children=(), _op=''):
        # warn if the input wasn't a string originally
        self.val = val   # number
        self.err = float(err)   # plus/minus error
        self._op = _op
        self._children = set(_children)
        self._backward = lambda: None
        self._input_nodes = set() # for finding all input nodes for error prop
        self.grad = 0

    def _topsort(self, children, visited, input_nodes):
        if self in visited:
            return
        visited.add(self)
        if self._op == '':
            input_nodes.add(self)

        for child in self._children:
            child._topsort(children, visited, input_nodes)
        children.append(self)

    def backward(self):
        self.grad = 1
        #top sort the nodes
        children = []
        self._topsort(children, set(), self._input_nodes)
        #call _backward() in order on all nodes
        for child in children[::-1]:
            child._backward()
        self._set_err()

    def zero_grad(self):
        #zero out all the gradients; lazy method
        self.grad = 0

        children = []
        self._topsort(children, set(), self._input_nodes)
        self._input_nodes = set() # reset input nodes too

        for child in children[::-1]:
            child.grad = 0

    @classmethod
    def _ensure_err(cls, o):
        if not isinstance(o, cls):
            return cls(o, 0)
        return o

    def _set_err(self):
        """ Compute the error for the current node assuming gradients are correct """
        # use _input_nodes to get the error
        self.err = 0
        for node in self._input_nodes:
            self.err += (node.grad * node.err)**2
        self.err = self.err**0.5

    # operation overloading

    def __add__(self, o):
        ans = self.val + o.val
        out = err(ans, _children=(self,o), _op='+')
        def _backward():
            self.grad += out.grad
            o.grad += out.grad

        out._backward = _backward
        return out
    __radd__ = __add__

    def __sub__(self, o):
        o = self._ensure_err(o)
        ans = self.val - o.val
        out = err(ans, _children=(self,o), _op='-')
        def _backward():
            self.grad += out.grad
            o.grad += -out.grad
        out._backward = _backward

        return out

    def __rsub__(self, o):
        o = self._ensure_err(o)
        ans = o.val - self.val
        out = err(ans, _children=(self,o), _op='-')
        def _backward():
            o.grad += out.grad
            self.grad += -out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        ans = -self.val
        out = err(ans, _children=(self,), _op='-')
        def _backward():
            self.grad += -out.grad
        out._backward = _backward

        return out

    def __mul__(self, o):
        o = self._ensure_err(o)
        ans = self.val * o.val
        out = err(ans, _children=(self,o), _op='*')
        def _backward():
            o.grad += self.val * out.grad
            self.grad += o.val * out.grad
        out._backward = _backward

        return out
    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._ensure_err(o)
        ans = self.val / o.val
        out = err(ans, _children=(self,o), _op='/')
        def _backward():
            o.grad += -self.val / (o.val**2) * out.grad
            self.grad += (1/o.val) * out.grad
        out._backward = _backward
        return out

    def __rtruediv__(self, o):
        o = self._ensure_err(o)
        ans =  o.val / self.val
        out = err(ans, _children=(self,o), _op='/')
        def _backward():
            self.grad += -o.val / (self.val**2) * out.grad
            o.grad += (1/self.val) * out.grad
        out._backward = _backward
        return out

    #TODO: Make integer and float powers different so negative bases don't cause math.log(negative_val) to crash the program.
    def __pow__(self, o): # INT o ONLY
        if type(o) != int:
            raise TypeError("Only integer exponents are supported")
        ans = self.val ** o
        out = err(ans, _children=(self,), _op='**')
        def _backward():
            self.grad += o * self.val ** (o-1) * out.grad
        out._backward = _backward
        return out
        # ans = self.val ** o.val
        # out = err(ans, _children=(self,o), _op='**')
        # def _backward():
        #     o.grad += math.log(self.val)*self.val**(o.val) * out.grad
        #     self.grad += o.val * self.val ** (o.val-1) * out.grad
        # out._backward = _backward
        # return out

    def __rpow__(self, o):
        o = self._ensure_err(o)
        ans =  o.val ** self.val
        out = err(ans, _children=(self,o), _op='**')
        def _backward():
            self.grad += math.log(o.val)*o.val**(self.val) * out.grad
            o.grad += self.val * o.val ** (self.val-1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        ans = math.tanh(self.val)
        out = err(ans, _children=(self,), _op='tanh')
        def _backward():
            self.grad += (1-math.tanh(self.val)**2) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        ans = max(0,self.val)
        out = err(ans, _children=(self,), _op='relu')
        def _backward():
            self.grad += (0 if ans == 0 else 1) * out.grad
        out._backward = _backward

        return out

    # def exp(self):
    #     ans = math.exp(self.val)
    #     out = err(ans, _children=(self,), _op='exp')
    #     def _backward():
    #         self.grad += ans * out.grad
    #     out._backward = _backward

    #     return out

    def sigmoid(self):
        ans = 1/(1+math.exp(-self.val))
        out = err(ans, _children=(self,), _op='exp')
        def _backward():
            self.grad += ans * (1-ans) * out.grad
        out._backward = _backward

        return out


    #output formatting
    # def __str__(self):
    #     # BUG: self.val == 0 means divide by 0
    #     return f'{self.val} ± {self.err} ( % {round(self.err/self.val*100,4)} )'

    def __repr__(self):
        return f'err({self.val}, err={self.err})'



if __name__ == "__main__":
    y_true=158
    a = err(5, .02)
    b = err(10, .02)
    ab = a*b
    c = err(3.1415926535)
    y_pred = ab*c

    L = 1/1*(y_true-y_pred)**2

    # Backprop on y_pred to calculate the error of y_pred
    y_pred.zero_grad()
    y_pred.backward()
    print(f'{str(y_pred)=}')
    print(f'{y_pred=}')

    # Normal backprop to update weights:
    L.zero_grad()
    L.backward()
    print(f'{str(L)=}')
    print(f'{L=}')
    # optimizer.step()
    
    # print(f'{a.grad=}, {b.grad=}, {c.grad=}')
    # # proof with definition of the derivative
    # h = 0.00001
    # print((((5+h)*(10)*(3.1415926535))-(5*10*3.1415926535))/h)
    # print((((5)*(10+h)*(3.1415926535))-(5*10*3.1415926535))/h)
    # print((((5)*(10)*(3.1415926535+h))-(5*10*3.1415926535))/h)
    # print()
    # print("Error of the solution y_pred and the percent error:")
    # print(y_pred.err)
    # print(y_pred.err/y_pred*100)
    #
    #
