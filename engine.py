import math
import warnings

"""
# ErrorGrad

The motivation for this autograd update is to:
1. Proprely handle backpropagation in cases like: L = ax + bx
- The first one would think dL/dx = a
- The second one would think dL/dx = b;
- Actual answer: dL/dx == a + b
2. I want to see how error propagates through neural networks to find out if the result is nonsensical.
- The more layers you have, the more calculations so possibly the more error? 
- If computed with the partial derivative of the output w.r.t inputs, it seems like it should work no matter what because it is based on how much a small change in the inputs affects the outputs)
"""
#NOTE: To know the error of your predicted values, you need to do y_pred.backward() instead of loss.backward()
# This is because you likely want to know the error of your predicted values rather than of the loss function
#WARN: If you compute both the loss.backward() to update gradients and y_pred.backward(), make sure to zero_grad before every call to .backward()


#NOTE: Use the local errors and gradients to propagate errors through the calculation tree.
# EX: 
# x1 and x2 have some error
# you did y_pred.backward()
# Use all input variables & uncertainties with their partial derivatives dy_pred/dx1, dy_pred/dx2 to compute

# I removed sig figs because they are not relevant to this update and didn't work before

class err():
    """
    Error Propagation Calculating type

    err(number, ±error)

    Overloaded math functions to auto-calculate error:
    * / + - **
    """

    def __init__(self, val, err=0.0, _children=(), _op=''):
        # warn if the input wasn't a string originally
        self.val = float(val)   # number
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
            #warnings.warn("WARNING: types do not match, autoconverting with o.err=0")
            return cls(str(o), 0)
        return o

    def get_err(self):
        # use _input_nodes to get the error
        total = 0
        for node in self._input_nodes:
            total += (node.grad * node.err)**2
        total = total**0.5
        return total

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
        # def _backward():
        #     o.grad += self.val * out.grad
        #     self.grad += o.val * out.grad
        # out._backward = _backward
        return out

    def __rtruediv__(self, o):
        o = self._ensure_err(o)
        ans =  o.val / self.val
        out = err(ans, _children=(self,o), _op='/')
        # def _backward():
        #     o.grad += self.val * out.grad
        #     self.grad += o.val * out.grad
        # out._backward = _backward
        return out

    # TODO: Redo these with autograd, then exponents won't have to be exact
    def __pow__(self, o):
        o = self._ensure_err(o)
        ans = self.val ** o.val
        out = err(ans, _children=(self,o), _op='**')
        # def _backward():
        #     o.grad += self.val * out.grad
        #     self.grad += o.val * out.grad
        # out._backward = _backward
        return out

    def __rpow__(self, o):
        o = self._ensure_err(o)
        ans =  o.val ** self.val
        out = err(ans, _children=(self,o), _op='**')
        # def _backward():
        #     o.grad += self.val * out.grad
        #     self.grad += o.val * out.grad
        # out._backward = _backward
        return out
    #output formatting
    #def __str__(self):
    #    return str(self.val) + " ± " + str(self.err) + " ( % " + str(round(self.err/self.val*100,4)) + " )"

    def __repr__(self):
        return "err(" + str(self.val) + ", err=" + str(self.err) + ")"



if __name__ == "__main__":
    y_true=157
    a = err(5, .02)
    b = err(10, .02)
    ab = a*b
    c = err(3.1415926535)
    y_pred = ab*c

    L = 1/1*(y_true-y_pred)**2

    # Normal backprop to update weights:
    # L.zero_grad()
    # L.backward()
    # optimizer.step()
    print(L)
    #
    # # Get Uncertainty:
    # y_pred.zero_grad()
    # y_pred.backward()
    y_pred.backward()
    # y_pred.get_err() # can do this for every variable; This is a function that computes the error assuming the gradient was calculated with y_pred (if you want your prediction's uncertainty)
    print(a.grad, b.grad, c.grad, ab.grad, y_pred.grad)
    print(y_pred.get_err())

    # proof with definition of the derivative
    h = 0.00001
    print((((5+h)*(10)*(3.1415926535))-(5*10*3.1415926535))/h)
    print((((5)*(10+h)*(3.1415926535))-(5*10*3.1415926535))/h)
    print((((5)*(10)*(3.1415926535+h))-(5*10*3.1415926535))/h)
    print()
    print("Error of the solution y_pred and the percent error:")
    print(y_pred.get_err())
    print(y_pred.get_err()/y_pred*100)
    

