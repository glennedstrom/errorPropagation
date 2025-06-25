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

# I removed sig figs because they are not relevant to this update and didn't work before

class err():
    """
    Error Propagation Calculating type

    err(number, ±error)

    Overloaded math functions to auto-calculate error:
    * / + - **
    """

    def __init__(self, val, err=0.0):
        # warn if the input wasn't a string originally
        self.val = float(val)   # number
        self.err = float(err)   # plus/minus error

    @classmethod
    def _ensure_err(cls, o):
        if not isinstance(o, cls):
            #warnings.warn("WARNING: types do not match, autoconverting with o.err=0")
            return cls(str(o), 0)
        return o

    # operation overloading

    def __add__(self, o):
        ans = self.val + o.val

        return err(ans, math.sqrt(self.err**2 + o.err**2))
    __radd__ = __add__

    def __sub__(self, o):
        o = self._ensure_err(o)

        ans = self.val - o.val

        return err(ans, math.sqrt(self.err**2 + o.err**2))

    def __rsub__(self, o):
        o = self._ensure_err(o)

        ans = o.val - self.val

        return err(ans, math.sqrt(self.err**2 + o.err**2))

    def __mul__(self, o):
        o = self._ensure_err(o)

        ans = self.val * o.val

        error = math.sqrt((self.err/self.val)**2 + (o.err/o.val)**2)*ans
        return err(ans, error)
    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._ensure_err(o)
        ans = self.val / o.val
        error = math.sqrt((self.err/self.val)**2 + (o.err/o.val)**2)*ans
        return err(ans, error)

    def __rtruediv__(self, o):
        o = self._ensure_err(o)
        ans =  o.val / self.val
        error = math.sqrt((self.err/self.val)**2 + (o.err/o.val)**2)*ans
        return err(ans, error)

    # TODO: Redo these with autograd, then exponents won't have to be exact
    def __pow__(self, o):
        if type(o) != err: #exponents need to be exact
            o = err(str(o), 0)
        elif o.err != 0:# x**n; where n needs to have no error
            warnings.warn("WARNING: your exponent needs to be exact; ignoring error...")
        ans = self.val ** o.val
        derivative = o.val*self.val**(o.val-1)
        error = abs(derivative)*self.err
        return err(ans, error)

    def __rpow__(self, o):
        if type(o) != err: #exponents need to be exact
            o = err(str(o), 0)
        elif self.err != 0:# x**n; where n needs to have no error
            warnings.warn("WARNING: your exponent needs to be exact; ignoring error...")

        ans =  o.val ** self.val
        derivative = self.val*o.val**(self.val-1)
        error = abs(derivative)*o.err
        return err(ans, error)
    #output formatting
    #def __str__(self):
    #    return str(self.val) + " ± " + str(self.err) + " ( % " + str(round(self.err/self.val*100,4)) + " )"

    def __repr__(self):
        return "err(" + str(self.val) + ", err=" + str(self.err) + ")"

