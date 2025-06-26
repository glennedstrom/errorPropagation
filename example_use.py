from engine import err

a = err(5, .02)
b = err(10, .02)
c = err(3.1415926535)

out = a/b*c
out.backward()
print(a.grad)
print(out.get_err())
