# Error-Propagation
Datatype to handle error Propagation using the rules for multiplication, division, addition, subtraction, and exponents. It also shows its work as you go so it's easier to learn and study.

## Motivation
Original project was for making a error propagation calculator that shows work to do these tedious calculations for me in my engineering classes
Years later, I've come back to update the project with auto-grad style automatic differentiation to clean up the edge cases (where the simplified multiplicitive, additive, etc. rules fail)
and to test error propagation on neural networks too. Inspired by Andrej Karpathy's micrograd video because it made me remember this project to redo it properly.

It works just by operator overloading all of the operators in the err class to break it up into smaller parts and then it just calculates it in the regular Python order of operations. It prints every step that it computes inside of each operator overload so that you can visualize what it is doing.

# How to run
A simple user interface is provided so you just have to type
$python3 errorProp.py

to start the program and then you enter a name for each variable then you enter the value then the ± uncertainty

Then you can hit enter to continue and type in your formula with Python syntax (** for powers or roots) and I imported pi that you can also use

The warnings I added occur when you use a number that isn't an err type like the constant pi, but it has no uncertainty so in this case it works fine.

# Example:

```py
└─$ python3 errorProp.py
Enter a variable; none to continue: r
r = 34
r = 34 ± .75
Enter a variable; none to continue: h
h = 124
h = 124 ± 2.5
Enter a variable; none to continue:

Type your equations below, and ctrl-c to exit

pi*r**2*h
r = 34.0 ± 0.75 ( % 2.2059 )
h = 124.0 ± 2.5 ( % 2.0161 )

derivative =  2.0 * 34.0 ^( 1.0 ) = 68.0
| 68.0 |* 0.75 = 51.0
/home/glenn/errorPropigation/errorProp.py:79: UserWarning: WARNING: types do not match, autoconverting with o.err=0
  warnings.warn("WARNING: types do not match, autoconverting with o.err=0")
sqrt(( 51.0 / 1156.0 )^2 + ( 0.0 / 3.141592653589793 )^2)* 3631.681107549801 = 160.22122533307945
sqrt(( 160.22122533307945 / 3631.681107549801 )^2 + ( 2.5 / 124.0 )^2)* 450328.4573361753 = 21843.68958900029
pi*r**2*h = 450328.4573361753 ± 21843.68958900029 ( % 4.8506 )
```
