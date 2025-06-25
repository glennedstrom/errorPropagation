import re
from errorProp import err

def main():
    while True:
        try:# for bad user input
            var = input("Enter a variable; none to continue: ")
            if var == '':
                break

            val = input(var + " = ")
            e = input(var + " = " + val + " ± ")

            err.vars[var] = err(val, e)

        except Exception as e:
            print(e)
    print("\nType your equations below, ctrl-c to exit\n")
    converted = ''
    while True:
        try:
            eq = input()
            if eq == '':#loop back to variable definition
                break

            converted = eq
            pattern = re.compile(r'(' + '|'.join(sorted(err.vars.keys(),key=len, reverse=True)) + r')') #match the keys in the order of their length longest to shortest to avoid repeats
            converted = pattern.sub(r'err.vars["\1"]', eq)
            [print(i, "=", err.vars[i]) for i in set(pattern.findall(eq))]
            print()

            print(eq + ' = ' + str(eval(converted))) # don't use eval unless you are the only user 
            print("\n")

        except Exception as e:
            print(str(e) + ": " + str(converted))
    main()#recursive call so you can go back and change variables


if __name__ == "__main__":
    main()

