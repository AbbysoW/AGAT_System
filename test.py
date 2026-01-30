def fun1(func):
    num = func()
    return num + 5

@fun1
def fun2():
    return 10 

print(fun2)  # This will print 15