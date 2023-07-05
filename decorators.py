def some_decorator(f):
    def wraps(*args):
        print(f"calling function '{f.__name__}'")
        return f(args)
    return wraps

@some_decorator
def decorated_function(x):
    print(f"With argument '{x}'")
