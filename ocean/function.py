from . variable import Variable


@dataclass
class Function:
    def __call__(self, *variables) -> Variable:

        result = self.forward(*variables)

        def back():
            self.backward(*variables, result.grad)

            for variable in variables:
                variable.backward(0.)
        
        result.back = back 
        return result
