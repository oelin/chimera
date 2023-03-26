class Module:

    @property
    def parameters(self):
        return tuple()
    
    def __call__(self, *args):
        return self.forward(*args)

    def forward(self):
        raise NotImplementedError()
