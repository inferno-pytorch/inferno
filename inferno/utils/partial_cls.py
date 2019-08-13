import functools
import sys
import types
import inspect


__all__ =  [
    'partial_cls',
    'register_partial_cls'
]


def partial_cls(base_cls, name, module, fix=None, default=None):

    # helper function
    def insert_if_not_present(dict_a, dict_b):
        for kw,val in dict_b.items():
            if kw not in dict_a:
                dict_a[kw] = val
        return dict_a

    # helper function
    def insert_call_if_present(dict_a, dict_b, callback):
        for kw,val in dict_b.items():
            if kw not in dict_a:
                dict_a[kw] = val
            else:
                callback(kw)
        return dict_a

    # helper class
    class PartialCls(object):
        def __init__(self, base_cls, name, module, fix=None, default=None):

            self.base_cls = base_cls
            self.name = name
            self.module = module
            self.fix = [fix, {}][fix is None]
            self.default = [default, {}][default is None]

            if self.fix.keys() & self.default.keys():
                raise TypeError('fix and default share keys')

            # remove binded kw
            self._allowed_kw = self._get_allowed_kw()

        def _get_allowed_kw(self):

            
            argspec = inspect.getfullargspec(base_cls.__init__)
            args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = argspec

            if varargs is not None:
                raise TypeError('partial_cls can only be used if __init__ has no varargs')

            if varkw is not None:
                raise TypeError('partial_cls can only be used if __init__ has no varkw')

            if kwonlyargs is not None and kwonlyargs != []:
                raise TypeError('partial_cls can only be used without kwonlyargs')

            if args is None or len(args) < 1:
                raise TypeError('seems like self is missing')
            
            
            return [kw for kw in args[1:] if kw  not in self.fix]   
         

        def _build_kw(self, args, kwargs):
            # handle *args
            if len(args) > len(self._allowed_kw):
                raise TypeError("to many arguments")

            all_args =  {}
            for arg, akw in zip(args, self._allowed_kw):
                all_args[akw] = arg

            # handle **kwargs
            intersection = self.fix.keys() & kwargs.keys()
            if len(intersection) >= 1:
                kw = intersection.pop()
                raise TypeError("`{}.__init__` got unexpected keyword argument '{}'".format(name, kw))

            def raise_cb(kw):
                raise TypeError("{}.__init__ got multiple values for argument '{}'".format(name, kw))
            all_args = insert_call_if_present(all_args, kwargs, raise_cb)

            # handle fixed arguments
            def raise_cb(kw):
                raise TypeError()
            all_args = insert_call_if_present(all_args, self.fix, raise_cb)

            # handle defaults
            all_args = insert_if_not_present(all_args, self.default)

            # handle fixed 
            all_args.update(self.fix)

            return all_args

        def build_cls(self):

            def new_init(self_of_new_cls, *args, **kwargs):
                combined_args = self._build_kw(args=args, kwargs=kwargs)

                #call base cls init
                super(self_of_new_cls.__class__, self_of_new_cls).__init__(**combined_args)

            return type(name, (self.base_cls,), {
                '__module__': self.module,
                '__init__' : new_init
            })
            return cls


    return PartialCls(base_cls=base_cls, name=name, module=module,
        fix=fix, default=default).build_cls()


def register_partial_cls(base_cls, name, module, fix=None, default=None):
    module_dict = sys.modules[module].__dict__
    generatedClass = partial_cls(base_cls=base_cls,name=name, module=module,
        fix=fix, default=default)
    module_dict[generatedClass.__name__] = generatedClass
    del generatedClass


if __name__ == "__main__":

    class Conv(object):
        def __init__(self, dim, activation, stride=1):
            print(f"dim {dim} act {activation} stride {stride}")


    Conv2D = partial_cls(Conv,'Conv2D',__name__, fix=dict(dim=2), default=dict(stride=2))


    #obj =  Conv2D(activation='a')
    #obj =  Conv2D('a',activation='a', stride=3)
    obj =  Conv2D('fu','bar')    

