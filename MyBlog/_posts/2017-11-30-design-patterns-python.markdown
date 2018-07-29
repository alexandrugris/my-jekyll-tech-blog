---
layout: post
title:  "The Common Design Patters Implemented In Python [WIP]"
date:   2017-12-3 13:15:16 +0200
categories: architecture
---
An unfinished post aiming to cover the implementation in Python of the most commonly used design patterns.

### SOLID Principles

[SOLID Principles](https://en.wikipedia.org/wiki/SOLID_(object-oriented_design)): 

- *Single Responsibility Principle:* one reason for change
- *Open-Closed Principle:* open for extension, closed for modification
- *Liskov Substitution Principle:* objects should be replaceble by instances of their subtypes without affecting the corectness of the program
- *Interface Segregation Principle:* keep interfaces simple and focused
- *Dependency Injection Principle:* one should depend on abstractions, not on concretions

### Abstract Classes In Python

```python
import abc # support for abstract classes

class MyAbstractClass(metaclass=abc.ABCMeta):
    """Abstract base class definition"""

    @abc.abstractmethod
    def my_abstract_method(self, value):
        """Required Method"""

    @abc.abstractmethod
    def my_abstract_property(self):
        """Required Property"""

# failed_instantiation = MyAbstractClass()

class MyInstantiation(MyAbstractClass):
    """Implementation"""

    def my_abstract_method(self, value):
        return value

    @property
    def my_abstract_property(self):
        return self

successful_instantation = MyInstantiation()
```

### Strategy Pattern (Behavioral)

With duck-typing:

```python
class GenericCalculator:
    def __init__(self, strategy):
        self.strategy = strategy

    def compute(self, a, b):
        return self.strategy.doStrategy(a, b);

class CustomAdditionStrategy:
    def doStrategy(self, a, b):
        return a + b

gc = GenericCalculator(CustomAdditionStrategy())
print(gc.compute(1, 1))
```

Or with explicit typing:

```python
import abc

class IBiOperation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def doStrategy(self, a, b):
        """Required operation"""

class GenericCalculator:

    def __init__(self, strategy : IBiOperation):
        self.strategy = strategy

    def compute(self, a, b):
        return self.strategy.doStrategy(a, b);

class CustomAdditionStrategy:

    def doStrategy(self, a, b):
        return a + b

gc = GenericCalculator(CustomAdditionStrategy())
print(gc.compute(1, 1))
```

### Observer Pattern (Behavioral)

One to many / publish subscribe: when state of the object changes, all dependant objects are notified. The beauty is that the observed can become observer and vice-versa.

```python
class ObservedBase:

    def __init__(self):
        self.observers = []

    def attach(self, observer):
        self.observers.append(observer)

    def detach(self, observer):
        self.observers.remove(observer)

    def notify(self, change):
        for observer in self.observers:
            observer.notify(change)


class Observed(ObservedBase):

    def set_value(self, value):
        self.notify(value)

class Observer:

    def __init__(self, name):
        self.name = name

    def notify(self, value):
        print(self.name + ":" + value)

observed = Observed()
o1 = Observer("1")
o2 = Observer("2")

observed.attach(o1)
observed.attach(o2)
observed.notify("all")

observed.detach(o1)
observed.notify("o2")
```

### Command Pattern (Behavioral)

```python
class AbstractCmd:
    def init_command(self, args):
        pass

    def do_command(self):
        pass

class FirstCommand(AbstractCmd):
    """This is my first command"""

    def init_command(self, *args):
        print("Command arguments: " + str(args))

    def do_command(self):
        print("Doing first command")

class NoCommand(AbstractCmd):
    """This is the default, does nothing"""

    def init_command(self, args):
        print ("Do nothing init")

    def do_command(self):
        print ("Nothing")


class CmdRepository:

    @classmethod
    def init(cls):
        cls.factory = {name: value
                 for (name, value) in globals().items()
                 if(isinstance(value, type) and issubclass(value, AbstractCmd))
                }

    @classmethod
    def get_cmd(cls, cmd_name, *args):
        ret = cls.factory.setdefault(cmd_name, NoCommand)()
        ret.init_command(args)
        return ret


CmdRepository.init();

cmd = CmdRepository.get_cmd('FirstCommand', "First Argument", "Second Argument")
cmd.do_command()
```

The code above incorporates ideas from the Factory pattern (the `CmdRepository` class with loops through the loaded modules) and the Null Object pattern (the `NoCommand` class, returned in case the factory cannot create the desired command)

### Singleton (Creational)

Uses metaclasses to customize class creation.

```python
class Singleton(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class MySingleton(metaclass=Singleton):

    def __init__(self):
        self.my_attribute = "Hello World"


class MySecondSingleton(metaclass=Singleton):

    def __init__(self):
        self.my_attribute = "Hello World"

i1 = MySingleton()
i2 = MySingleton()
i3 = MySecondSingleton()

assert i1 == i2
assert i3 != i2
```

### Builder (Creational)

Factors out the the process of building an instance of a class. Instantiation can be a multi-step, async process. Same steps, but with different results.

```python
class ThreeStepBuilder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def step_1(self, obj):
        """Step 1"""

    @abc.abstractmethod
    def step_2(self, obj):
        """Step 2"""


    @abc.abstractmethod
    def step_3(self, obj):
        """Step 3"""


class Concrete3StepBuilder(ThreeStepBuilder):

    def step_1(self, obj):
        obj.step_1 = "Done"

    def step_2(self, obj):
        obj.step_2 = "Done"

    def step_3(self, obj):
        obj.step_3 = "Done"

class TheObject:

    def __str__(self):
        return self.step_1 + " / " + self.step_2 + " / " + self.step_3;

    pass

class Director:
    """ The creator itself. Process the same, different parameters depending on the builder object """

    def __init__(self, builder : ThreeStepBuilder):
        self.object = None
        self.builder = builder

    def get_built(self):
        return self.object

    def build(self):
        ret = TheObject()

        self.builder.step_1(ret)
        self.builder.step_2(ret)
        self.builder.step_3(ret)

        # only set the fully constructed object
        self.object = ret

        return ret

dir = Director(Concrete3StepBuilder())
dir.build();

print(dir.get_built())
```





