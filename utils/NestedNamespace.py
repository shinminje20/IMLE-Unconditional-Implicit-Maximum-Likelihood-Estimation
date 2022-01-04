import os

class NestedNamespace():
    """Class representing a NestedNamespace. Any value that can be interpreted
    as a dictionary (dictionary, JSON file, anything with a __dict__) attribute
    is pulled into the hierarchy as a NestedNamespace itself. There are three
    cases for adding a new item:

    1. If a key is None or ends with config and maps to a dictionary-like value,
        then the key-value pairs in the value are added to the hierarchy where
        the key would have been, and the key isn't added.
    2. The value is dictionary-like, but case (1) doesn't hold. Then the
        key-value pairs in the value are added below the key in the hierarchy.
    3. Otherwise, the key value pair is simple and added to the hierarchy.
    """
    def __init__(self, *inputs):
        for input in inputs:
            if isinstance(input, dict):
                for k,v in input.items():
                    self.add_item(k, v)
            elif hasattr(input, "__dict__"):
                for k,v in input.__dict__.items():
                    self.add_item(k, v)
            elif (isinstance(input, str) and input.endswith(".json")
                and os.path.exists(input)):
                with open(input, "r") as f:
                    for k,v in json.load(f).items():
                        self.add_item(k, v)
            else:
                raise ValueError(f"Not able to build {input} into Namespace")

    def __str__(self):
        return f"NestedNamespace({', '.join([f'{k}={str(v)}' for k,v in self.__dict__.items()])})"

    def __repr__(self): return self.__str__()

    def set_attribute(self, k, v):
        """Creates attribute [key] set to value [value] or raises an error if
        an attribute named [key] already exists.
        """
        if hasattr(self, k):
            raise ValueError(f"key {k} used twice")
        else:
            setattr(self, k, v)

    def add_item(self, k, v):
        """Adds key [k] and value [v] to the NestedNamespace."""
        if k.endswith("config") and NestedNamespace.is_dict_like(v):
            for k_,v_ in NestedNamespace(v).__dict__.items():
                self.set_attribute(k_, v_)
        elif NestedNamespace.is_dict_like(v):
            self.set_attribute(k, NestedNamespace(v))
        else:
            self.set_attribute(k, v)

    @staticmethod
    def is_dict_like(x):
        """Returns if [x] can be interpreted like a dictionary."""
        return ((hasattr(x, "__dict__"))
            or (isinstance(x, dict))
            or (isinstance(x, str) and x[:-5] == ".json" and os.path.exists(x)))
