from copy import deepcopy
import os
from tqdm import tqdm

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

    def __str__(self): return f"NestedNamespace({', '.join([f'{k}={str(v)}' for k,v in self.__dict__.items()])})"
    def __repr__(self): return self.__str__()
    def __eq__(self, other): return self.__repr__() == other.__repr__()


    def set_attribute(self, k, v, override=False):
        """Creates attribute [key] set to value [value].

        Args:
        k           -- the key to set
        v           -- the value to set
        overrride   -- whether or not to allow setting an already existing key
        """
        if hasattr(self, k) and not override:
            raise ValueError(f"key {k} used twice")
        else:
            setattr(self, deepcopy(k), deepcopy(v))

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
    def to_dict(x, allow_identity=False):
        """Returns the NestedNamespace [x] as a dictionary."""
        if allow_identity and not isinstance(x, NestedNamespace):
            return deepcopy(x)
        elif isinstance(x, NestedNamespace):
            return {deepcopy(k): NestedNamespace.to_dict(v, allow_identity=True)
                for k,v in x.__dict__.items()}
        else:
            raise TypeError(f"'{x}' is of type {type(x)} not NestedNamespace")

    @staticmethod
    def leaf_union(x, y):
        """Returns a NestedNamespace that is the union of NestedNamespaces [x]
        and [y].

        When the values of an attribute in both [x] and [y] are
        NestedNamespaces, then the returned NestedNamespace has this attribute
        set to leaf_union(x.attribute, y.attribute).

        Otherwise, precedence is given to [y], so when [x] and [y] have
        different values for an attribute, the returned NestedNamespace has that
        attribute set to the value of that attribute in [y].
        """
        x, y = NestedNamespace(x), NestedNamespace(y)
        result = NestedNamespace(x)

        for k,y_v in y.__dict__.items():
            x_v = x.__dict__[k] if k in x.__dict__ else None
            if (NestedNamespace.is_dict_like(x_v)
                and NestedNamespace.is_dict_like(y_v)):
                new_val = NestedNamespace.leaf_union(x_v, y_v)
                result.set_attribute(k, new_val, override=True)
            else:
                result.set_attribute(k, y_v, override=True)

        return result

    @staticmethod
    def is_dict_like(x):
        """Returns if [x] can be interpreted like a dictionary."""
        return ((hasattr(x, "__dict__"))
            or (isinstance(x, dict))
            or (isinstance(x, str) and x[:-5] == ".json" and os.path.exists(x)))

if __name__ == "__main__":
    tqdm.write("----- Tested NestedNamespace -----")

    tqdm.write("\n Testing leaf_union()")
    x = NestedNamespace({
        "A": 1,
        "B": 2,
        "C": {"D": 4, "E": {"test": 0}}
    })
    y = NestedNamespace({
        "C": {"D": 4, "E": {"test": 1}}
    })
    result = NestedNamespace({
        "A": 1,
        "B": 2,
        "C": {"D": 4, "E": {"test": 1}}
    })
    assert NestedNamespace.leaf_union(x, y) == result
