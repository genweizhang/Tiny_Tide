from loguru import logger


class Component(object):
    """
    One of the individial, irreducible parts of a flow chemistry setup.

    All components in an `Apparatus` must be of type `Component`. However, it is unlikely that a user will directly
    instantiate a `Component`.

    Attributes:
    - name (str, optional): The name of the component.
    """

    _id_counter = 0
    _used_names = set()

    def __init__(self, name=None):
        # name the object, either sequentially or with a given name
        if name is None:
            self.name = self.__class__.__name__ + "_" + str(self.__class__._id_counter)
            self.__class__._id_counter += 1
        else:
            self.name = str(name)
        self._visualization_shape = "box"

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name}>"

    def __str__(self):
        return f"{self.__class__.__name__} {self.name}"

    def __enter__(self):
        logger.trace(f"Entering context for {self}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.trace(f"Exiting context for {self}")
        pass

    def validate(self, dry_run):
        """Components are valid for dry runs, but not for real runs."""
        if dry_run:
            return True
        else:
            return False