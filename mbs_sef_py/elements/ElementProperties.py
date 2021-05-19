import abc


class ElementProperties:
    def __init__(self):
        pass

    @staticmethod
    def get_element_type():
        return False


class ElementWithConstraintsProperties(ElementProperties, abc.ABC):
    def __init__(self):
        ElementProperties.__init__(self)
        self.constraint_scaling = 1.e0
