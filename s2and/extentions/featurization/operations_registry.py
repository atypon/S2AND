from s2and.extentions.featurization.base_operation import BaseOperation


class Registry():
    """
    Class that implements registry for storing different operations
    """

    operations = {}

    @classmethod
    def register_operation(cls, operation_name: str):
        """
        Used for registering operations in registry
        :param operations_name: Name used as alias for the given implemented operation
        """
        def func(operation):
            cls.operations[operation_name] = operation
            return operation
        return func

    @classmethod
    def get_operation(cls, operation_name: str) -> BaseOperation:
        """
        Used for returning registered operation
        :param operation_name: Name of the registered operation to be retrieved
        :return: The selected operation object
        """
        return cls.operations[operation_name]()
