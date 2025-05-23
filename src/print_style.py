class Style():
    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    ORANGE = "\33[93m"
    RESET = "\033[0m"

    @classmethod
    def red(cls, text):
        return f'{cls.RED}{text}{cls.RESET}'

    @classmethod
    def green(cls, text):
        return f'{cls.GREEN}{text}{cls.RESET}'

    @classmethod
    def blue(cls, text):
        return f'{cls.BLUE}{text}{cls.RESET}'

    @classmethod
    def orange(cls, text):
        return f'{cls.ORANGE}{text}{cls.RESET}'
