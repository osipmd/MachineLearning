class smart_dict(dict):
    def __missing__(self, key):
        return 0