from typing import Union
import numpy as np

class RangeDictionary:
    def __init__(self, array:Union[np.ndarray, list]=[]):
        self.dataset = {}
        tmp = np.nan
        key = 0
        for value in array:
            if value != tmp:
                self.dataset[key] = value
                key += 1
                tmp = value
            else:
                tmp_key = list(self.dataset.keys())[-1]
                self.dataset.pop(tmp_key)
                if isinstance(tmp_key, int):
                    self.dataset[(tmp_key,key)] = tmp
                    key += 1
                elif isinstance(tmp_key,tuple):
                    self.dataset[(tmp_key[0],key)] = tmp
                    key += 1

        self.len = key

    def __getitem__(self, key:int) -> Union[int,float,list]:
        try:
            return self.dataset[key]
        except KeyError:
            for keys in list(self.dataset.keys()):
                if isinstance(keys, tuple):
                    if keys[0] <= key <= keys[1]:
                        return self.dataset[keys]
        raise KeyError

    def __len__(self) -> int:
        return self.len

    def pop(self, index:Union[int,tuple] = None) -> Union[int,float,list]:
        if index is None:
            return self.dataset.popitem()
        elif isinstance(index, int):
            try:
                return self.dataset.pop(index)
            except KeyError:
                for keys in list(self.dataset.keys()):
                    if isinstance(keys, tuple):
                        if keys[0] <= index <= keys[1]:
                            return self.dataset.pop(keys)
        elif isinstance(index, tuple):
            return self.dataset.pop(index)
        
        raise KeyError

    def __repr__(self) -> str:
        to_ret = ''
        for key, value in self.dataset.items():
            if isinstance(key, int):
                to_ret += f"{key} : {value}"
            elif isinstance(key, tuple):
                to_ret += f"{key[0]} - {key[1]} : {value}"
            to_ret += '\n'
        return to_ret

    def __str__(self) -> str:
        return self.__repr__()
    
    def clear(self) -> None:
        self.dataset.clear()
        self.len = 0
        