from typing import Union
import numpy as np
import math

class RangeDictionary:
    def __init__(self, array:Union[np.ndarray, list]=[]):
        self.dataset = {}
        tmp = np.nan
        key = 0
        for value in array:
            if value != tmp and not math.isnan(value):
                self.dataset[key] = value
                tmp = value
            else:
                tmp_key = list(self.dataset.keys())[-1]
                self.dataset.pop(tmp_key)
                if isinstance(tmp_key, int):
                    self.dataset[(tmp_key,key)] = tmp
                elif isinstance(tmp_key,tuple):
                    self.dataset[(tmp_key[0],key)] = tmp
                
            key += 1

        self.len = len(self.dataset.values())

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

    def __call__(self, array:Union[np.ndarray, list]) -> None:
        return self.__init__(array)

    def __iter__(self) -> dict:
        keys_list = list()
        values_list = list()
        for keys,values in self.dataset.items():
            if isinstance(keys, int):
                keys_list.append(keys)
                values_list.append(values)
            elif isinstance(keys, tuple):
                keys_list.append(keys[0])
                values_list.append(values)

        return zip(keys_list, values_list)

    
    def clear(self) -> None:
        self.dataset.clear()
        self.len = 0    

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
    
    def keys(self):
        keys_list = list()
        for keys in self.dataset.keys():
            if isinstance(keys, int):
                keys_list.append(keys)
            elif isinstance(keys, tuple):
                keys_list.append(keys[0])

        return keys_list

    def values(self):
        return self.dataset.values()

    def items(self):
        return zip(self.keys(), self.values())

if __name__ == '__main__':
    tmp = RangeDictionary([0.1,0.2,0.2,0.25,0.3,0.4,0.4])
    print(tmp)
