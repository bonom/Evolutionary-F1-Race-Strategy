import pickle
import os
import numpy as np

class Car:
    def __init__(self,):
        self.name = "Car"

class Cars:
    def __init__(self,):
        self.cars = dict()
        for i in range(19):
            self.cars[i] = Car()


def get_cars(path):
    path = os.path.abspath(path)
    if os.name == 'posix' and path.split('/')[-2] != 'Data':
        path = path.split('/')
        while path[-2] != 'Data':
            path = path[:-1]
        new_path = ''
        for p in path:
            new_path += p + '/'
        path = new_path
    elif os.name == 'nt' and path.split('\\')[-2] != 'Data':
        path = path.split('\\')
        while path[-2] != 'Data':
            path = path[:-1]
        new_path = ''
        for p in path:
            new_path += p + '\\'
        path = new_path

    print(path)


