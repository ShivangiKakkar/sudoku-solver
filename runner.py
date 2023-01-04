import pandas as pd

from dfs_solver import *
from tensorflow.keras.models import load_model


if __name__ == '__main__':
    test()

    '''For solving problems from the textfiles'''
    # solve_all(from_file("data/easy50.txt", '========'), "easy", None)
    # solve_all(from_file("data/hardest.txt"), "hardest", None)
    solve_all(from_image("images\p6.jpg",
                         model=load_model('imdetect/textmod.h5')), "Image", showif=0)
    # solve_all([random_puzzle() for _ in range(99)], "random", 100.0)

    '''For solving puzzles from the million problem dataset'''
    # df = pd.read_csv('data/raw.csv')
    # puzz = df.iloc[0:1000]['quizzes'].values
    # solve_all(puzz, "single")
