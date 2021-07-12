# TODOs

1. Code refactoring\
    *1.1.* Better commenting\
    *1.2* Find better names for methods\
    *1.3* Add docstrings (see sklearn code for an example). For building docstrings                 use some package that does it for you. For instance if you use ```sublime```, the you can use https://packagecontrol.io/packages/AutoDocstring
2. Handling of the sensitive attribute.\
    *2.1* Allow the user to specify the column with the sensitive attributes\
    *2.2.* Eventually let the user pass the column of sensitive attributes separately.
3. Add alpha parameter to ```Wasserstein.py``` allowing for relaxed fairness constraints. See ```Chzhen, Schreuder 2020``` paper.
4. Extend the code base to classification. (Think of having two modules -- classification and regression as in sklearn)
5. Documentation: for every algorithms add a basic use-case example.