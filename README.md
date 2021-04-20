Intermediate Machine learning with scikit-learn
========================================================

Part 2 of 4
-----------
Other parts:
- [Part 1](https://github.com/amueller/ml-workshop-1-of-4)
- [Part 3](https://github.com/amueller/ml-workshop-3-of-4)
- [Part 4](https://github.com/amueller/ml-workshop-4-of-4)


Content
-------
- [Reminder on supervised learning](https://amueller.github.io/ml-workshop-2-of-4/slides/01-reminder-supervised-learning.html)
- [Grid search and cross-validation](https://amueller.github.io/ml-workshop-2-of-4/slides/02-cross-validation-grid-search.html)
- [Linear models for regression](https://amueller.github.io/ml-workshop-2-of-4/slides/03-linear-models-regression.html)
- [Linear models for classification](https://amueller.github.io/ml-workshop-2-of-4/slides/04-linear-models-classification.html)
- [Decision trees](https://amueller.github.io/ml-workshop-2-of-4/slides/05-trees-forests.html)
- [Random Forests](https://amueller.github.io/ml-workshop-2-of-4/slides/05-trees-forests.html#26)


Instructor
-----------

- [Andreas Mueller](http://amuller.github.io) [@amuellerml](https://twitter.com/amuellerml) - Columbia University; [Book: Introduction to Machine Learning with Python](http://shop.oreilly.com/product/0636920030515.do)

---

This repository will contain the teaching material and other info associated
with the "Intermediate Machine Learning with scikit-learn" course.

About the workshop
------------------
Scikit-learn is a machine learning library in Python, that has become a
valuable tool for many data science practitioners. This workshop will go beyond
the basics and show how to effectively evaluate and tune algorithms. We will
also discuss the most important machine learning algorithms that you're likely
to see in practice, how and when to use them, and some details about how they
work internally. The session will focus on linear models for classification and
regression and tree-based models, including random forests.

Prerequisites
-------------
This workshop assumes familiarity with Jupyter notebooks and basics of pandas, matplotlib and numpy.
It also assumes familiarity with the basics of supervised learning, like training and test data and basics of model evaluation.
You should have build a model with scikit-learn (or attend Introduction to Machine learning with scikit-learn) before
taking this workshop.

Obtaining the Tutorial Material
--------------------------------


If you are familiar with git, it is most convenient if you clone the GitHub repository. This
is highly encouraged as it allows you to easily synchronize any changes to the material.

```
git clone https://github.com/amueller/ml-workshop-2-of-4.git
```

If you are not familiar with git, you can download the repository as a .zip file by heading over to the GitHub repository (https://github.com/amueller/ml-workshop-2-of-4) in your browser and click the green “Download” button in the upper right.

![](images/download-repo.png)

Please note that I may add and improve the material until shortly before the tutorial session, and we recommend you to update your copy of the materials one day before the tutorials. If you have an GitHub account and forked/cloned the repository via GitHub, you can sync your existing fork with via the following commands:

```
git pull origin master
```


Installation Notes
------------------

This tutorial will require recent installations of

- [NumPy](http://www.numpy.org)
- [SciPy](http://www.scipy.org)
- [matplotlib](http://matplotlib.org)
- [pillow](https://python-pillow.org)
- [pandas](http://pandas.pydata.org)
- [scikit-learn](http://scikit-learn.org/stable/) (>=0.22.1)
- [IPython](http://ipython.readthedocs.org/en/stable/)
- [Jupyter Notebook](http://jupyter.org)

The last one is important, you should be able to type:

    jupyter notebook

in your terminal window and see the notebook panel load in your web browser.
Try opening and running a notebook from the material to see check that it works.

For users who do not yet have these  packages installed, a relatively
painless way to install all the requirements is to use a Python distribution
such as [Anaconda](https://www.continuum.io/downloads), which includes
the most relevant Python packages for science, math, engineering, and
data analysis; Anaconda can be downloaded and installed for free
including commercial use and redistribution.
The code examples in this tutorial requires Python 3.5 or later.

After obtaining the material, we **strongly recommend** you to open and execute
a Jupyter Notebook `jupter notebook check_env.ipynb` that is located at the
top level of this repository. Inside the repository, you can open the notebook
by executing

```bash
jupyter notebook check_env.ipynb
```

inside this repository. Inside the Notebook, you can run the code cell by
clicking on the "Run Cells" button as illustrated in the figure below:

![](images/check_env-1.png)


Finally, if your environment satisfies the requirements for the tutorials, the executed code cell will produce an output message as shown below:

![](images/check_env-2.png)
