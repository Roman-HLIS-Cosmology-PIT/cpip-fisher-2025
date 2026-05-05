# Welcome to our organization's code demo repository

This code repository (or "repo") is designed to demonstrate one way to organise your code and provides some suggested auxiliary files like .gitignore for example. It assumes your repository is mainly Python code.

# Get code

`git clone git@github.com:Roman-HLIS-Cosmology-PIT/template-code-repo.git`

# Install code

`pip install .`

# Run

`python templatecode`

Or open up the Jupyter Notebook called `demos/template-demo.ipynb` and have a look around.

# Contribute

* Ask Dida to add you to the repo,
* Clone as above,
* Make sure you `pip install -e .`,
* Make changes,
* `git fetch`,
* `git pull --rebase`,
* `git add [files you changed]`,
* `git commit [nice description of what you did]`,
* `git fetch`,
* `git status`, and if the status is clean:
* `git push`.

# Data

Make sure to place all data in folders hidden to git. Make two new folders in the root directory and this template is already set up so that git will ignore them:
`inputs/`
`outputs/`

Thanks for playing!
