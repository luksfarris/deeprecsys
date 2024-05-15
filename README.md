# Deep RecSys

`deeprecsys` is an <u>open</u> <u>tool belt</u> to speed up the development of <u>modern</u> <u>data science</u> projects at an <u>enterprise</u> level.

These words were chosen very carefully, and by them we mean:
- **Open**: we rely on OSS and distribute openly with [a GNU GPLv3 license](./LICENSE) that won't change in the future. The official distribution channels are pypi ([see deeprecsys at pypi](https://pypi.org/project/deeprecsys/)) and GitHub ([see deeprecsys at Github](https://github.com/luksfarris/deeprecsys)).
- **Tool belt**: this project contains code that may extract, process, analyse, aggregate, test, and present data.
- **Modern**: the code will be updated as much as possible to the newest versions, as long as they are stable and don't break pre-existing functionality.
- **Data Science**: This project will contain a mixture of data engineering, machine learning engineering, data analysis, and data visualization.
- **Enterprise**: The code deployed here will likely have been battle-tested by large organizations with millions of customers. Unless stated, it is production-ready. All code including dependencies is audited and secure.

## Historical Note

If you're here from the research piece [Optimized Recommender Systems with Deep Reinforcement Learning](https://arxiv.org/abs/2110.03039), please checkout the old branch `origin/thesis` for reproducibility. The README should contain instructions to get you started.

## Installation and usage

Installation depends on your framework, so you may need to adapt this. Here's an example using pip:

```
pip install deeprecsys
```

## For Developers

### Source Control

All source control is done in `git`, via GitHub. Make sure you have a modern version of git installed. For instance, you can checkout the project using SSH with:

```
git clone git@github.com:luksfarris/deeprecsys.git
```

### Automation

All scripts are written using Taskfile. You can install it following [Task's instructions](https://taskfile.dev/installation/). The file with all the tasks is `Taskfile.yml`.