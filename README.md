# MPM Space Sciences - Machine Learning

- [MPM Space Sciences - Machine Learning](#mpm-space-sciences---machine-learning)
  - [Preliminaries](#preliminaries)
    - [Python](#python)
      - [Linux - Ubuntu & Mint](#linux---ubuntu--mint)
      - [Mac OS](#mac-os)
      - [Windows](#windows)
    - [Virtualenv](#virtualenv)
    - [Virtualenv-wrapper (Optional)](#virtualenv-wrapper-optional)
      - [Ubuntu, Mint, and Mac OS](#ubuntu-mint-and-mac-os)
      - [Windows](#windows-1)
    - [Editor](#editor)
  - [Setup](#setup)

## Preliminaries

Install and test Python, Virtualenv and Virtualenv-wrappers on your machine. We suggest to work with Python 3.8 or higher versions.

### Python

Here, we report minimal instructions. See [this guide](https://realpython.com/installing-python/) for details.

#### Linux - Ubuntu & Mint

Open your terminal and first check which Python version is installed on your machine:

```shell
python --version

python3 --version
```

If the version does not match the requirements, run the following command:

```shell
sudo apt-get update
sudo apt-get install python3.8 python3-pip
```
or 
```shell
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.8 python3-pip
```

After the installation, you should be able to open a Python shell:

```
 ~/veos_digital/repos/mpm_space_sciences/python3                                                                                                   
Python 3.9.1 (default, Dec 10 2020, 11:11:14) 
[Clang 12.0.0 (clang-1200.0.32.27)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

Exit by pressing Ctrl + D

#### Mac OS

Open your terminal and first check which Python version is installed on your machine:

```shell
python --version

python3 --version
```

If the version does not match the requirements, first install [Homebrew by following these instructions](https://brew.sh/). Then, update homebrew by running:

```shell
brew update && brew upgrade
```

Finally, install Python:

```shell
brew install python3
```

After the installation, you should be able to open a Python shell:

```
 ~/veos_digital/repos/mpm_space_sciences/python3                                                                                                   
Python 3.9.1 (default, Dec 10 2020, 11:11:14) 
[Clang 12.0.0 (clang-1200.0.32.27)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

#### Windows

Open PwerShell, then check which Python version is installed on your machine:

```shell
python --version
```

If the version does not match the requirements, open the Microsoft Store app, search for Python and choose the highest version available.

### Virtualenv

You can install Virtualenv through pip:

```shell
sudo pip3 install virtualenv 
```

You can now create a virtualenv running:

```shell
virtualenv venv_name 
```
or specifying a Python interpreter, e.g.,
```
virtualenv -p /usr/bin/python3 venv_name
```
You can now activate the virtual environment by running:

```shell
source venv_name/bin/activate
```
run `deactivate` to exit the virtual environment.


### Virtualenv-wrapper (Optional)

#### Ubuntu, Mint, and Mac OS

Run

```shell
sudo pip install virtualenvwrapper
```

Then, on Mac OS, Ubuntu, and Mint add to your shell startup file (.bashrc, .profile) the following lines:

```shell
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/Devel
source /usr/local/bin/virtualenvwrapper.sh
```
Save, exit, and reload the startup file, e.g.,
```shell
source ~/.bashrc
```

Now try creating a virtual environment by running:

```shell
mkvirtualenv --python=python3 mpm_space_sciences                                   
```


#### Windows

Install [virtualenvwrapper-win](https://pypi.org/project/virtualenvwrapper-win/) and follow the instruction on the website.

Thereafter, try creating a virtual environment by running:

```shell
mkvirtualenv --python=python3 mpm_space_sciences                                   
```

### Editor

You can use your preferred editor. If you do not have one we suggest [Visual Studio](https://code.visualstudio.com/).

## Setup

After successfully installing Python and Virtualenv, 

1. let us create a virtual environment called **mpm_space_sciences**. Follow the instructions above, if you do not know how to do it.

2. Clone this repository:
    ```shell
    git clone ...
    ```
    
3. Install it:
    ```shell
    cd mpm_space_sciences
    pip install -e ./
    ```
    Where `./` means that pip will use the `./setup.py` script to install the packagge, and the `-e` option allows you to modify the source code of the package you installed.