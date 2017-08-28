.. highlight:: shell

==================================
Installation
==================================

Install on Linux and OSX
------------------------

Developers
~~~~~~~~~~~~~~~~~~~~~~

First, make sure `you have Pytorch installed <http://pytorch.org/>`_. 

Then, clone this repository with: 

.. code:: python

  $ git clone https://github.com/nasimrahaman/inferno.git


Next, install the dependencies.

.. code:: python

  $ cd inferno
  $ pip install -r requirements.txt


If you use python from the shell: 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, add *inferno* to your `PYTHONPATH` with:

.. code:: python

  source add2path.sh

If you use PyCharm:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Refer to this `QA <https://askubuntu.com/questions/684550/importing-a-python-module-works-from-command-line-but-not-from-pycharm>`_ about setting up paths with Pycharm.

Users
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installation with `pip` will be ready when the release is.











======================================================
Installation via PyPi / pip / setup.py(Experimental)
======================================================

You need to install pytorch via pip before installing
inferno.  Follow the `pytorch installation guide`_.

Stable release
--------------

To install inferno, run this command in your terminal:

.. code-block:: console

    $ pip install pytorch-inferno

This is the preferred method to install inferno, as it will always install the most recent stable release. 

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _pytorch installation guide: http://pytorch.org/

From sources
------------------------

The sources for inferno can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/nasimrahaman/inferno

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/nasimrahaman/inferno/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/nasimrahaman/inferno
.. _tarball: https://github.com/nasimrahaman/inferno/tarball/master
