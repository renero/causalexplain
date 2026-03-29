Installation
============

You can install CausalExplain using pip:

.. code-block:: bash

   pip install causalexplain

Requirements
-------------

CausalExplain requires Python 3.10+. The main dependencies are:

* numpy
* pandas
* networkx
* scikit-learn
* torch
* matplotlib

For a complete list of dependencies, see the ``requirements.txt`` file in the repository.

PyGraphViz installation issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you encounter issues installing PyGraphViz, you can install the package using the
following command:

.. code-block:: bash

   pip install pygraphviz --config-settings="--include-path=/usr/local/include/graphviz" --config-settings="--library-path=/usr/local/lib/graphviz/"

This command assumes that the GraphViz library is installed in the default location
(``/usr/local``). To install the library, I used homebrew in MacOS:

.. code-block:: bash

   brew install graphviz

These options may not work in all environments, so you can also add these configuration
to your ``~/.bashrc`` or ``~/.zshrc`` file:

.. code-block:: bash

   export CPLUS_INCLUDE_PATH=/usr/local/include/graphviz
   export LIBRARY_PATH=/usr/local/lib/graphviz

where the paths are the locations of the GraphViz library in your system. After adding
these lines, you can install PyGraphViz using the following command:

.. code-block:: bash

   pip install pygraphviz
