Quickstart
==========

This guide will help you get started with CausalExplain.

Basic Usage
----------

Here's a simple example of how to get CausalExplain help from the command line:

.. code-block:: bash

   python -m causalexplain --help

In order to run CausalExplain from the command line, you need to have Python 3.10
or later installed on your system. To install CausalExplain, run the following
command:

.. code-block:: bash

   pip install causalexplain

Once CausalExplain is installed, you can run it from the command line by typing
``python -m causalexplain``.

To run a simple case with a ``toy_dataset.csv`` file using ReX model, you can
use the following command, assuming default parameters:

.. code-block:: bash

   python -m causalexplain -d /path/to/toy_dataset.csv

That will generate the ReX model and run the model on the dataset, and print
the results to the terminal, like this:

.. code-block:: bash

   Resulting Graph:
   ---------------
   X1 -> X2
     X2 -> X4
     X2 -> X3
   X1 -> X4

which is the true graph expected.

GUI Mode
--------

If you prefer a browser-based interface, launch the local GUI with:

.. code-block:: bash

   python -m causalexplain --gui

This starts a local app that lets you train models, load/evaluate saved runs,
and generate synthetic datasets.


Input Arguments Information
--------------------------

The basic arguments are:

* ``-d`` or ``--dataset``: The path to the dataset file in CSV format.
* ``-t`` or ``--true_dag``: The path to the true DAG file in DOT format.
* ``-m`` or ``--method``: The method to use to infer the causal graph.
* ``-p`` or ``--prior``: JSON file with prior knowledge for ReX (optional).

These options allow you to specify the dataset, true DAG, and method to be used.
In case you don't have a true DAG, the result is the plausible causal graph,
which is the causal graph that is inferred by the method without taking into
account the true DAG.

Regarding the output of the ``causalexplain`` command, the following information is
provided:

- The plausible causal graph, which is the causal graph that is inferred by
the method without taking into account the true DAG.
- The metrics obtained from the evaluation of the causal graph against the true
DAG.

In those cases where training or running a method takes a long time, ``causalexplain``
allows you to save the model (``-s`` or ```--save_model```) trained in a file and
load it later. To load the model, use the ``-l`` or ``--load_model`` option.

ReX can also use prior knowledge to constrain edge directions. The prior is a
JSON file with a single ``prior`` key whose value is a list of tiers; each tier
is a list of column names. Variables in earlier tiers may cause variables in
later tiers, but not vice versa. All names must match the dataset columns.

.. code-block:: json

   {
     "prior": [
       ["A", "B"],
       ["C", "D"]
     ]
   }

Use it from the CLI like this:

.. code-block:: bash

   python -m causalexplain -d /path/to/data.csv -p /path/to/prior.json

Adaptive SHAP sampling
----------------------

For direct SHAP usage in notebooks, the explainability module exposes a
high-level wrapper that defaults to adaptive sampling.

Notebook example

.. code-block:: python

   from causalexplain.explainability.shapley import compute_shap

   # default: adaptive_shap_sampling=True
   res, diag = compute_shap(X, model, backend="kernel", adaptive_shap_sampling=True)

   # disable (may be slow for large m)
   res, diag = compute_shap(X, model, backend="kernel", adaptive_shap_sampling=False)

CLI example

.. code-block:: bash

   python -m causalexplain --shap-sampling
   python -m causalexplain --no-shap-sampling

Available SHAP backends are ``kernel``, ``gradient``, ``explainer``, and
``tree``. ReX defaults to ``tree`` when running the GBT regressor.

When adaptive sampling is enabled, the key knobs are ``max_shap_samples``,
``K_max``, ``max_explain_samples``, and ``stratify``.

If ``adaptive_shap_sampling=False`` and ``m > 2000``, the tool emits a warning
about potential non-termination (the threshold is conservative).

Why adaptive sampling is mathematically reasonable
--------------------------------------------------

Many SHAP explainers approximate an expectation over a background
distribution; using ``n`` background points gives a Monte Carlo estimate.
The standard error scales approximately as :math:`SE \propto 1/\sqrt{n}`.

When sampling without replacement from a finite dataset of size ``m``, the
finite population correction factor applies as :math:`\sqrt{1 - n/m}`.
This means increasing ``n`` yields diminishing returns, so capping the
background around 200-250 is a pragmatic speed/accuracy tradeoff.

Repeating the sampling (``K`` runs) provides a stability diagnostic: compute
a global importance vector per run as ``mean(|SHAP|)`` per feature, then check
variability (CV) and rank stability (Spearman correlation) across runs.

Backend-aware note: Kernel SHAP is particularly sensitive and expensive, so
caps like ``max_explain_samples`` matter most there. Gradient and generic
explainers often have different performance profiles, but still benefit from
controlled baselines/background sizes.

The option ``-b`` or ``--bootstrap`` allows you to specify the number of iterations
for bootstrap in the ReX method.
The default value is 20, but you can change it to a different
value, to test the effect of the number of iterations on the performance of the
method. This option is linked to the next one, ``-T``.

The option ``-T`` or ``--threshold`` allows you to specify a threshold for the
bootstrapped adjacency matrix computed for the ReX method. The default value is
0.3, but you can change it to a different value, to test the effect of the
threshold on the performance of the method. Lower values in the adjacency matrix
represent edges that appear less frequently in the bootstrap samples, while higher
values represent edges that appear more frequently. So, a higher threshold
represents a more conservative approach to the inference of the causal graph.

The option ``-r`` or ``--regressor`` allows you to specify a list of comma-separated
names of the regressors to be used. The default value is ``dnn,gbt``, but you can
change it to a different list of regressors. Current implementation only supports
DNN and GBT regressors, but they can be extended in the future.

The option ``-u`` or ``--union`` allows you to specify a list of comma-separated
names of the DAGs to be unioned. This option is only valid for the ReX method,
and it is used to combine the causal graphs inferred by the method with different
hyperparameters. By default, ReX combines the DAGs inferred with the DNN and
GBT regressors, but you can extend ReX with more regressors and combine them
with different hyperparameters.

The option ``-i`` or ``--iterations`` allows you to specify the number of iterations
that the hyper-parameter optimization will perform in the ReX method. The default
value is 100, but you can change it to a different value, to test the effect of
the number of iterations on the performance of the method.

The option ``-S`` or ``--seed`` allows you to specify a seed for the random number
generator. The default value is 1234, but you can change it to a different value,
to test the effect of the seed on the performance of the method.

The option ``-o`` or ``--output`` allows you to specify the path to the output file
where the resulting DAG will be saved in DOT format. The default value is
``./output.dot``, but you can change it to a different value, to save the DAG in a
different file.
