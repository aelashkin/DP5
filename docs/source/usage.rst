Usage
=====

.. _installation:

Installation
------------

To use DP5, clone this repository:

.. code-block:: console

    git clone https://github.com/ruslankotl/DP5.git


Then change to the top-level directory of the repository and install the repository using

.. code-block:: console

    pip install -e .

To also install documentation build dependencies (``sphinx`` and ``sphinx-rtd-theme``), use the ``dev`` extra:

.. code-block:: console

    pip install -e ".[dev]"

.. note::
    The repository works best on x86 Linux machines. 
    For Apple Silicon, additional library `Tensorflow-metal <https://developer.apple.com/metal/tensorflow-plugin/>`_ will be installed automatically.

.. _configuration:

Configuration
-------------

This version adds :doc:`human-editable configuration files <config_anatomy>`. They can be supplied in `.json` and `.toml` formats.
The basic elements of former command-line interface are retained for experienced user convenience.

Command line arguments
^^^^^^^^^^^^^^^^^^^^^^

.. attention::
    Arguments from command line override arguments from configuration file!

================================== ==========================
 Command Line Flags                  What they mean 
================================== ==========================
``-s``, ``--structure_files``          Paths to structure files.
   ``-n``, ``--nmr_file``            Paths to NMR spectra or their description.
   ``-c``, ``--config``              Path to configuration file for the run. Defaults to ``dp5/config/default_config.toml``.
   ``-o``, ``--output``             Path to output directory. Defaults to current working folder.
   ``-i``, ``--input_type``         Input file type. Can be ``sdf``, ``smiles``, ``smarts``, or ``inchi``. Default is ``sdf``.
   ``-w``, ``--workflow <flags>``    Workflow type. Must be followed by :ref:`workflow flags <workflowflags>` without spaces. 
   ``--stereocentres``               When generating diastereomers, limit generation to specified stereocentres.
   ``--nmr-assignment-mode``         Manual NMR assignment mode: ``strict`` by default, or ``legacy`` for the previous loose matching behavior.
   ``--allow-extra-peaks``           In strict manual mode, report extra experimental carbon shifts instead of aborting.
   ``--allow-missing-peaks``         In strict manual mode, report missing required carbon groups instead of aborting.
   ``-l``, ``--log_filename``       Log file name.
   ``--log_level``                   Logging levels. Can be ``warning``, ``info``, or ``debug``. Default level is ``info``.
================================== ==========================


Workflow arguments
^^^^^^^^^^^^^^^^^^

Specifies workflow actions. Will load the values from the :ref:`configuration file <cfg_workflowflags>` if left unset.

.. _workflowflags:


=============== ======================== ===============
Workflow Flags   Config file equivalents What they mean
=============== ======================== ===============
``c``           ``cleanup``              generate 3D structure, optimise using MMFF
``g``           ``generate``             generate diastereomers
``m``           ``conf_search``          perform conformational search
``o``           ``dft_opt``              optimise geometries using DFT
``e``           ``dft_energies``         calculate single point energies using DFT
``n``           ``dft_nmr``              calculate NMR spectra using DFT-GIAO method
``a``           ``assign_only``          assignment only **(currently not supported)**
``s``           ``dp4``                  perform DP4 analysis
``w``           ``dp5``                  perform DP5 analysis
=============== ======================== ===============

Default DP4 workflow for establishing stereochemistry would be specified as ``-w gnms``, 
the best results would be produced using ``-w gnomes``.

In general, conformational search should provide representative geometries, 
DFT optimisation would provide accurate geometries, 
and single-point energies would increase precision 
by re-weighting the conformers.

Manual NMR descriptions
^^^^^^^^^^^^^^^^^^^^^^^

Manual description files are strict and equivalence-aware by default. The first
non-empty section contains comma-separated carbon shifts, the second contains
comma-separated proton shifts, and later sections may contain one equivalence
group per line plus optional ``OMIT`` lines:

.. code-block:: text

    140.1(any),129.7(any),126.7(C10),20.3(C14,C23)

    0.0(any)

    C1,C3
    C4,C5
    C14,C23
    OMIT C19,H51

Labels are element symbols plus 1-based atom indices from the supplied
molecule/SDF atom order. ``(any)`` leaves a shift unassigned. In strict mode,
equivalent carbons collapse into one required carbon signal and every
non-omitted carbon group must receive exactly one experimental signal. Use
``--nmr-assignment-mode legacy`` to restore the previous loose atom-level
matching behavior.
