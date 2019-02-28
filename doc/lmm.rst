*******************
Linear Mixed Models
*******************

Linear mixed models (LMMs) are a generalisation of linear models [#f1]_ to allow the
ouctome to be described as a summation of both fixed and random effects [#f2]_.
LMM inference is implemented by the :mod:`glimix_core.lmm` module and described here.

.. note::

   Please refer to the Variables_ section for the definition of the
   otherwise-unspecified mathematical symbols.

.. include:: lmm/intro.inc

.. include:: lmm/scan.inc

.. include:: lmm/api.inc

.. include:: lmm/impl.inc

.. _Variables:

.. glossary::

    |n|
        Number of samples.

    |m|
        Number of traits.

    |c|
        Number of candidates.

    |d|
        Number of covariates.

    |k|
        Number of random effects.

    |r|
        Covariance-matrix rank.

.. |n| replace:: :math:`n`
.. |m| replace:: :math:`m`
.. |c| replace:: :math:`c`
.. |d| replace:: :math:`d`
.. |k| replace:: :math:`k`
.. |r| replace:: :math:`r`

.. rubric:: References

.. [#f1] Wikipedia contributors. (2018, May 22). Linear model. In Wikipedia, The Free
         Encyclopedia. Retrieved 16:00, August 5, 2018, from
         https://en.wikipedia.org/w/index.php?title=Linear_model&oldid=842479751.

.. [#f2] Introduction to linear mixed models. UCLA: Institute for Digital Research and
         Education. Retrieved from August 5, 2018, from
         https://stats.idre.ucla.edu/other/mult-pkg/introduction-to-linear-mixed-models/.

.. [#f3] Lippert, Christoph, Listgarten, Jennifer, Liu, Ying, Kadie, Carl M,
         Davidson, Robert I & Heckerman, David (2011). FaST linear mixed
         models for genome-wide association studies. Nature methods, 8,
         833-835.