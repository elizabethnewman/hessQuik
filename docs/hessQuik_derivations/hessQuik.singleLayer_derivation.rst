Single Layer Derivation
=======================

.. math::
   :nowrap:

   \begin{align}
        [\nabla^2 f(\mathbf{x})]_{j, :, :}  & = \nabla^2 f_j(\mathbf{x})\\
                                            & = \nabla^\top [\nabla f_j(\mathbf{x})]\\
                                            & = \nabla^\top [\nabla u(\mathbf{x})\mathbf{K}^\top\text{diag}(\sigma'(\mathbf{K} u(\mathbf{x}) + \mathbf{b})) \mathbf{e}_j]\\
                                            & = \nabla^\top [\nabla u(\mathbf{x})\mathbf{K}^\top \sigma'(\mathbf{K}_{j,:}u(\mathbf{x}) + \mathbf{b}_j)\mathbf{e}_j]\\
                                            & = \nabla^\top [\nabla u(\mathbf{x})\mathbf{K}^\top \mathbf{e}_j \sigma'(\mathbf{K}_{j,:}u(\mathbf{x}) + \mathbf{b}_j)]\\
                                            & = \nabla^\top [\nabla u(\mathbf{x})\mathbf{K}_{j,:}^\top \sigma'(\mathbf{K}_{j,:}u(\mathbf{x}) + \mathbf{b}_j)]\\
                                            & = \nabla u(\mathbf{x})\mathbf{K}_{j,:}^\top\text{diag}(\sigma''(\mathbf{K} u(\mathbf{x}) + \mathbf{b}))\mathbf{K}_{j,:}\nabla u(\mathbf{x})^\top\\
                                            & \qquad + \sum_{i=1}^\ell [\nabla^2 u(\mathbf{x})]_{i,:,:} (\mathbf{K}_{j,i}\sigma'(\mathbf{K}_{j,:}u(\mathbf{x}) + \mathbf{b}_j)).
   \end{align}
