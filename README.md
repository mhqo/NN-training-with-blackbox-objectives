Some code to train an NN for a combination of a white-box function $L_{wb}(\cdot)$ (with gradient) and a black-box function $L_{bb}(\cdot)$ (no graident). The gradients for the black-box objective are estimated through the score function estimate.

Let $\mu$ be the networks parameters and $\theta \sim p(\theta; \mu) = \mathcal{N}(\mu, \sigma^2 \cdot I) $ with $\sigma$ fixed (hyperparameter).
Then, the training objective becomes

$$\min_\mu  L(\mu) + \lambda \int  L_{bb}(\theta) p(\theta; \mu) \; \rm d \theta $$

For details and an example see `NN training with blackbox lossfunction.ipynb`