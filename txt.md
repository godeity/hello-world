Q-Learning for Web Crawling
===========================

Q-learning estimator with linear function approximation,
experience replay and double learning.

State-action value :math:`Q(s, a)` function is used. This function
predicts "return" :math:`R` - a discounted sum of all future rewards after
following action :math:`a` from state :math:`s`.

.. math::

    R_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \gamma^3 r_{t+4} + ... = \\
          r_{t+1} + \gamma R_{t+1}

    0 \leq \gamma < 1


Q function parameters are learned from "training examples":

* :math:`a_t` action taken (i.e. a feature vector for a link followed);
* :math:`r_{t+1}` observed reward (scalar value, e.g. whether a form is
  found or is a page on-topic);
* a set of actions :math:`A_{t+1}` (i.e. a feature matrix of links) available
  at this page; next action :math:`a_{t+1} \in A_{t+1}` used for TD updates
  is chosen from this set. In Q-learning it is a link with the highest
  :math:`Q(a_{t+1})` score. We need to store all available actions in
  experience replay memory because Q function changes over time.
* :math:`s_t` state (feature vector for the page a link is extracted from);
* :math:`s_{t+1}` state (feature vector for the current page, i.e. a page
  the link leads to);

With this data we can train a regression model for :math:`Q(s,a)` function
using any machine learning method. The trick is that instead of true
return values (which are unknown) in Q learning (and TD methods in general)
current estimates are used to train the model. Recall that Q function
is an approximation of R.

.. math::

    R_{predicted} = Q(s_t, a_t)

    R_{observed} = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}),

The regression model is trained on samples from experience replay memory;
currently it is not trained online. Experience replay provides several
benefits:

* data is used more efficiently;
* training is more stable - pure online training introduces strong biases
  because examples don't come in random order;
* a less obvious benefit is that even though we use a first-order
  approximation :math:`R_{observed} = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1})`,
  credit can be assigned to states and actions from several steps back if
  we repeatedly sample from the replay memory. So initially it works
  like :math:`TD(0)`, but over time, as we keep sampling, it moves towards
  :math:`TD(1)`. The effect is similar to :math:`TD(\lambda)`.

To stabilize training instead of a single :math:`Q(s, a)` function
two functions are used:

* target :math:`Q(s, a)` - this function is used for predictions, to define
  which action to follow; it doesn't change for a specified number of steps;
* online :math:`Q(s, a)` - this function is being trained using samples from
  experience replay memory; each N steps parameters of online Q function
  are copied to the target Q function.

For efficiency reasons instead of two (s, a) vectors a single vector is used,
with all features joined. It requires ~2x RAM because multiple
`s` copies are stored in memory, but the scipy-based implementation becomes
10x faster.
