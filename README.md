# Introduction

## Dutch Books and Irrationality

Within the field of philosophy of probability, the irrationality of an
agent is commonly established through the use of a Dutch book argument.
Dutch books are simply collections of bets which, if all taken,
guarantee a loss. Assuming an agent bets in accordance with their
beliefs, the purchase of a Dutch book is indicative of inconsistent
beliefs. This inconsistency is attributed to irrationality on the
agent's part. A famous theorem states that if an agent's beliefs,
represented as probabilities, do not obey the Kolmogorov axioms of
probability theory, then there exists a Dutch book that they are willing
to purchase (again, if they bet according to their beliefs). These
considerations raise an interesting question about Large Language Models
(LLMs): Are LLMs rational agents in the sense that they are immune to
Dutch book exploitation?

## Diachronic vs. Synchronic Consistency

Being immune to Dutch book exploitation is equivalent to having beliefs
that are consistent with the axioms of probability theory, and much work
has been done in training LLMs to be consistent probabilistic reasoning
machines. However, these results fail to address the above question.
This is because other researchers are examining a fundamentally
different kind of probabilistic consistency, one which we call
*synchronic* consistency. By synchronic consistency, we mean consistency
within a single context. For example, an LLM is given propositions and
their probabilities, and then makes correct probabilistic inferences
from that information. However, all that is needed to be synchronically
consistent is memorization of schematics and linguistic structure. Even
very irrational agents can memorize Bayes' rule or Chebyshev's
inequality and correctly answer questions that require their
application. Such an agent is still susceptible to Dutch book
exploitation.

For our purposes, we will need a much stronger notion of probabilistic
consistency, one which we call *diachronic* consistency. By diachronic
consistency, we mean consistency across multiple contexts. For example,
an LLM is asked to give the probability of proposition $Q$ in one
context, the probability of $\neg Q$ in another, and both probabilities
add to one. What we find in practice is that LLMs are not diachronically
consistent. For example, we asked ChatGPT for the probability of a
Republican winning the 2028 presidential election, it returned 0.45. In
a new context, we asked for the probability that Republicans do not win
said election, and it returned 0.6. Whichever way it is synthesizing
information to produce these numbers, it is not doing so in a consistent
or robust manner. We find similar results with other models such as
Google Gemini, Qwen2.5-3B, Qwen3-4B, etc. The conclusion is that LLMs
are irrational agents. A new question now arises: Can LLMs be trained to
overcome this defect?

# Experimental Design

## Setup

Our experiments are designed under the assumption that LLMs have
beliefs. What we mean by 'beliefs' is that, given a proposition $Q$,
there is a probability $p$ such that the LLM assigns $p$ to $Q$ in a
robust and reliable manner (invariant to wording of the query, or other
semantically irrelevant variations). This may prove to be false.

In what follows, let $P_{\theta}(Q)$ be the probability that a model
parameterized by $\theta$ assigns to proposition $Q$.

## GRPO RLVR to Align Model with Kolmogorov Axioms

Since we are trying to refine the behavior of an LLM, we reasoned that
RLVR would be most appropriate training paradigm. Due to anticipated GPU
limitations, GRPO was selected as the reinforcement learning (RL)
algorithm.

Our first step will be to train a model that returns well-formatted
strings from which we can easily extract a numerical probability. Once
our model can return correctly formatted strings, will begin training it
to return probabilities that are consistent with the Kolmogorov axioms
*across* contexts. We will first look at the non-negativity axiom, which
states that all probabilities are non-negative. While training, we will
give a reward of +1.0 if 

$$0 \leq P_\theta \leq 1.$$ 

If our model can do
this much, we will move on to the normativity axiom. If our LLM is
consistent with normativity, then

$$P_\theta(Q) + P_\theta(\neg Q) = 1,$$ 

where the probabilities of $Q$
and $\neg Q$ have been elicited from separate contexts. We will grant a
reward of +1.0 if 

$$|P_\theta(Q) + P_\theta(\neg Q) - 1| \leq 0.01,$$

since we want to allow for a little deviation to prevent rewards from
being too sparse. Finally, we will train our model to be consistent with
the axiom of finite sub-additivity. This means that if $Q_1, \dots, Q_n$
are propositions, then

$$P_\theta\left(\bigvee_{i=1}^n Q_i\right) \leq \sum_{i=1}^n P_\theta(Q_i),$$

where each $P_\theta(Q_i)$ is evaluated in a different session. We will
grant a reward of +1.0 if

$$P_\theta\left(\bigvee_{i=1}^n Q_i\right) \leq \sum_{i=1}^n P_\theta(Q_i) + 0.005n,$$

again, allowing for minor deviations.

## Data to Source Our Queries

For the sake of simplicity, we will begin by deterministically
generating queries from European soccer data publicly available at
https://www.football-data.co.uk. Specifically, we will extract the date,
home team, and away team. Then, we generate the pair of propositions

$$
Q_i = \text{On }\{\text{DATE}_i\},\text{ }\{\text{HOME TEAM}_i\} \text{ wins} \text{ against }\{\text{AWAY TEAM}_i\}
$$

$$
\neg Q_i = \text{On }\{\text{DATE}_i\},\text{ }\{\text{HOME TEAM}_i\} \text{ does not win} \text{ against }\{\text{AWAY TEAM}_i.\}
$$ 

If we are successful, we will deploy more sophisticated
means of generating propositions, such as employing a separate LLM.

## Models and Hyperparameters

We trained Qwen2.5-3B and Qwen3-4B with a learning rate of $1e-6$. For
Qwen3-4B, we alternate between `enable`\_`reasoning=True` and
`enable`\_`reasoning=False`, since the finetuning of the reasoning model
may conflict with our desire to avoid schema-driven responses based on
the prompt.

## Reinforcement Learning Environment

For our first attempt at creating an reinforcement learning environment,
we tried to employ `RLib` from `Ray`, a common library used in industry
for LLM fine-tuning. `RLib` is notorious for having a steep learning,
curve and although we got our unique pipeline working, we ran into some
tricky errors with supercomputer user limits. We eventually settled on
modifying an existing GRPO implementation, but the time spent getting
familiar with `RLib` and other reinforcement learning libraries like
`TRL` were worthwhile and may prove useful in the future.

We extensively experimented with different reward functions. Early on,
we noticed the model was converging to non-informative distributions,
i.e. assign all possible outcomes equal probability. This trivially
satisfies the consistency requirements, but the probabilities are no
longer meaningful. We want the model to report probabilities based on
its background information accumulated from training data. To remedy
this, we added a checkpoint that ensured that all the sampled
probabilities were roughly equal before continuing with the rewards.
Checkpoint criteria must be satisfied before further rewards can be
computed. This way, the model is settling on some method of assigning
probabilities to propositions before trying to game the reward function.
However, this kills the relative advantage in GRPO, which relies on at
least some diversity of outcomes.

After numerous attempts at resolving this dilemma, we settled on
averaging the sampled probabilities for each query and rewarding each
outcome if the variance was below a certain threshold. Then, we set a
checkpoint to ensure that the averaged probabilities are in $[0, 1]$.
This way, there is diversity enough for there to be relative advantages
between the samples, while still encouraging the model to learn some
method of assigning probabilities to propositions. Finally, we grant a
reward if these averaged probabilities assigned to propositions and
their negations add to one (within a reasonable tolerance). The highest
possible reward is 5.0, which we see is attained in Figure
[2](#nonreasoning){reference-type="ref" reference="nonreasoning"}.

# Experimental Results

Although we were unable to train a model that was consistently aligned
in probabilities, our work reveals interesting insights on LLM behavior.
Initially, we assumed that models fine-tuned for probabilistic reasoning
would perform better. However, this is conflating diachronic and
synchronic consistency. We realized by the model responses that although
the reasoning models would begin to match the format reward, they would
actually struggle to give good estimates for probabilities due to a
behavior we call schema-matching. Looking at the model responses we
hypothesize that models fine-tuned on math and probability problems
become reliant on memorized rules (such as normativity) and finding the
relevant bits of information in the prompt. However, when this data is
not provided, they fail to incorporate any prior beliefs that may have
been learned during training and would sometimes hallucinate team
records or standings in order to work the problem out. On the other
hand, models without reasoning capabilities were able to more reliably
produce valid probabilities even if they were not always consistent.
However, the responses of the non-reasoning models did not indicate a
convincing ability to draw on prior knowledge either. We include an
example of the non-reasoning model demonstrating use of prior knowledge,
but we omit the responses of the reasoning model for the sake of
brevity.

## Train metrics for Qwen3-4B in reasoning mode 
<img width="2310" height="1219" alt="3c92e33872750d5a2547c5db9a178349206550af" src="https://github.com/user-attachments/assets/3a78a1f8-33f2-4c3f-ad76-704b27feee37" />


### Train metrics for Qwen3-4B in non-reasoning mode
<img width="2310" height="1219" alt="a2872a7d67ecc6f7d2170339120961790c3d500b" src="https://github.com/user-attachments/assets/737f2977-c851-45dd-b585-120346d93c43" />


**Sample Responses (Non-Reasoning Model):**

> **1.**
> `The probability of Sheffield United winning against Tottenham on 02/07/2020 cannot be accurately estimated without specific historical data or current form analysis. However, based on general knowledge of football match outcomes, the probability is low for the underdog team (Sheffield United) against a strong team like Tottenham.`
>
> $<$`answer`$>$ ` 0.1 ` $<$`/answer`$>$
>
> **2.**
> `The probability of Sheffield United not winning the match cannot be determined with certainty without specific data on their performance and the match conditions. However, based on general knowledge that Tottenham is a stronger team, the probability that Sheffield United does not win is estimated to be high.`
>
> $<$`answer`$>$ `0.9` $<$`/answer`$>$

# Conclusion

So far, we have only seen limited successful in training LLMs to be
diachronically consistent/immune to Dutch book exploitation. However, we
have found experimental evidence that non-reasoning models are better at
developing cross-context reasoning (see Figures
[1](#reasoning){reference-type="ref" reference="reasoning"} and
[2](#nonreasoning){reference-type="ref" reference="nonreasoning"}). Our
explanation for this is that reasoning models have been fine-tuned to
follow certain logical schemas to solve tasks that only require
reasoning about information that has been given in the current context.
Because we are demanding our models engage in reasoning that extends
beyond the current context, this behavior must be unlearned at some
level.

Our training task requires a deep understanding of the given
propositions that goes beyond linguistic structure. It requires the
model to synthesize information in a way that is not necessary for next
token prediction. This requires a very rich semantic feature space, and
the smaller models which we employed might not be fully adequate.
Furthermore, what we are doing with LLMs is very unique, something which
these models have not been trained to accomplish. Given that the
non-reasoning Qwen3-4B was able to maximize the reward function, we
conclude that more time training, and potentially larger models, will be
sufficient to see much greater success in this task.
