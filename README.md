# Preconfirmation auction simulations: bidder strategy

## Using the notebook

### Environment setup

The notebook is bundled with a `pyproject.toml` containing all dependencies, so you can create your virtual environment using that and run it in a Jupyter instance using a kernel inside that environment. I've bundled `jupyter` itself into the dependencies, so an easy way to do this is to run `jupyter` from within the venv, where a suitable ipykernel will be automatically installed. I use `pdm run jupyter notebook` to run this thing, but you can probably get by with whatever virtual environment tooling you prefer.

If you use another Jupyter installation, make sure jupytext is installed so the percent script `preconf-sims.py` (checked into Git) can be automatically converted to `.ipynb` format.

### Quickstart

To get a feel for usage, jump straight to the Examples section.

### Summary of sections

* **Provider and bidder population parameters.** Setting parameters for the model as global variables.
* **Sampling methods.** Functions to sample populations of static and dynamic bidders, and to realise the private values of Wiener bidders at the second timestep.
* **Selection rule.** Implementation of the greedy split mechanism selection rule, and a function that computes the "floor" price of a knapsack against a given bid profile and item size — that is, the infimum of the set of bids that would get an item of the given size into the knapsack against a given bid pool.
* **Bid strategies.** Implementation of static bidding strategies. We implement passthrough bidding and two variants of an algorithm that simulates the behaviour of other bidders to predict a floor bid.



## Model description

The mev-commit preconfirmation auction model is a dynamic allocation of fixed capacity. A population of bidders want to get items (transactions, bundles) of various sizes into a knapsack (block) of size $K$, and issue bids to a revenue-maximising provider with the power to decide which items get in. Bidders can issue bids at any time in the auction interval $[0,T]$. Due to a mechanism where a bid decays automatically over time, providers must respond quickly to bids or discard them; for simplicity, we assume that providers must make an *immediate* decision on each bid. Once a bid is accepted, an appropriate amount of the knapsack is committed and cannot be reallocated to other items; that bidder subsequently becomes inactive.

In continuous time, the provider's decision space is infinite-dimensional. Clearly, the provider needs to impose some constraints upon itself to get a tractable problem. In this repo we've taken the approach of restricting to a finite number of *epochs*, assuming that in each epoch, all bids and the provider decision occur instantaneously at the end of each round. This discrete time approach has the advantage of analytic tractability,[^pavan-myerson] and can be used to approximate continuous time environments by increasing the number of rounds.

[^pavan-myerson]: See for example https://faculty.wcas.northwestern.edu/apa522/DMD-final.pdf. 

### Provider parameters

The provider needs to decide what mechanism to use to sell the preconfs.

* A natural number $N>0$, the number of rounds to hold preconf auctions.
* A sequence of natural numbers $K_0\leq\cdots \leq K_{N-1}$, where $K=K_{N-1}$ is the block capacity. We adopt the notational convention that $K_i=\check{K_i}=0$ for $i<0$.
* Optionally: a sequence of prices $R_0,\ldots,R_{N-1}$ to use as reservation prices in each round.

Starting from round zero, in round $i+1$ we suppose that a quantity $\check{K}_i$ has been allocated in round $i$ and put a quantity $K_i-\check{K_i}$ up for sale in a static knapsack auction.[^provider-strategy]

[^provider-strategy]: There are a number of reasons the provider may wish to do this. One is that restricting capacity can create artificial congestion which pushes up floor prices. Another is that information surfaced by bidder behaviour in earlier rounds can be used to make more informed decisions about parameters in later rounds. See https://mirror.xyz/preconf.eth/iPfGsj55-C-D13hyrj_hj2tHAKU7xzeqltZ6gIum3j4.

Bids are selected using the greedy split selection rule, which selects bids into the knapsack in descending order of fee (per unit capacity) until it encounters one that doesn't fit. 

```python
# Greedy Split algorithm pseudocode
def greedy_split_selection(bids: array[tuple[float, int]], capacity: int) -> array[bool]:
    n_bids <- length(bids)
    selected: array[bool; n_bids] # selection mask
    
    bids_sorted_by_fee <- descending_sort(bids, on="fee") # sort bids in descending order of fee
    bids_cumulative_total: array[int; n_bids]
    
    for i = 0..n_bids:
        bids_cumulative_total[i] <- sum(bids_sorted_by_fee)[:i]
        selected[i] <- (bids_cumulative_total[i] <= capacity)
```

A full greedy algorithm would carry on iterating to the end of the list, selecting any further bids it finds that fit. Greedy Split is used here purely because it is easier to vectorise using numpy. In general, greedy split will waste more of the knapsack capacity than greedy, which accordingly puts some upward pressure on prices.

### Bidder parameters

#### Bidder private values ("type")

In a single parameter bidder model, the bidder's *type* $\theta$ — that is, the value they assign to outcomes given their private information — is captured by the *limit fee*, $v=v(\theta)$, that is, the highest fee the bidder would pay, under all possible mechanisms, to get their item into the knapsack.[^type-disambiguation]

[^type-disambiguation]: In implementations, care must be taken not to confuse the auction theory notion of type and the *data type* of the bidder struct, which should encode the data type of the bidder's private values (but not any particular realisation of those values!) as well as the data types of other relevant information such as common knowledge and the bidder strategy.

In a dynamic model, the type is a process that evolves with each round $\{v_t\}_{t=0}^{N-1}$. To constrain the dimensionality of the parameter space, we assume that bidder types evolve according to a Wiener process with drift $d$ and instantaneous volatility $\sigma$. The drift $d$ can be interpreted as the *preconf premium*, that is, the expected premium per unit time a bidder is prepared to pay for early preconfirmation. We note that, since early bids give up optionality and incur additional execution outcome risk, it is realistic for $d$ to be negative as well as positive.

Hence, our bidder types have three parameters $\theta = (v_0, d, \sigma)$. 

Our population implementations assume that bidder types are i.i.d.; this is the *independent private values* model typical in auction theory. It is probably not realistic, as in practice bidders compute their valuations for inclusion based on a mixture of public and private signals.

#### Bidder state

As the rounds progress, bidders accumulate state as a history accumulates of what the bidder learned during previous rounds. This history consists of private information such as the present internal type $v_n$, the history of previous bids the bidder has made, and whether they were accepted, as well as public information such as exogenous price signals.

In our simplistic model, the only data bidders will use at epoch $n$ is their present limit fee $v_n$ and the parameters $(d, \sigma)$ of the limit fee Wiener process.

#### Bidder private history (experimental)

Abstractly, the bidder state in round $n$ forms a vector $\theta_n = (v_n, \theta^{n-1}, h^{n-1})$ where $\theta^i := (\theta_0,\ldots,\theta_i)$ and $h^i=(E_0,\ldots,E_k)$ is a *history* of events in epochs up to $i$​. In mev-commit auctions, bidders do not learn directly about any bids submitted by their competitors until the block is produced; but they recall their states in previous rounds, the bids they submitted, and whether or not those bids were accepted. In sophisticated models, this can be used to update models of the provider and the other bidders and hence adjust strategy in later rounds. However, we haven't implemented this.

We can model an event as an element of the set
$$
\texttt{Event} := \{0,\ldots,i\} \times \left(\{\texttt{Bid}(b,h)\} \cup \{\texttt{Accept}(h)\} \right),
$$
where:

* $\texttt{Bid}(b,h)$ represents a bid with amount $b$ and hash $h$;
* $\texttt{Accept(h)}$ represents an accept message for the bid with hash $h$;
* $h^i$ is a sequence $((t_0,m_0),\ldots,(t_k,m_k))$ where:
  * $t_0\leq \cdots \leq t_k\in\{0,\ldots,i\}$.
  * $\texttt{Accept}(h)$ can occur only immediately after a bid of the form $\mathtt{Bid}(b,h)$.

We can encode further assumptions about dynamic bidder strategy as constraints on this sequence:

+ A bidder stops bidding after a bid is accepted $\Rightarrow$ the $\texttt{Accept}$ message occurs at most once, at the end of the sequence.
+ A bidder bids at most once per round $\Rightarrow$ 

#### Common knowledge

We assume the bidders know in advance the numbers $K_0\leq K$ (and the reserves $R_0,R_1$ if used).

In some theoretical models, the bidder *population* (more precisely, the distribution of bidder private values) is also considered common knowledge. Such models can be used to analyse Bayes Nash equilibria, but raise implementation challenges in practice (how can a bidder $i$ simulate its competitors using the same strategy as $i$?) and we have not included them.

#### Bidder strategy

The most straightforward bidder strategy is:

1. *Direct revelation.* Bid your current limit value $v$. This is a very naive model: in reality, given that our auction model is pay-as-bid, we would expect bidders to [shade](https://en.wikipedia.org/wiki/Bid_shading) their bids. It has a payoff of zero in all realisations.

A more realistic strategy is to bid against a *simulation.* Bidders should have access to a *simulator* from which they can sample or compute statistics over the actions of other bidders and the provider. Note that we have assumed the provider announces in advance what they will do, so the meat is in forecasting the behaviour of other bidders.

2. *Bid against simulated population.* Sample once from an opposing bidder population and bid just above the resulting floor. Easy to implement and may be reasonable in aggregate over many runs.
3. *Empirical chance bound.* Fix a value ${0 < p < 1}$ and bid against a simulated population so as to achieve at least probability $p$ of inclusion. This probability can be estimated by running the simulation many times and evaluating an empirical quantile function at $p$. (Strategy 2. is the special case of this where the sample size is 1.)
7. *Expectation maximisation.* Numerically optimise the expected surplus $u(b) = p(b)(v-b)$ where $p$ is the probabilty of acceptance of a bid $b$. Given a simulator, the empirical distribution function can be used to estimate $p(b)$.

The strategies described so far all apply to a single round static knapsack auction. The dynamic, multi-round setting complicates the strategy space in two ways:

* Bidders can use a forecast of the outcomes of future rounds to inform decision-making in the current one.
* Bidders recall whatever information about the outcomes of previous rounds was made available to them. In the mev-commit model, this consists of the list of bids they made and whether or not they were accepted.

Let's focus on the simplest possible case of a two-round auction. Bidders have two fundamental options:

5. Bid in the first round (using any of the single-round strategies), and if the bid is rejected, try again in the second.
6. Sit out the first round and bid only in the second.

How does one decide between these strategies? A forward-thinking bidder should not issue a bid $b$ in the first round if he believes there is a high probability of achieving a greater surplus in the second. Intuitively, this could happen if $K_0$ is small and so the expected floor in the first round is high, and the bidder's own internal preconf premium (equal to $-d$, the negation of the drift of their value process) is not too large. 

The simplest approach to express is to choose the one with higher expected surplus.

More complex strategies might even use this forecast to adjust the size of the bid in the first round, rather than simply deciding whether or not to participate.

### Simulators

1. *IID direct revelation population.* Assume other bidders use a direct revelation strategy and sample their internal types i.i.d. from some distribution.
2. TODO


## Things to work on

Here we will discuss the ways in which the model likely diverges from reality and what could be done to improve it.

* *TODO*

## Glossary

I include a short glossary of terms from general knapsack problems and mechanism design literature to Ethereum-specific terms.

| Ethereum/mev-commit    | knapsack/mechanism |
| ---------------------- | ------------------ |
| Block                  | Knapsack           |
| Block size             | Capacity           |
| Gas limit              | Size               |
| Transaction/bundle     | Item               |
| Fee (bid) per unit gas | Efficiency         |
| Total fee (bid)        | Profit, value      |
| Provider               | Auctioneer         |

For the most part, we prefer to use generic terms from the theory of knapsack problems and mechanism design.

* *Capacity* is more general than block size: it can apply to constrained amounts of blockspace made available in an auction.
* *Size* is shorter than gas limit and reasonably intuitive. (However, in specific contexts like trading there may be danger of confusion with parameters like trade size.)
* *Item* can refer to either a transaction or bundle. We use the term as the model is agnostic to the contents or internal structure of items.
* We understand the term *fee* as always meaning *fee per unit size*. The bid/value for the whole item is the *total fee*, computed as $fee\cdot size$.
