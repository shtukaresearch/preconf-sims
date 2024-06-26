# Preconfirmation auction simulations: bidder strategy

I use `pdm run jupyter notebook` to run this thing, but you can probably get by with whatever virtual environment tooling you prefer, as long as it supports `pyproject.toml`. If you use your system Jupyter installation, make sure jupytext is installed so the percent script `preconf-sims.py` can be automatically converted to `.ipynb` format.

## Introduction

* 

## Model description

The mev-commit preconfirmation auction model is a dynamic allocation of fixed capacity. A population of bidders want to get items (transactions, bundles) of various sizes into a knapsack (block) of size $K$, and issue bids to a revenue-maximising provider with the power to decide which items get in. Bidders can issue bids at any time in the auction interval $[0,T]$. Due to a mechanism where a bid decays automatically over time, providers must respond quickly to bids or discard them; for simplicity, we assume that providers must make an *immediate* decision on each bid. Once a bid is accepted, an appropriate amount of the knapsack is committed and cannot be reallocated to other items; that bidder subsequently becomes inactive.

In continuous time, the provider's decision space is infinite-dimensional. Clearly, the provider needs to impose some constraints upon itself to get a tractable problem. In this repo we've taken the approach of restricting to a finite number of *epochs*, assuming that in each epoch, all bids and the provider decision occur instantaneously at the end of each round.

### Provider parameters

The provider needs to decide what mechanism to use to sell the preconfs.

* A natural number $N>0$, the number of rounds to hold preconf auctions.
* A sequence of natural numbers $K_0\leq\cdots \leq K_{N-1}$, where $K=K_{N-1}$ is the block capacity. We adopt the notational convention that $K_i=\check{K_i}=0$ for $i<0$.
* Optionally: a sequence of prices $R_0,\ldots,R_{N-1}$ to use as reservation prices in each round.

Starting from round zero, in round $i+1$ we suppose that a quantity $\check{K}_i$ has been allocated in round $i$ and put a quantity $K_i-\check{K_i}$ up for sale in a static knapsack auction. 

Bids are selected using the greedy split selection rule, which selects bids into the knapsack in descending order of fee (per unit capacity) until it encounters one that doesn't fit. (A full greedy algorithm would carry on iterating to the end of the list, selecting any further bids it finds that fit.)

```python
def greedy_split_selection(bids: array[tuple[float, int]], capacity: int) -> array[bool]:
    n_bids <- length(bids)
    selected: array[bool; n_bids] # selection mask
    
    bids_sorted_by_fee <- descending_sort(bids, on="fee") # sort bids in descending order of fee
    bids_cumulative_total: array[int; n_bids]
    
    for i = 0..n_bids:
        bids_cumulative_total[i] <- sum(bids_sorted_by_fee)[:i]
        selected[i] <- (bids_cumulative_total[i] <= capacity)
```

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

We implement two fairly trivial approaches to bidder strategies:

* *Direct revelation.* Bid your current limit value $v$. This is a very naive model: in reality, given that our auction model is pay-as-bid, we would expect bidders to [shade](https://en.wikipedia.org/wiki/Bid_shading) their bids.
* *Bid against simulated population.* Bidders make an ansatz about the population against which they are competing, including assumptions about the distribution of their opponents' types and the strategy they will use, then simulate a bid profile and bid in order to beat the *ask* of the bid pool at the target item size.
* *Bid against simulated population with quantile bound.* (Not implemented yet).

Note that all of these strategies are *static*, in that they only take into consideration the situation in the current round. A basic improvement on this is to using the expected return from bidding in *later* rounds to bound bidding in the current round: indeed, a bidder is unlikely to bid early if he is highly likely to achieve a greater surplus by waiting until a later round.

### Implementation notes

We introduce the following interfaces:

```python
# Proposed types

interface BidderPopulation # distribution from which bidder types are sampled
    def sample() -> list[Bidder] # sample a number
    
# Implementations of this interface need to choose what bidder "(computer) type" they 
type NormalBidderPopulation:
    rng = np.random.default_rng()
    pop_size: int
    mean: float
    variance: float
        
    def sample(self) -> list[Bidder]:
        return self.rng.normal(mean, variance, size=pop_size)
    
type StaticBidderPopulation:
    data: array[Bid]
        
    def sample(self) -> list[Bidder]:
        return data
    
type Bid = (float v, float g) # positive reals representing the limit value and gas.

interface Bidder
    def bid(self) -> Bid
    
# Bidder implementations

interface StaticBidder:
    theta: float
    
type BidProfile = list[Bid]

interface 
```





## Assumptions

* The Primev auction model is fully dynamic, meaning that bidders can issue bids at any time and providers can accept bids in its inbox at any time. That means that even with a deterministic arrival process for bids, the provider's decision space is infinite-dimensional. Clearly, the provider needs to impose some constraints upon itself to get a tractable problem.

* The simplest possible approach that still retains some of the dynamic aspects of the problem is what the dynamic mechanism design literature calls *sequential allocation of fixed capacity*. In it, the block capacity $K$ is repeatedly offered up for auction in $N$ rounds. Bids that are not accepted in one round expire by the next. (In mev-commit, one can approximate this situation by assuming that the rounds are far enough apart that the bid decay function makes any bid available in round $n$ unattractive in round $n+1$.)

* We push this approach a bit further by allowing the provider to choose restricted capacities $K_0\leq K_1\leq\cdots\leq K_{N-1}=K$ to auction off in each of the $N$ rounds. If $\check{K}_i$ is the amount of the knapsack sold off by the $i$th round, $K_{i+1}-\check{K}_i$ will be available in the $(i+1)$th round. 

  There are a number of reasons the provider may wish to do this.[^pricing] One is that restricting capacity can create artificial congestion which pushes up floor prices. Another is that information surfaced by bidder behaviour in earlier rounds can be used to make more informed decisions about parameters in later rounds.

* In general, discrete time models are theoretically more tractable than continuous time ones. Continuous time environments can be approximated by a high resolution discrete time model.

  For example, Myerson-style optimal mechanisms have been analysed in the discrete time environment.[^pavan-myerson]

* I implemented the greedy split knapsack packing algorithm, which gives up after first encountering an item that doesn't fit, rather than the full greedy algorithm. This was purely because the greedy split algorithm is easier to vectorise using numpy. In general, greedy split will waste more of the knapsack capacity than greedy, which accordingly puts some upward pressure on prices.

[^pricing]: https://mirror.xyz/preconf.eth/iPfGsj55-C-D13hyrj_hj2tHAKU7xzeqltZ6gIum3j4
[^pavan-myerson]: https://faculty.wcas.northwestern.edu/apa522/DMD-final.pdf

## Things to work on

Here we discuss the ways in which the model likely diverges from reality and what could be done to improve it.

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

* We use *block* because
* *Capacity* is more general than block size: it can apply to constrained amounts of blockspace made available in an auction.
* *Size* is shorter than gas limit and reasonably intuitive. (However, in specific contexts like trading there may be danger of confusion with parameters like trade size.)
* *Item* can refer to either a transaction or bundle. We use the term as the model is agnostic to the contents or internal structure of items.
* We understand the term *fee* as always meaning *fee per unit size*. The bid/value for the whole item is the *total fee*, computed as $fee\cdot size$.