# Preconfirmation auction simulations: bidder strategy

## Introduction

### Glossary

I include a short glossary of terms from general knapsack problems and mechanism design literature to Ethereum-specific terms.

| Ethereum/mev-commit | knapsack/mechanism |
| ------------------- | ------------------ |
| Gas/blockspace      | Capacity           |
| Transaction/bundle  | Item               |
| Fee per unit gas    | Efficiency         |
| Total fee           | Profit             |
| Provider            | Auctioneer         |
|                     |                    |

## Model description

Provider parameters:

* A natural number $N>0$, the number of rounds to hold preconf auctions.
* A sequence of natural numbers $K_0\leq\cdots \leq K_{N-1}$, where $K=K_{N-1}$ is the block capacity. We adopt the notational convention that $K_i=\check{K_i}=0$ for $i<0$.
* Optionally: a sequence of prices $R_0,\ldots,R_{N-1}$ to use as reservation prices in each round.

Starting from round zero, in round $i+1$ we suppose that a quantity $\check{K}_i$ has been allocated in round $i$ and put a quantity $K_i-\check{K_i}$ up for sale in a static knapsack auction. 

Bids are selected using the greedy split selection rule, which selects bids into the knapsack in descending order of fee (per unit capacity) until it encounters one that doesn't fit. (A full greedy algorithm would carry on iterating to the end of the list, selecting any further bids it finds that fit.)

```python
def greedy_split_selection(bids: array[tuple[float, int]], capacity: int) -> array[bool]:
    n_bids = length(bids)
    selected: array[bool; n_bids] # selection mask
    
    bids_sorted_by_fee <- descending_sort(bids, on="fee") # sort bids in descending order of fee
    bids_cumulative_total = array[int; n_bids]
    
    for i = 0..n_bids:
        bids_cumulative_total[i] = sum(bids_sorted_by_fee)[:i]
        if bids_cumulative_total[i] <= capacity:
            selected[i] = True
        else:
            selected[i] = False
```

### Bidder types

In a single parameter bidder model, the bidder's *type* $\theta$ — that is, the value they assign to outcomes given their private information — is captured by the highest price $v(\theta)$ the bidder would pay, under all possible mechanisms, to get their item into the knapsack.

In a dynamic model, the type is a process that evolves with each round $\{v_t\}_{t=0}^{N-1}$. To constrain the dimensionality of the parameter space, we assume that bidder types evolve according to a Wiener process with drift $d$ and instantaneous volatility $\sigma$. The drift $d$ can be interpreted as the *preconf premium*, that is, the expected premium per unit time a bidder is prepared to pay for early preconfirmation. We note that, since early bids give up optionality and incur additional execution outcome risk, it is realistic for $d$ to be negative as well as positive.

Hence, our bidder types have three parameters $\theta = (v_0, d, \sigma)$.

### Bidder strategies

We assume the bidders know in advance the numbers $K_0\leq K$ (and the reserves $R_0,R_1$ if used).

We implement two fairly trivial approaches to bidder strategies:

* *Direct revelation.* Bidders bid their true value $v(\theta)$. (This is a very naive model: in reality, given that our auction model is pay-as-bid, we would expect bidders to [shade](https://en.wikipedia.org/wiki/Bid_shading) their bids.)
* *Bid against simulated population.* Bidders make an ansatz about the population against which they are competing, including assumptions about the distribution of their opponents' types and the strategy they will use, then simulate a bid profile and bid as if against that.
* *Bid against simulated population with Monte-Carlo sampling.* (Not implemented yet).

### Implementation notes

We introduce the following interface

```python
# Proposed types

type BidderPopulation # distribution from which bidder types are sampled
    def sample() -> list[Bidder] # sample a number
    
type Bidder
    bidder_type: dict
        
    def bid(environment) -> Bid
    
type BidProfile = list[Bid]
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