# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # mev-commit preconf sims

# %%
# !python -V

# %%
# imports
import numpy as np

# seed randomness
rng = np.random.default_rng()

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Provider and bidder population parameters

# %% [markdown]
# Bidder population model: 
# 1. Each bidder has private values following an arithmetic Wiener process with starting value $v_t$, drift $\mu$, and instantaneous volatility $\sigma$. (This "drift" is the preconf premium.) Hence bidder types have three parameters.
# 2. With this model, `v_0` can be negative. To avoid this we replace it with max{v_0,0}, although negative bids can also be a thing.
# 3. $v_t$ values are drawn from a geometric distribution with rate `v_t_mean`. (This is the max entropy hypothesis.)
# 4. mu is normally distributed with mean drift_loc and standard deviation drift_scale
# 5. For simplicity, instantaneous variance is a constant 1.
# 6. Bidder gas requirements are drawn from a geometric distribution with mean `mean_gas_usage`
#
# ```mermaid
# graph TD;
#     A-->B;
#     A-->C;
#     B-->D;
#     C-->D;
# ```

# %%
# PROVIDER

BLOCK_SIZE = 30000000 # block size in gas
PRECONF_CAPACITY = 3000000 # amount of block available for preconfirmation

# PRECONF_TIME = 1 # time to delivery of preconfs, i.e. how long before block time preconf auction is held
# we don't use this

# %%
# BIDDER POPULATION

N_BIDDERS = 300 # default bidder sample size

# population parameters for sampling types \theta = (v_t,\mu,\sigma)
v_t_mean = 100    # imagine this is denominated in mwei
drift_loc = 0     # average preconf premium vanishes
drift_scale = 75  # standard deviation of preconf premium types
sigma = 100       # (constant) type instantaneous volatility

# distribution of bidder gas usage
mean_gas_usage = 150000


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Sampling

# %%
# Methods to sample bidder types from population

def population_gen_types_static(sample_size=N_BIDDERS) -> dict:
    return {
        "value": rng.geometric(p=1/v_t_mean,       size=sample_size),
        "gas":   rng.geometric(p=1/mean_gas_usage, size=sample_size)
    }

def population_gen_types_dynamic(sample_size=N_BIDDERS) -> dict:
    return {
        "value":   rng.geometric(p=1/v_t_mean,       size=sample_size),
        "gas":     rng.geometric(p=1/mean_gas_usage, size=sample_size),
        "premium": rng.normal(loc=drift_loc, scale=drift_scale, size=sample_size),
        "noise":   np.full(sample_size, sigma, dtype=np.float64),
    }


# %%
# A population drawn by copying some data,
# assuming those bidders used a direct revelation strategy in the past
# and remain unchanged today.

def population_gen_from_data(fee, gas) -> dict:
    return {
        "value": np.array(fee),
        "gas": np.array(gas)
    }


# %%
# realised increments

def bidder_gen_realization(bidder):
    "Generate the realized type at delivery for a sample of bidders. Negative realizations are allowed."
    
    realized_increment = rng.normal(loc=-bidder["premium"], scale=bidder["noise"])
    # arithmetic increment
    return bidder["value"] + realized_increment

def bidder_gen_realizations(bidder):
    "Generate the realized type at delivery for a sample of bidders. Negative realizations are allowed."
    sample_size = len(bidder["value"])
    
    realized_increments = np.zeros(sample_size, dtype=np.float64)
    for i in range(sample_size):
        realized_increments[i] = rng.normal(loc=-bidder["premium"][i], scale=bidder["noise"][i])
        
    # arithmetic increment
    return bidder["value"] + realized_increments


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Selection rule

# %% [markdown]
# A *bid profile* on the population $[N]$ is a vector of tuples $((b_i,g_i)_{i=0}^{N-1})$ consisting of the *bid value* $b_i$ and *item weight*, a.k.a. *gas limit* $g_i$ of each bid; here $N$ is the population size `N_BIDDERS`. 
#
# *TODO: Decide once and for all on use of abstract or concrete terminology, i.e. "weight" vs. "gas".*
#
# We store a bid profile as a pair of arrays `(value, gas)` of the same shape. An absent bid is represented by a `NaN` value. Although in deployments both of these arrays are (big) integer typed, for the purposes of this simulation there is little reason to preserve this. Hence, we treat everything as a `float64`.
#
# We will operate on bid profiles by:
# * Running selection algorithms
# * Computing minimum bids to add to the profile to get into the selection.
#
# Maybe these functions should go into a trait on an `Auctioneer` type at some point.

# %%
def mechanism_selection(bid_price, bid_gas, capacity=BLOCK_SIZE, reserve_fee=0):
    """
    Takes a bid profile and returns a mask indicating which bids were selected for inclusion.
    
    Implements greedy split algorithm, which gives up as soon as it reaches an item that doesn't fit.
    """
    # TODO apply reservation price and skip NaN values (representing no bid)
    
    indices_sorted = np.flip(np.argsort(bid_price), axis=-1) # sort from largest to smallest
    
    gas_summed = np.zeros(bid_price.shape, dtype=np.int64)
    gas_summed[indices_sorted] = np.cumsum(bid_gas[indices_sorted])
    return gas_summed <= capacity


# %% [markdown]
# ### Minimum bid
#
# Suppose given an auction environment and a bid profile. We define the *ask* at a given `target_gas` to be the fee one's bid must clear in order to get an item of weight `target_gas` into the knapsack in competition against the given bid pool.

# %%
def bid_profile_ask(bid_profile_fee, bid_profile_gas, target_gas, capacity=BLOCK_SIZE, reserve_fee=0):
    """
    Return the bid one must beat to get an item of size `target_gas` into a 
    knapsack of size `capacity` competing against a given bid profile.
    
    Assume the provider uses the greedy split algorithm. (However, I think this
    approach finds the support bid even if the full greedy algorithm is used.)
    """
    
    # Algorithm:
    # Compute greedy packing of knapsack of size capacity - target_gas
    # Floor is the next highest bid after the last one that got in. 
    
    indices_sorted = np.flip(np.argsort(bid_profile_fee), axis=-1) # sort from largest to smallest
    
    gas_summed = np.zeros(bid_profile_fee.shape, dtype=np.int64)
    gas_summed[indices_sorted] = np.cumsum(bid_profile_gas[indices_sorted])
    
    # compute last index in sorted list that makes it into the truncated knapsack
    # np.searchsorted is the fastest numpy method to do this according to
    # https://stackoverflow.com/questions/16243955/numpy-first-occurrence-of-value-greater-than-existing-value
    
    support_index = np.searchsorted(gas_summed[indices_sorted], capacity-target_gas, side="right")
    #print(f"Support index: {support_index}")
    
    # Theorem: last_index < len(bid_profile_fee)
    if support_index == len(bid_profile_fee):
        support = reserve_fee
    else:
        support = bid_profile_fee[indices_sorted][support_index]
    
    return support


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Bid strategies

# %% [markdown]
# Note: `value = max_priority_fee_per_gas + preconf_bid/gas_limit`. We assume `base_fee_per_gas` is known.
#
# Interfaces:
#
# ```
# simulator(environment, population) -> bid_profile
# bid_against(type, environment, bid_profile) -> bid
# ```

# %%
def bid_direct_revelation(limit):
    "Simply bid the current value."
    return limit

def bid_against_simulated_population(limit, weight, pop_size = N_BIDDERS, capacity=BLOCK_SIZE):
    """
    Simulate the bids of other bidders and bid an amount that would just get in
    if the simulated bids were realised.
    """
    
    ### Each bidder simulates a population bidding profile
    
    # Use a globally defined prior distribution to simulate opponents
    #type_profile_fee = rng.geometric(p=1/v_t_mean,       size=pop_size) 
    #type_profile_gas = rng.geometric(p=1/mean_gas_usage, size=pop_size)

    type_profile = population_gen_types_static(pop_size)
    type_profile_fee = type_profile["value"]
    type_profile_gas = type_profile["gas"]
    
    # Assume opponents use the direct revelation bidding strategy
    bid_profile_fee = bid_direct_revelation(type_profile_fee)
    bid_profile_gas = type_profile_gas
    #print(f"Total gas usage in message pool: {bid_profile_gas.sum()}")
        
    ### Obtain simulated ask price
    
    ask = bid_profile_ask(bid_profile_fee, bid_profile_gas, weight, capacity) # correctly handles underfull knapsack
    bid = ask + 1
        
    # Apply limit constraint
    bid = min(bid, limit)

    # Don't bid a negative number (although this is possible in principle)
    return max(0, bid)


# Another opponent simulation strategy: match each opponent to a randomly selected (winning) bid from
# the previous block and have them bid that (again).
# Even simpler: just use the exact set of winning bids from the previous block as a bid profile.

# %%
def sim_floor(size, pop_size = N_BIDDERS, capacity=BLOCK_SIZE):
    """
    Simulate one round of bidding and compute the floor at a given size.
    """
    type_profile = population_gen_types_static(pop_size)
    type_profile_fee = type_profile["value"]
    type_profile_gas = type_profile["gas"]
    
    # Assume opponents use the direct revelation bidding strategy
    bid_profile_fee = bid_direct_revelation(type_profile_fee)
    
    bid_profile_gas = type_profile_gas
    
    return bid_profile_ask(bid_profile_fee, bid_profile_gas, size, capacity=capacity)

DEFAULT_SAMPLE_SIZE = 1000

def sim_floor_quantile(size, quantile, pop_size = N_BIDDERS, capacity=BLOCK_SIZE, sample_size=DEFAULT_SAMPLE_SIZE):
    """
    Get an estimate of the quantile of the floor price of a simulated population using the "median_unbiased" estimator.
    """

    floor_sims = [sim_floor(size, pop_size=pop_size, capacity=capacity) for _ in range(sample_size)]
    return np.quantile(floor_sims, quantile, method="median_unbiased")


def bid_against_sim_quantile(size, quantile, pop_size = N_BIDDERS, capacity=BLOCK_SIZE, sample_size=DEFAULT_SAMPLE_SIZE):
    """
    Return a bid that would make it in with probability `quantile`
    against a simulated direct revelation populations.
    """

    return max(0, 1 + sim_floor_quantile(size, quantile, pop_size=pop_size, capacity=capacity, sample_size=sample_size))

def bid_against_sim_quantile_limit(limit, size, quantile, pop_size = N_BIDDERS, capacity=BLOCK_SIZE, sample_size=DEFAULT_SAMPLE_SIZE):
    # Apply limit constraint
    return max(0,min(limit, bid_against_sim_quantile(size, quantile, pop_size=pop_size, capacity=capacity, sample_size=sample_size)))


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Example: floor prices at various quantiles, sizes, and capacity
#
# Illustrating the fairly obvious fact that shrinking the capacity pushes the floor price right up. Hence, an early preconf auction with a restricted capacity may well be expected to fetch premium prices.

# %%
def print_floor(size, quantile, capacity):
    print("size", end=": ");     print(f"{size}".rjust(7), end="\t")
    print(f"quantile: {quantile}", end="\t")
    print("capacity", end=": "); print(f"{capacity}".rjust(8), end="\t")
    print("|", end="\t")
    print(f"floor: {sim_floor_quantile(size, quantile, capacity=capacity)}")

print_floor( 60000, 0.99, 30000000)
print_floor(120000, 0.99, 30000000)
print_floor(120000, 0.9 , 30000000)
print_floor(120000, 0.99,  3000000)
print_floor(120000, 0.9 ,  3000000)
print_floor(2000000, 0.99, 30000000)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Example: bidding in two rounds against simulated opponents
#
# An example application for a bidder who wishes to bid in the preconf round and then, if unsuccessful, again in the spot round.
#
# 1. Bid against simulated opponents at 7th decile.
# 2. Realize result (here simulated by a separate "environment" simulator).
# 3. If successful, stop here.
# 4. Else, realize second stage inner value.
# 5. Bid against simulated opponents at 99th centile.
# 6. Realize result.
#
# Observation: if opponents are stupid and bid as though the whole capacity is available, it becomes self fulfilling and the price drops substantially.
# Then X's strategy tends to overbid quite a lot.

# %%
# Bidder X parameters

X = {
    # private moduli
    "value": 330,    # big spender (0.33 gwei)
    "premium": 100,  # positive preconf premium
    "noise": 100,    # standard instantaneous volatility
    "gas": 340000,     # 2-hop swap
    
    # internal model of the competitor population
    "simulator": {   
        "N_BIDDERS": 200,
        "N_BIDDERS_PRECONF_ROUND": 50,
        "v_t_mean": 100,
        "drift_loc": 0,
        "drift_scale": 75,
        "sigma": 100,
        "mean_gas_usage": 180000
    }
}

# 1. Bid against simulated opponents
# Note. Implementation here doesn't use X's model of the population, except for N_BIDDERS_PRECONF_ROUND.
# Using a low quantile so that the bidding quite often progresses to a second round.
X_bid = bid_against_sim_quantile_limit(X["value"], X["gas"], 0.7, pop_size = X["simulator"]["N_BIDDERS_PRECONF_ROUND"], capacity=PRECONF_CAPACITY)
print(f"X bids {X_bid}.")

# 2. Realize preconf round result.
# 2a. Generate geometrically distributed Wiener bidders. (These persist through rounds.)
# Only the first 75 bid in the preconf round.
bidders = population_gen_types_dynamic()
bidders_g = list(zip(bidders["value"], bidders["gas"]))[:75]
# 2b. Generate bids as though the bidders use a similar strategy to X (but with common knowledge of true population parameters)
bid_queue = [(X_bid, X["gas"])]
bid_queue.extend([(bid_against_simulated_population(*Y, capacity=PRECONF_CAPACITY), Y[1]) for Y in bidders_g])
# 2c. Apply selection rule
bid_fee, bid_gas = zip(*bid_queue)
mask = mechanism_selection(np.array(bid_fee), np.array(bid_gas), capacity=PRECONF_CAPACITY)

# 3. If X's bid (at index 0) is selected, stop.
if mask[0]:
    print(f"Made it in at preconf round with surplus {X["value"] - bid_fee[0]}.")
    second_round = False
else:
    print("Continuing to next round.")
    second_round = True
    
    # 4. Realize second stage inner value.
    X["v_1"] = bidder_gen_realization(X)
    
    # 5. Bid against simulated opponents at 99th centile.
    X_bid = bid_against_sim_quantile_limit(X["v_1"], X["gas"], 0.99, pop_size = X["simulator"]["N_BIDDERS"])
    print(f"X bids {X_bid}.")

    # 6. Realize spot round result.
    # 6a. Realize population spot round inner types
    bidders["v_1"] = bidder_gen_realizations(bidders)
    bidders_g = zip(bidders["v_1"], bidders["gas"])
    
    # 6b. Populate spot bid queue.
    bid_queue_spot = [(X_bid, X["gas"])]
    bid_queue_spot.extend([(bid_against_simulated_population(*Y), Y[1]) for Y in bidders_g])

    # 6c. Apply selection rule.
    bid_fee_spot, bid_gas_spot = zip(*bid_queue_spot)
    mask = mechanism_selection(np.array(bid_fee), np.array(bid_gas))
    if mask[0]:
        print(f"Made it in at spot round with surplus {X["v_1"] - bid_fee_spot[0]}.")
    else:
        print("Whoops! Missed the block!")

# %%
# Let's look at the bids people made wth that strategy
from matplotlib import pyplot
s = np.full_like(bid_fee, 0.2)
s[0] = 8 # highlight X's bid
ax = pyplot.scatter(bid_fee, bid_gas, s)
pyplot.title("Bid queue, preconf round")
pyplot.xlabel("fee bid")
pyplot.ylabel("gas_limit")
pyplot.show()

# %%
if second_round:
    s = np.full_like(bid_fee_spot, 0.2)
    s[0] = 8 # highlight X's bid
    ax = pyplot.scatter(bid_fee_spot, bid_gas_spot, s)
    pyplot.title("Bid queue, spot round")
    pyplot.xlabel("fee bid")
    pyplot.ylabel("gas_limit")
    pyplot.show()

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Tests

# %%
# Test support function

bid_profile_fee = rng.geometric(1/100, size=300)
bid_profile_gas = rng.geometric(1/150000, size=300)

ask = bid_profile_ask(bid_profile_fee, bid_profile_gas, 345000)

# The bid (ask+1, 345000) should fit into the block.

bid_profile_fee_extended = np.append(bid_profile_fee, ask+1)
bid_profile_gas_extended = np.append(bid_profile_gas, 345000)

assert mechanism_selection(bid_profile_fee_extended, bid_profile_gas_extended)[-1]


# The bid (ask-1, 345000) should *not* fit into the block. (Assuming greedy split algorithm.)

bid_profile_fee_extended = np.append(bid_profile_fee, ask-1)
bid_profile_gas_extended = np.append(bid_profile_gas, 345000)

assert not mechanism_selection(bid_profile_fee_extended, bid_profile_gas_extended)[-1]

# %%
