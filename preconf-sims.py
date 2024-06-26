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

# %% [markdown]
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


# %% [markdown]
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
# realised increments

def bidder_gen_realizations(bidder):
    "Generate the realized type at delivery for a sample of bidders. Negative realizations are allowed."
    sample_size = len(bidder["value"])
    
    realized_increments = np.zeros(sample_size, dtype=np.float64)
    for i in range(sample_size):
        realized_increments[i] = rng.normal(loc=bidder["premium"][i], scale=bidder["noise"][i])
        
    # arithmetic increment
    return bidder["value"] + realized_increments


# %% [markdown]
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
    print(f"Support index: {support_index}")
    
    # Theorem: last_index < len(bid_profile_fee)
    if support_index == len(bid_profile_fee):
        support = reserve_fee
    else:
        support = bid_profile_fee[indices_sorted][support_index]
    
    return support


# %% [markdown]
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
    print(f"Total gas usage in message pool: {bid_profile_gas.sum()}")
        
    ### Obtain simulated ask price
    
    ask = bid_profile_ask(bid_profile_fee, bid_profile_gas, weight, capacity) # correctly handles underfull knapsack
    bid = ask + 1
        
    # Apply limit constraint
    bid = min(bid, limit)
    
    return bid


# Another opponent simulation strategy: match each opponent to a randomly selected (winning) bid from
# the previous block and have them bid that (again).
# Even simpler: just use the exact set of winning bids from the previous block as a bid profile.

# %%
bid_against_simulated_population(100, 230000, pop_size=300)

# %%
# Main loop 
# Generate a sample population and run two rounds of bidding.

# %% [markdown]
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

# %%
