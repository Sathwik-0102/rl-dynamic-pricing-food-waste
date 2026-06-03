#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# env_pricing.py

# 'gymnasium' and 'spaces' to define the RL environment interface.
# 'numpy' and 'pandas' for numerical operations and handling the daily panel.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class PricingEnv(gym.Env):
    """
    RL environment for dynamic pricing of a single perishable product family
    in a single store, using a daily panel built from the Favorita dataset.

    - Observation: [normalized base_demand, promo_frac, footfall, dcoilwtico,
                    is_holiday, dow, month, inventory_level, last_action_id]
    - Action: discrete price index -> price multiplier
    - Reward: revenue - waste_penalty * waste_quantity
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        price_multipliers=None,
        base_price=1.0,
        elasticity=0.8,
        shelf_life=3,
        safety_stock_factor=1.3,
        waste_penalty=2.0,
        max_inventory=10_000,
        random_seed=None,
    ):
        super().__init__()

        # Store data
        self.df = df.reset_index(drop=True).copy()
        self.n_days = len(self.df)

        # Config
        self.base_price = base_price
        self.elasticity = elasticity
        self.shelf_life = shelf_life
        self.safety_stock_factor = safety_stock_factor
        self.waste_penalty = waste_penalty
        self.max_inventory = max_inventory

        self.rng = np.random.default_rng(random_seed)

        # Discrete price multipliers (actions)
        if price_multipliers is None:
            # 0=50% off, 1=30% off, 2=no discount, 3=10% markup
            self.price_multipliers = np.array([0.5, 0.7, 1.0, 1.1], dtype=np.float32)
        else:
            self.price_multipliers = np.array(price_multipliers, dtype=np.float32)

        self.n_actions = len(self.price_multipliers)

        # Pre-compute normalized features inside df
        self._build_normalization()

        # Gym spaces
        self.action_space = spaces.Discrete(self.n_actions)

        # obs = [base_demand_norm, promo_frac_norm, footfall_norm, dcoil_norm,
        #        is_holiday, dow_norm, month_norm, inventory_norm, last_action_norm]
        obs_low = np.zeros(9, dtype=np.float32)
        obs_high = np.ones(9, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Internal state
        self.current_day = 0
        self.inventory_buckets = None  # list length shelf_life
        self.last_action = 2  # start with "no discount" index

    # ---------- Normalisation helpers ---------- #
    def _build_normalization(self):
        df = self.df

        def _norm(col):
            c = df[col]
            cmin, cmax = c.min(), c.max()
            if cmax == cmin:
                return c * 0.0, 0.0, 1.0
            normed = (c - cmin) / (cmax - cmin)
            return normed, float(cmin), float(cmax)

        # Ensure required columns exist
        if "dcoilwtico" not in df.columns:
            df["dcoilwtico"] = 0.0

        self.df["base_demand_norm"], self.bd_min, self.bd_max = _norm("base_demand")
        self.df["promo_frac_norm"], self.pf_min, self.pf_max = _norm("promo_frac")
        self.df["footfall_norm"], self.ff_min, self.ff_max = _norm("footfall")
        self.df["dcoil_norm"], self.oil_min, self.oil_max = _norm("dcoilwtico")

        # dow: 0-6 -> /6
        self.df["dow_norm"] = self.df["dow"] / 6.0
        # month: 1-12 -> /11 (0–1)
        self.df["month_norm"] = (self.df["month"] - 1) / 11.0

    def _get_obs(self):
        # Clamp current_day to the last valid index to avoid out-of-bounds
        idx = min(self.current_day, self.n_days - 1)
        row = self.df.iloc[idx]

        inv_level = float(sum(self.inventory_buckets))
        inv_norm = min(inv_level / self.max_inventory, 1.0)

        last_action_norm = self.last_action / max(self.n_actions - 1, 1)

        obs = np.array(
            [
                row["base_demand_norm"],
                row["promo_frac_norm"],
                row["footfall_norm"],
                row["dcoil_norm"],
                float(row["is_holiday"]),
                row["dow_norm"],
                row["month_norm"],
                inv_norm,
                last_action_norm,
            ],
            dtype=np.float32,
        )
        return obs

    def _simulate_demand(self, row, price_multiplier):
        """
        Compute stochastic demand for the current day given the
        panel row and chosen price multiplier.
        """
        # Base demand
        base = float(row["base_demand"])
    
        # Context features (use .get to be safe if missing)
        promo_frac = float(row.get("promo_frac", 0.0))
        footfall   = float(row.get("footfall", 0.0))
        is_holiday = float(row.get("is_holiday", 0.0))
    
        mu = base
    
        # Promotional lift: if many items are on promo, demand rises
        mu *= (1.0 + 0.5 * promo_frac)
    
        # Holiday lift
        if is_holiday > 0.5:
            mu *= 1.2
    
        # Footfall scaling (relative to mean)
        if "footfall" in self.df.columns and self.df["footfall"].notna().any():
            footfall_mean = float(self.df["footfall"].mean())
            if footfall_mean > 0:
                # keep factor around ~1; avoid going crazy when footfall is large
                mu *= (0.5 + 0.5 * (footfall / footfall_mean))
    
        # Price elasticity: higher price → lower demand
        mu *= np.exp(-self.elasticity * (price_multiplier - 1.0))
    
        # Numeric safety
        mu = max(mu, 1e-3)
    
        # Poisson noise for stochastic demand
        return int(self.rng.poisson(mu))


    # ---------- Gym API ---------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_day = 0

        # Initial inventory: 1 day of safety stock based on first day's demand
        first_bd = float(self.df.iloc[0]["base_demand"])
        init_inv = int(self.safety_stock_factor * first_bd)
        init_inv = min(init_inv, self.max_inventory)

        # inventory_buckets[0] is freshest, [-1] is oldest (about to expire)
        self.inventory_buckets = [0] * self.shelf_life
        self.inventory_buckets[0] = init_inv

        self.last_action = 2  # index 2 = "no discount"
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        row = self.df.iloc[self.current_day]
        base_demand = float(row["base_demand"])

        # ---- 1) Items that were oldest yesterday expire today ---- #
        expired = self.inventory_buckets[-1]

        # Age inventory: shift buckets so everything gets 1 day older
        new_buckets = [0] * self.shelf_life
        for i in range(1, self.shelf_life):
            new_buckets[i] = self.inventory_buckets[i - 1]
        self.inventory_buckets = new_buckets

        # ---- 2) Replenish today based on baseline demand ---- #
        order_qty = int(self.safety_stock_factor * base_demand)
        order_qty = max(0, min(order_qty, self.max_inventory))
        self.inventory_buckets[0] += order_qty

        # ---- 3) Demand given chosen price ---- #
        price_multiplier = float(self.price_multipliers[action])
        effective_price = self.base_price * price_multiplier

        # ❗ CHANGE 1: pass the whole row into _simulate_demand
        demand = self._simulate_demand(row, price_multiplier)

        # ---- 4) Fulfil demand from OLDEST stock first (FEFO) ---- #
        remaining_demand = demand
        sold = 0

        for idx in reversed(range(self.shelf_life)):  # oldest to youngest
            available = self.inventory_buckets[idx]
            if available <= 0:
                continue

            take = min(available, remaining_demand)
            self.inventory_buckets[idx] -= take
            remaining_demand -= take
            sold += take

            if remaining_demand <= 0:
                break

        total_inv = sum(self.inventory_buckets)
        waste_qty = expired

        # ---- 5) Reward = revenue - waste_penalty * waste ---- #
        revenue = sold * effective_price
        reward = revenue - self.waste_penalty * waste_qty

        # ❗ CHANGE 2: move time forward + terminal leftover penalty
        self.last_action = action
        self.current_day += 1

        # End of episode?
        if self.current_day >= self.n_days:
            done = True

            # Penalise leftover inventory at the end of the horizon
            leftover_inventory = sum(self.inventory_buckets)
            if leftover_inventory > 0:
                reward -= self.waste_penalty * leftover_inventory
        else:
            done = False

        obs = self._get_obs()
        info = {
            "date": row["date"],
            "revenue": revenue,
            "sold": sold,
            "waste": waste_qty,
            "inventory": total_inv,
            "demand": demand,
            "effective_price": effective_price,
            "base_demand": base_demand,
            "order_qty": order_qty,
        }

        # Gymnasium API: obs, reward, terminated, truncated, info
        return obs, reward, done, False, info


    def render(self):
        row = self.df.iloc[self.current_day]
        print(
            f"Day {self.current_day} | "
            f"date={row['date']} | "
            f"base_demand={row['base_demand']:.1f} | "
            f"inventory={sum(self.inventory_buckets)} | "
            f"last_action={self.last_action}"
        )

