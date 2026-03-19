# library imports 

from __future__ import annotations
from dataclasses import dataclass, field
from tokenize import group
from turtle import pd
from typing import Dict, List, Optional
import itertools
import random
import statistics
import pandas


# -------------------
# CONFIGURATION class
# All tuneable simulation parameters are in this single dataclass. Changing
# values here will alter the behaviour of the entire simulation.

@dataclass
class run_config:
    """Configuration parameters for the Antarctic station simulation"""

    #Simulation scale parameters   
    pop_size: int = 20                  # total number of agents (crew members) in the station
    team_size: int = 4                  # number of agents per team/group (pop_size should be divisible)
    n_days: int = 30                    # how many days to simulate
    rng_seed: int = 42                  # seed for the random-number generator (for reproducibility)

    # Allowed activity and season labels 
    ACTIVITIES = ("sleep", "individual", "team", "exercise", "leisure", "food", "free")       # the set of recognised activity types an agent can perform
    SEASONS = ("summer", "winter", "other")   # the set of recognised Antarctic seasons

    #Environmental / station parameters 
    
    station_net_volume: float = 5000.0          # usable interior volume of the station in cubic metres
    temperature: float = -20.0                  # outside air temperature in degrees Celsius    
    altitude: float = 300.0                     # station altitude in metres above sea level  
    pressure: float = 960.0                      # atmospheric pressure in hPa (hectopascals)   
    wind_speed: float = 12.0                   # average wind speed in metres per second  
    sunlight_hours: float = 6.0                  # hours of natural sunlight per day   
    sleep_start_time: int = 22                  # hour of day (0-23) when the sleep block begins
    season: str = "winter"                      # current Antarctic season ("winter", "summer", or "other")


    # prob_env_accident_by_season : probability (per agent per hour) of an
    #   environmental accident occurring, keyed by season name.
    #   Winter is most dangerous; summer is safest.
    prob_env_accident_by_season: Dict[str, float] = field(default_factory=lambda: {
        "winter": 0.02, 
        "summer": 0.005, 
        "other": 0.01
    })

    env_accident_impact_on_stress: float = 0.25                         # flat stress increase when an accident hits


    # Task / activity parameters 

    # per-hour probability that a work task fails, by type
    prob_task_failure: Dict[str, float] = field(default_factory=lambda: {
        "individual": 0.05, 
        "team": 0.08
    })

    # base difficulty of each work-task type (0.0 – 1.0 scale)
    task_difficulty: Dict[str, float] = field(default_factory=lambda: {
        "individual": 0.1, 
        "team": 0.15
    })

    # Stress and health impact maps 
    # change applied to an agent's stress each hour they perform the given activity.  
    # Negative values REDUCE stress.

    activity_stress_impact = {
        "sleep": -0.1,       # sleeping relieves stress
        "individual": 0.03,  # solo work adds a small amount of stress
        "team": 0.02,        # team work adds a small amount of stress
        "exercise": -0.2,    # exercise is the best stress reliever
        "leisure": -0.1,     # leisure time reduces stress
        "food": 0.0,         # meals have no direct stress effect
        "free": -0.005,      # free time provides minimal stress relief
    }

    # change applied to an agent's physical health each hour they perform the given activity.  
    # Positive values IMPROVE health.
    activity_health_impact = {
        "sleep": 0.5,        # sleep is the strongest health restorer
        "individual": 0,     # solo work has no health effect
        "team": 0,           # team work has no health effect
        "exercise": 0.3,     # exercise boosts physical health
        "leisure": 0.1,      # leisure provides a small health boost
        "food": 0.2,         # eating supports physical health
        "free": 0.0,         # free time has no health effect
    }

    # additional stress penalty when a task fails
    taskfailure_stress_impact = {
        "individual": 0.05,  # solo failure 
        "team": 0.03,        # team failure 
    }

    # ---- Personality-based team dynamics ------------------------------------

    # modifier to team-task difficulty based on the personality-type pairing of two team members.
    # Positive : pair increases difficulty (friction).
    # Negative : pair decreases difficulty (synergy).
    # Zero     : neutral pairing.
    ptype_dyadic_interactions = {
        ("dominant", "dominant"): 0.00,         # neutral
        ("dominant", "conscientious"): -1,      # good synergy
        ("dominant", "steady"): 1,              # friction 
        ("dominant", "influential"): 0.0,       # neutral
        ("conscientious", "conscientious"): 0.00, # neutral
        ("conscientious", "steady"): 0.00,      # neutral
        ("conscientious", "influential"): 1,    # friction 
        ("steady", "steady"): 0.00,             # neutral
        ("steady", "influential"): -1,          # good synergy
        ("influential", "influential"): 0,      # neutral
    }

    # scaling weight that controls how strongly the aggregate personality-pair score affects team-task difficulty
    aggptype_diff_parameter = 0.05

    # health penalty applied each hour when an agent's stress exceeds the stress_health_threshold
    stress_on_health_impact = 0.05

    # stress level above which physical health starts to erode
    stress_health_threshold = 0.8  

# Instantiate a single global config object used throughout the model
config = run_config()





# ---------------------
# ENVIRONMENT class
# Models the Antarctic station's physical environment.  Tracks the current
# hour-of-day, provides a composite "harshness" score, and determines whether environmenjtal accidents occur.

@dataclass
class Environment:
    """Represents the Antarctic station environment and its conditions"""

    # Physical and atmospheric properties of the station / surroundings
    station_net_volume: float           # usable interior volume (m³)
    temperature: float               # outside temperature (°C)
    altitude: float                    # station altitude (m above sea level)
    pressure: float                   # atmospheric pressure (hPa)
    wind_speed: float                # average wind speed (m/s)
    sunlight_hours: float             # hours of daylight per day
    sleep_start_time: int            # default bedtime hour (0-23)
    prob_env_accident_by_season: Dict[str, float]  # accident probability map
    team_size: int                   # number of agents per group 
    season: str = "winter"            # current season label
    hour: int = 0                    # current simulation hour (0-23, wraps)

    # Advance the clock by one hour, wrap at 24 back to 0
    def tick_time(self) -> None:
        self.hour = (self.hour + 1) % 24

    # Roll the dice to decide if an environmental accident happens this hour.
    # and returns True if the random draw falls below the season's accident probability.
    def accident_occurs(self, rng: random.Random) -> bool:
        return rng.random() < self.prob_env_accident_by_season.get(self.season, 0.0)

    # Compute a composite harshness score (0.0 = benign)
    # Four sub-factors are summed:
    def env_harshness(self) -> float:                               
        cold_load = max(0.0, (0 - self.temperature) / 60.0)          # how far below 0 °C the temperature is (temporary normalisation values)
        wind_load = min(1.0, self.wind_speed / 25.0)                 # wind speed relative to 25 m/s cap  (temporary normalisation values)
        pressure_load = max(0.0, (1013 - self.pressure) / 400.0)    # how far below sea-level pressure (1013 hPa) (temporary normaliration values)
        light_load = max(0.0, (12 - self.sunlight_hours) / 12.0)      # how far below 12 h of daylight we are 
        return cold_load + wind_load + pressure_load + light_load





# ---------------------------
# ACTIVITY class
# task-failure probabilities and difficulty calculations for the activities agents can perform.
@dataclass
class Activity:
    """Represents activities agents perform and their characteristics"""

    name: str                              # descriptive label for this activity set
    prob_task_failure: Dict[str, float]      # per-activity failure probabilities
    task_difficulty: Dict[str, float]       # per-activity base difficulty values

    # Determine whether a task failure happens this hour for the given activity.
    # Only "individual" and "team" activities have non-zero failure probabilities.
    def task_failure_occurs(self, activity: str, rng: random.Random) -> bool:
        return rng.random() < self.prob_task_failure.get(activity, 0.0)

    # Calculate the effective difficulty of a task this hour.
    # - Starts with the base difficulty for the activity type..
    # - Adds the environment's harshness score (weather, light, etc.).
    # - For team tasks, also factors in personality-pair dynamics within the group, scaled by aggptype_diff_parameter.
    # - Non-work activitieis (sleep, food, exercise, etc.) return 0.0.
    def difficulty(self, activity: str, env: Environment, group: Group) -> float:
        """Calculate task difficulty based on activity type, environment, and group dynamics"""

        # If the activity is not recognised, difficulty is zero
        if activity not in config.ACTIVITIES:
            return 0.0

        # Base difficulty = configured task difficulty + environmental harshness
        base = self.task_difficulty.get(activity, 0.0) + env.env_harshness()

        # Individual tasks depend only on the base difficulty
        if activity == "individual":
            return base

        # Team tasks add a personality-interaction modifier, capped at 1.0
        if activity == "team":
            return min(1.0, base + group.team_interaction_difficulty() * config.aggptype_diff_parameter)

        # All other activities (sleep, food, etc.) have zero difficulty
        return 0.0


# ----------------------------------
# AGENT class
# Represents one crew member and holds personal state (stress, health, personality, schedule) 
# and contains the core per-hour simulation logic.

@dataclass
class Agent:
    """Represents an individual stationed in Antarctica with personality and health attributes"""

    agent_id: int                    # unique identifier for this agent
    group_id: int                    # which team/group this agent belongs to
    personality_type: str               # one of: dominant, conscientious, steady, influential
    stress: float = 0.0                 # current stress level (0.0 = none)
    physical_health: float = 1.0        # current health (1.0 = perfect)
    sleep_duration: int = 7            # assigned sleep hours per night (6-8)
    schedule: List[str] = field(default_factory=lambda: ["free"] * 24)      # 24-slot activity plan
    conflicts: int = 0               # running count of conflicts (for future use)

    # Look up the scheduled activity for the current hour of the day.
    # "free" if the schedule contains an unrecognised activity.
    def current_activity(self, env: Environment) -> str:
        act = self.schedule[env.hour % 24]
        return act if act in config.ACTIVITIES else "free"

    # Core per-hour simulation step for this agent.
    # Determines what the agent is doing, applies the corresponding stress
    # and health changes, checks for task failures and environmental accidents, then updates the agent's state.
    def perform_hour(
        self,
        env: Environment,
        group: "Group",
        activity: Activity,
        peers: List["Agent"],
        rng: random.Random) -> None:

        # Identify what the agent is scheduled to do this hour
        hour_activity = self.current_activity(env)

        # Accumulates stress changes this hour 
        stress_delta = 0.0

        #Branches on activity type and apply effects 

        if hour_activity == "sleep":
            # Sleeping reduces stress and restores health
            self.stress = max(0.0, self.stress + config.activity_stress_impact["sleep"])
            self.physical_health = min(1.0, self.physical_health + config.activity_health_impact["sleep"])

        elif hour_activity == "exercise":
            # Exercise also reduces stress and boosts health
            self.stress = max(0.0, self.stress + config.activity_stress_impact["exercise"])
            self.physical_health = min(1.0, self.physical_health + config.activity_health_impact["exercise"])

        elif hour_activity == "food":
            # Eating restores health but does not affect stress
            self.physical_health = min(1.0, self.physical_health + config.activity_health_impact["food"])

        elif hour_activity == "leisure":
            # Leisure time reduces stress (no direct health change)
            self.stress = max(0.0, self.stress + config.activity_stress_impact["leisure"])

        elif hour_activity == "individual":
            # Individual work: difficulty adds to stress delta
            stress_delta += activity.difficulty(hour_activity, env, group)

            # Check for task failure: if the tassk fails, extra stress is added
            if activity.task_failure_occurs(hour_activity, rng):
                stress_delta += config.taskfailure_stress_impact["individual"]

        elif hour_activity == "team":
            # Team work: difficulty (including personality dynamics) adds to stress delta
            stress_delta += activity.difficulty(hour_activity, env, group)

            # Check for task failure: team failures add shared stress
            if activity.task_failure_occurs(hour_activity, rng):
                stress_delta += config.taskfailure_stress_impact["team"]

        else:
            # "free" time or any unrecognised activity: apply the "free" stress impact
            self.stress = max(0.0, self.stress + config.activity_stress_impact["free"])

        # Environmental accident check applies on top of any activity
        # Each hour there is a season-dependent chance of an accident
        # and If one happens, all agents suffer a stress impact and a small health penaltuy.
        if env.accident_occurs(rng):
            stress_delta += config.env_accident_impact_on_stress 
            self.physical_health = max(0.0, self.physical_health - 0.05)

        # Apply the accumulated stress change (stress cannot drop below 0)
        self.stress = max(0.0, self.stress + stress_delta)

        # If stress is high, it starts eroding physical health
        self._apply_health_from_stress()

    # When stress exceeds the stress_health_threshold, the agent loses a fixed amount of physical health each hour.  
    def _apply_health_from_stress(self) -> None:
        if self.stress > config.stress_health_threshold:
            self.physical_health = max(0.0, self.physical_health - config.stress_on_health_impact)


# ---------------
# POPULATION CLASS
# class responsible for creating all agents and assigning them
# randomised attributes and daily schedules.
@dataclass
class population:
    """Manages creation and initialization of the agent population"""

    size: int                                                   # total number of agents to create
    team_size: int                                               # agents per team (used to assign group_id)
    individuals: list[Agent] = field(default_factory=list)      # populated by generate()
    rng: random.Random = field(default_factory=random.Random)  # RNG for schedule variance

    # Create all agents with randomised personality types, initial stress,
    # health, sleep durations, and schedules.  Each agent is assigned a
    # group_id based on integer division of their agent_id by team_size,
    # so agents 0-3 are group 0, agents 4-7 are group 1, etc.
    def generate(self) -> None:
        for i in range(self.size):
            # The four personality archetypes available for random assignment (based on DISK model)
            personality_types = ["dominant","conscientious","steady","influential"]

            # Build a structured daily schedule (see _make_schedule below)
            schedule = self._make_schedule()

            # Create the agent with randomised starting conditions
            agent = Agent(
                agent_id=i,
                group_id=i // self.team_size,       # integer division assigns groups
                personality_type= random.choice(personality_types),  # random personality
                stress=random.uniform(0.0, 0.2),    # small random starting stress
                physical_health=1.0,                 # everyone starts fully healthy
                sleep_duration=random.randint(6, 8), # 6-8 hours of sleep
                schedule=schedule,
            )
            self.individuals.append(agent)

    # Build a 24-slot daily schedule for one agent.
    # The schedule follows a structured Antarctic-station routine

    def _make_schedule(self) -> List[str]:
        # Start with every hour set to "free"
        sched = ["free"] * 24

        # Sleep block
        # Begins at hour 22 and lasts 6-8 hours (wraps past midnight)
        sleep_start = 22 
        for h in range(self.rng.randint(6, 8)):
            sched[(sleep_start + h) % 24] = "sleep"

        # Meals 
        # Only placed if the slot is still "free" (it won't overwrite sleep)
        for h in (7, 12, 18):
            if sched[h] == "free":
                sched[h] = "food"

        #Team work blocks
        for h in (8, 9, 10, 11, 16):
            if sched[h] == "free":
                sched[h] = "team"

        # Individual work block
        for h in (12,13,14,15):
            if sched[h] == "free":
                sched[h] = "individual"

        # Exercise
        # Scheduled at 17:00 
        if sched[17] == "free":
            sched[17] = "exercise"

        # Leisure
        # Evening wind-down: 19-21
        for h in (19,20,21):
            if sched[h] == "free":
                sched[h] = "leisure"

        return sched


# ----------------------
# GROUP Class
#Represents a team of agents.  
# The main purpose is to calculate the aggregate personality-pair interaction score that modifies team-taskdifficulty.

@dataclass
class Group:
    """Represents a team of agents and their interactions"""

    group_id: int                                                # unique group identifier
    members: list[Agent] = field(default_factory=list)               # agents in this group

    conflicts: dict[tuple[int, int], int] = field(default_factory=dict)   # tracks pairwise conflict counts (for future use)
    interactions: Dict[tuple[int, int], int] = field(default_factory=dict)   #  tracks pairwise interaction counts (for future use)

    # Calculate the average personality-dyadic interaction score for all unique pairs of members in this group.
    # Steps:
    # 1. Enumerate every unique pair of group members.
    # 2. Look up their personality-type pair in the dyadic interaction table.(Order-independent)
    # 3. Sum all matching interaction valuess and divide by the number of pairs to get a mean score.
    # A positive score means the group's personality mix increases task difficulty;
    # a negative score means it decreases difficulty (better teamwork).

    def team_interaction_difficulty(self) -> float:
        # Empty group: no interaction difficulty
        if not self.members:
            return 0.0

        interaction_sum = 0.0  # running totall of dyadic interaction values
        count = 0              # number of matched pairs

        # Iterate over every unique pair of group members
        for pair in itertools.combinations(self.members, 2):
            member1, member2 = pair

            # Search the dyadic interaction table for a matching personality pair
            for (ptype1, ptype2), value in config.ptype_dyadic_interactions.items():
                # Match regardless of ordering (dominant-conscientious is the same as conscientious-dominant)
                if (ptype1 == member1.personality_type and ptype2 == member2.personality_type) or \
                   (ptype2 == member1.personality_type and ptype1 == member2.personality_type):
                    interaction_sum += value
                    count += 1

        # Return the mean interaction score, or 0 if no pairs matched
        return (interaction_sum / count) if count > 0 else 0.0


# ----------------------
# SIMULATION class. 
# Ties togethert the Environment, Population, Groups, and Activity objects, runs the hour-by-hour simulation loop, collects
# per-hour and per-day logs, and exports results to CSV.

@dataclass
class Simulation:
    """Main simulation controller for the Antarctic station model"""

    # Constructor parameters (provided at instantiation) ---
    pop_size: int              # total number of agents
    team_size: int             # agents per team
    n_days: int                # number of days to simulate
    rng_seed: int = 42         # random seed for reproducibility

    # Internal state (created during setup, not passed in) 
    population: population = field(init=False)     # population factory (created in setup)
    activity: Activity = field(init=False)          # shared activity calculator
    env: Environment | None = None                  # reference to the environment
    agents: List[Agent] = field(default_factory=list)       # flat list of all agents
    groups: Dict[int, Group] = field(default_factory=dict)  # group_id → Group mapping
    rng: random.Random = field(default_factory=random.Random)  # master RNG
    hour_log: List[Dict] = field(default_factory=list)      # per-agent-per-hour records
    daily_log: List[Dict] = field(default_factory=list)     # per-day summary records

    # Preparing all components before running the simulation.
    #1. Seed the RNG for reproducibility.
    #2. Store a reference to the provided Environment.
    #3. Generate the agent population.
    #4. Organise agents into Group objects by their group_id.
    #5. Create the shared Activity object with configured failure/difficulty rates.

    def setup(self, env: Environment) -> None:
        # Seed the random number generator (for reproduciblity)
        self.rng = random.Random(self.rng_seed)

        # set the environment
        self.env = env

        # Create and populate the agent pool
        self.population = population(size=self.pop_size, team_size=self.team_size)
        self.population.generate()
        self.agents = self.population.individuals

        # Build group objects by grouping agents that share a group_id
        self.groups = {}
        self.activity = Activity(
            name="Base Activity",
            prob_task_failure=config.prob_task_failure,
            task_difficulty=config.task_difficulty)

        # Sort agents by group_id, then use itertools.groupby to cluster them
        for gid, members in itertools.groupby(sorted(self.agents, key=lambda a: a.group_id), key=lambda a: a.group_id):
            self.groups[gid] = Group(group_id=gid, members=list(members))

    # Execute the main simulation loop.
    # Iterates over every hour of every day.  For each hour:
    # - Each agent performs their scheduled activity (updating stress/health).
    # - The hour's data is logged.
    # - At the end of each 24-hour cycle, a daily assessment is performed.
    # - The environment clock advances by one hour.
    def run(self) -> None:
        # Total number of hourly ticks to simulate
        total_hours = self.n_days * 24

        for t in range(total_hours):
            # Process every agent for this hour 
            for agent in self.agents:
                # Find the agent's group for team-dynamics calculations
                group = self.groups[agent.group_id]

                # Build a list of all other agents (peers): currently unused but available for future social mechanics
                peers = [a for a in self.agents if a.agent_id != agent.agent_id]

                # Snapshot the agent's stress before this hour (for delta logging)
                before_stress = agent.stress

                # Run the agent's hourly simulation step
                agent.perform_hour(self.env, group=group, activity=self.activity, peers=peers, rng=self.rng)

                # Record this hour's outcome for the agent
                self.hour_log.append({
                    "hour": self.env.hour,            # current hour of day (0-23)
                    "agent": agent.agent_id,          # which agent
                    "group": agent.group_id,          # which group they belong to
                    "activity": agent.current_activity(self.env),  # what they did
                    "stress": agent.stress,           # stress level after this hour
                    "stress_delta": agent.stress - before_stress,  # change this hour
                    "health": agent.physical_health,  # health level after this hour
                })

            # Daily assessment: runs once every 24 hours
            if (t + 1) % 24 == 0:                           # (t+1) % 24 == 0 is true at the end of hour 23 each day
                self._daily_assessment(day=(t + 1) // 24)

            # Advance the environment clock by one hour
            self.env.tick_time()

    # Compute end-of-day statistics for each group and for the whole population, then export the results to CSV files.
    def _daily_assessment(self, day: int) -> None:
        """Calculate and log daily statistics for agents and groups"""
        group_stats = {}

        # Per-group statistics 
        for gid, group in self.groups.items():
            avg_group_stress = statistics.mean(a.stress for a in group.members)               # Mean stress across all members of this group
            avg_group_health = statistics.mean(a.physical_health for a in group.members)        # Mean physical health across all members of this group
            group_ptypes = [a.personality_type for a in group.members]                        # Collect personality types of group members for reference

            group_stats[gid] = {
                "avg_group_stress": round(avg_group_stress, 3),
                "avg_group_health": round(avg_group_health, 3),
                "group_personality_types": group_ptypes,
            }

        # Convert group stats dict into a DataFrame (groups as rows)
        group_stats_df = pandas.DataFrame.from_dict(group_stats, orient='index')

        # Population-wide statistics
        # Average stress and health across all agents
        avg_stress = statistics.mean(a.stress for a in self.agents)
        avg_health = statistics.mean(a.physical_health for a in self.agents)

        # Append this day's summary to the running daily log
        self.daily_log.append({
            "day": day,
            "avg_stress": round(avg_stress, 3),
            "avg_health": round(avg_health, 3),
            "group_stats": group_stats,
        })

        # Build a DataFrame of all daily summaries so far
        daily_summary = pandas.DataFrame(self.daily_log)
        
        #Export results to CSV 
        # daily_summary.csv : one row per day with population averages
        daily_summary.to_csv("daily_summary.csv", index=False)
        # group_stats.csv   : one row per group with group-level averages
        group_stats_df.to_csv("group_stats.csv", index=True)

        # Return both DataFrames 
        return daily_summary, group_stats_df


# ------------------------------
# MAIN
# 1. Creates an Environment from the global config.
# 2. Creates and sets up a Simulation.
# 3. Runs the simulation for the configured number of days.
# 4. Performs a final daily assessment and exports CSV results.

if __name__ == "__main__":

    # Build the Antarctic environment from config values
    env = Environment(
        station_net_volume=config.station_net_volume,
        temperature=config.temperature,
        altitude=config.altitude,
        pressure=config.pressure,
        wind_speed=config.wind_speed,
        sunlight_hours=config.sunlight_hours,
        sleep_start_time=config.sleep_start_time,
        prob_env_accident_by_season=config.prob_env_accident_by_season,
        team_size=config.team_size,
        season=config.season,
    )

    # Create the simulation with population and timing settings
    sim = Simulation(
        pop_size=config.pop_size,
        team_size=config.team_size,
        n_days=config.n_days,
        rng_seed=config.rng_seed
    )

    # Initialise internal structures (agents, groups, activity)
    sim.setup(env)

    # Run the full simulation (n_days × 24 hours) 
    sim.run()

    #Final export: re-run assessment for the last day to capture CSVs
    daily_summary, group_stats_df = sim._daily_assessment(day=sim.n_days)
