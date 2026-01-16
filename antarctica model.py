from __future__ import annotations
from dataclasses import dataclass, field
from tokenize import group
from turtle import pd
from typing import Dict, List, Optional
import itertools
import random
import statistics
import pandas


@dataclass
class run_config:
    """Configuration parameters for the Antarctic station simulation"""
    # Simulation parameters
    pop_size: int = 20
    team_size: int = 4
    n_days: int = 30
    rng_seed: int = 42

    # Define available activity types and seasons
    ACTIVITIES = ("sleep", "individual", "team", "exercise", "leisure", "food", "free")
    SEASONS = ("summer", "winter", "shoulder")

    # Environment parameters
    station_net_volume: float = 5000.0
    temperature: float = -20.0
    altitude: float = 300.0
    pressure: float = 960.0
    wind_speed: float = 12.0
    sunlight_hours: float = 6.0
    sleep_start_time: int = 22
    season: str = "winter"
    prob_env_accident_by_season: Dict[str, float] = field(default_factory=lambda: {
        "winter": 0.02, 
        "summer": 0.005, 
        "shoulder": 0.01
    })

    env_accident_impact_on_stress: float = 0.25

    # Activity parameters
    prob_task_failure: Dict[str, float] = field(default_factory=lambda: {
        "individual": 0.05, 
        "team": 0.08
    })
    task_difficulty: Dict[str, float] = field(default_factory=lambda: {
        "individual": 0.1, 
        "team": 0.15
    })



    # Impact of each activity on stress level (negative values reduce stress)
    activity_stress_impact = {
        "sleep": -0.1,
        "individual": 0.03,
        "team": 0.02,
        "exercise": -0.2,
        "leisure": -0.1,
        "food": 0.0,
        "free": -0.005,
    }

    # Impact of each activity on physical health (positive values improve health)
    activity_health_impact = {
        "sleep": 0.5,
        "individual": 0,
        "team": 0,
        "exercise": 0.3,
        "leisure": 0.1,
        "food": 0.2,
        "free": 0.0,
    }

    # Additional stress when task failure occurs
    taskfailure_stress_impact = {
        "individual": 0.05,
        "team": 0.03,       
    }

    # Personality type dyadic interaction effects on team task difficulty
    # Positive values indicate increased difficulty (more stress), negative values indicate decreased difficulty (less stress)
    ptype_dyadic_interactions = {
        ("dominant", "dominant"): 0.00,
        ("dominant", "conscientious"): -1,
        ("dominant", "steady"): 1,
        ("dominant", "influential"): 0.0,
        ("conscientious", "conscientious"): 0.00,
        ("conscientious", "steady"): 0.00,
        ("conscientious", "influential"): 1,
        ("steady", "steady"): 0.00,
        ("steady", "influential"): -1,
        ("influential", "influential"): 0, 
    }

    # Weight for personality type interaction effects on team difficulty
    aggptype_diff_parameter = 0.05

    # Health penalty when stress exceeds threshold
    stress_on_health_impact = 0.05



# Create global config instance
config = run_config()


@dataclass
class Environment:
    """Represents the Antarctic station environment and its conditions"""
    station_net_volume: float
    temperature: float
    altitude: float
    pressure: float
    wind_speed: float
    sunlight_hours: float
    sleep_start_time: int
    prob_env_accident_by_season: Dict[str, float]
    team_size: int
    season: str = "winter"
    hour: int = 0

    # Advance time by one hour
    def tick_time(self) -> None:
        self.hour = (self.hour + 1) % 24

    # Check if an environmental accident occurs this hour
    def accident_occurs(self, rng: random.Random) -> bool:
        return rng.random() < self.prob_env_accident_by_season.get(self.season, 0.0)

    # Calculate overall environmental harshness from multiple factors
    def env_harshness(self) -> float:
        cold_load = max(0.0, (0 - self.temperature) / 49.0)
        wind_load = min(1.0, self.wind_speed / 25.0)
        pressure_load = max(0.0, (1013 - self.pressure) / 400.0)
        light_load = max(0.0, (12 - self.sunlight_hours) / 12.0)
        return cold_load + wind_load + pressure_load + light_load

@dataclass
class Activity:
    """Represents activities agents perform and their characteristics"""
    name: str
    #task_location: Dict[str, str]  # "indoor" or "outdoor"
    prob_task_failure: Dict[str, float]
    task_difficulty: Dict[str, float]

    # Check if a task failure occurs for the given activity
    def task_failure_occurs(self, activity: str, rng: random.Random) -> bool:
        return rng.random() < self.prob_task_failure.get(activity, 0.0)

    # Calculate task difficulty based on activity type, environment, and group dynamics
    def difficulty(self, activity: str, env: Environment, group: Group) -> float:
        """Calculate task difficulty based on activity type, environment, and group dynamics"""
        if activity not in config.ACTIVITIES:
            return 0.0
        base = self.task_difficulty.get(activity, 0.0) + env.env_harshness()
        if activity == "individual":
            return base
        if activity == "team":
            return min(1.0, base + group.team_interaction_difficulty() * config.aggptype_diff_parameter)
    
        return 0.0



@dataclass
class Agent:
    """Represents an individual stationed in Antarctica with personality and health attributes"""
    agent_id: int
    group_id: int
    personality_type: str
    stress: float = 0.0
    physical_health: float = 1.0
    sleep_duration: int = 7
    schedule: List[str] = field(default_factory=lambda: ["free"] * 24)
    conflicts: int = 0

    # Get the scheduled activity for the current hour
    def current_activity(self, env: Environment) -> str:
        act = self.schedule[env.hour % 24]
        return act if act in config.ACTIVITIES else "free"

    # Perform one hour of activity and update stress and health
    def perform_hour(
        self,
        env: Environment,
        group: "Group",
        activity: Activity,
        peers: List["Agent"],
        rng: random.Random) -> None:
        hour_activity = self.current_activity(env)
        stress_delta = 0.0

        if hour_activity == "sleep":
            self.stress = max(0.0, self.stress + config.activity_stress_impact["sleep"])
            self.physical_health = min(1.0, self.physical_health + config.activity_health_impact["sleep"])
        elif hour_activity == "exercise":
            self.stress = max(0.0, self.stress + config.activity_stress_impact["exercise"])
            self.physical_health = min(1.0, self.physical_health + config.activity_health_impact["exercise"])
        elif hour_activity == "food":
            self.physical_health = min(1.0, self.physical_health + config.activity_health_impact["food"])
        elif hour_activity == "leisure":
            self.stress = max(0.0, self.stress + config.activity_stress_impact["leisure"])
        elif hour_activity == "individual":
            stress_delta += activity.difficulty(hour_activity, env, group)
            # Check for task failure
            if activity.task_failure_occurs(hour_activity, rng):
                stress_delta += config.taskfailure_stress_impact["individual"]
        elif hour_activity == "team":
            stress_delta += activity.difficulty(hour_activity, env, group)
            # Check for task failure
            if activity.task_failure_occurs(hour_activity, rng):
                stress_delta += config.taskfailure_stress_impact["team"]
        else:
            self.stress = max(0.0, self.stress + config.activity_stress_impact["free"])

        # Handle environmental accidents
        if env.accident_occurs(rng):
            stress_delta += config.env_accident_impact_on_stress 
            self.physical_health = max(0.0, self.physical_health - 0.05)

        self.stress = max(0.0, self.stress + stress_delta)
        self._apply_health_from_stress()

    # Apply health penalty when stress exceeds threshold
    def _apply_health_from_stress(self) -> None:
        if self.stress > 0.8:
            self.physical_health = max(0.0, self.physical_health - config.stress_on_health_impact)


@dataclass
class population:
    """Manages creation and initialization of the agent population"""
    size: int
    team_size: int
    individuals: list[Agent] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)

    # Generate the population of agents with random attributes
    def generate(self) -> None:
        for i in range(self.size):
            personality_types = ["dominant","conscientious","steady","influential"]
            schedule = self._make_schedule()
            agent = Agent(
                agent_id=i,
                group_id=i // self.team_size,
                personality_type= random.choice(personality_types),
                stress=random.uniform(0.0, 0.2),
                #physical_health=random.uniform(0.8, 1.0),
                physical_health=1.0,
                sleep_duration=random.randint(6, 8),
                schedule=schedule,
            )
            self.individuals.append(agent)

    # Create a daily activity schedule for the agent
    def _make_schedule(self) -> List[str]:
        sched = ["free"] * 24
        #sleep_start = 22 if self.env is None else self.env.sleep_start_time
        sleep_start = 22 
        for h in range(self.rng.randint(6, 8)):
            sched[(sleep_start + h) % 24] = "sleep"
        for h in (7, 12, 18):
            if sched[h] == "free":
                sched[h] = "food"
        for h in (8, 9, 10, 16, 17):
            if sched[h] == "free":
                sched[h] = "team"
        for h in (11,12,13,14,15):
            if sched[h] == "free":
                sched[h] = "individual"
        if sched[18] == "free":
            sched[18] = "exercise"
        for h in (19,20,21):
            if sched[h] == "free":
                sched[h] = "leisure"
        return sched

@dataclass
class Group:
    """Represents a team of agents and their interactions"""
    group_id: int
    members: list[Agent] = field(default_factory=list)
    conflicts: dict[tuple[int, int], int] = field(default_factory=dict)
    interactions: Dict[tuple[int, int], int] = field(default_factory=dict)

    # Calculate team task difficulty based on personality type dyadic interactions
    def team_interaction_difficulty(self) -> float:
        if not self.members:
            return 0.0
        interaction_sum = 0.0
        count = 0
        for pair in itertools.combinations(self.members, 2):
            member1, member2 = pair
            for (ptype1, ptype2), value in config.ptype_dyadic_interactions.items():
                if (ptype1 == member1.personality_type and ptype2 == member2.personality_type) or \
                   (ptype2 == member1.personality_type and ptype1 == member2.personality_type):
                    interaction_sum += value
                    count += 1
        return (interaction_sum / count) if count > 0 else 0.0



@dataclass
class Simulation:
    """Main simulation controller for the Antarctic station model"""
    pop_size: int
    team_size: int
    n_days: int
    rng_seed: int = 42
    population: population = field(init=False)
    activity: Activity = field(init=False)
    env: Environment | None = None
    agents: List[Agent] = field(default_factory=list)
    groups: Dict[int, Group] = field(default_factory=dict)
    rng: random.Random = field(default_factory=random.Random)
    hour_log: List[Dict] = field(default_factory=list)
    daily_log: List[Dict] = field(default_factory=list)

    # Initialize the simulation with population, groups, and activities
    def setup(self, env: Environment) -> None:
        self.rng = random.Random(self.rng_seed)
        self.env = env
        self.population = population(size=self.pop_size, team_size=self.team_size)
        self.population.generate()
        self.agents = self.population.individuals
        self.groups = {}
        self.activity = Activity(
            name="Base Activity",
            prob_task_failure=config.prob_task_failure,
            task_difficulty=config.task_difficulty)

        for gid, members in itertools.groupby(sorted(self.agents, key=lambda a: a.group_id), key=lambda a: a.group_id):
            self.groups[gid] = Group(group_id=gid, members=list(members))

    # Run the simulation for the specified number of days
    def run(self) -> None:
        total_hours = self.n_days * 24
        for t in range(total_hours):
            for agent in self.agents:
                group = self.groups[agent.group_id]
                peers = [a for a in self.agents if a.agent_id != agent.agent_id]
                before_stress = agent.stress
                agent.perform_hour(self.env, group=group, activity=self.activity, peers=peers, rng=self.rng)
                # Log hourly data for each agent
                self.hour_log.append({
                    "hour": self.env.hour,
                    "agent": agent.agent_id,
                    "group": agent.group_id,
                    "activity": agent.current_activity(self.env),
                    "stress": agent.stress,
                    "stress_delta": agent.stress - before_stress,
                    "health": agent.physical_health,
                })

            # Perform daily assessment at end of each day
            if (t + 1) % 24 == 0:
                self._daily_assessment(day=(t + 1) // 24)

            self.env.tick_time()

    # Calculate and log daily statistics for agents and groups
    def _daily_assessment(self, day: int) -> None:
        """Calculate and log daily statistics for agents and groups"""
        group_stats = {}

        # Calculate statistics for each group
        for gid, group in self.groups.items():
            avg_group_stress = statistics.mean(a.stress for a in group.members)
            avg_group_health = statistics.mean(a.physical_health for a in group.members)
            group_ptypes = [a.personality_type for a in group.members]
            group_stats[gid] = {
                "avg_group_stress": round(avg_group_stress, 3),
                "avg_group_health": round(avg_group_health, 3),
                "group_personality_types": group_ptypes,
            }

        group_stats_df = pandas.DataFrame.from_dict(group_stats, orient='index')

        # Calculate overall population statistics
        avg_stress = statistics.mean(a.stress for a in self.agents)
        avg_health = statistics.mean(a.physical_health for a in self.agents)
        self.daily_log.append({
            "day": day,
            "avg_stress": round(avg_stress, 3),
            "avg_health": round(avg_health, 3),
            "group_stats": group_stats,
        })

        daily_summary = pandas.DataFrame(self.daily_log)
        
        # Export data to CSV files
        daily_summary.to_csv("daily_summary.csv", index=False)
        group_stats_df.to_csv("group_stats.csv", index=True)

        return daily_summary,group_stats_df

# Run simulation with specified parameters
if __name__ == "__main__":
    # Configure Antarctic environment from config
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

    # Initialize and run simulation using config parameters
    sim = Simulation(
        pop_size=config.pop_size,
        team_size=config.team_size,
        n_days=config.n_days,
        rng_seed=config.rng_seed
    )
    sim.setup(env)
    sim.run()
    daily_summary, group_stats_df = sim._daily_assessment(day=sim.n_days)
    
