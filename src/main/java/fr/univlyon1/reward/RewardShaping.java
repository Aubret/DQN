package fr.univlyon1.reward;

import java.util.HashMap;

public interface RewardShaping {
    public Double constructReward(HashMap<Double,Double> rewardTime, double simulationTime);

    }
