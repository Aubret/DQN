package fr.univlyon1.reward;

import fr.univlyon1.configurations.Configuration;

import java.util.HashMap;
import java.util.Map;

public class NstepTime implements RewardShaping {
    public double gamma ;
    public NstepTime(Configuration conf){
        this.gamma = conf.getGamma();
    }

    public Double constructReward(HashMap<Double,Double> rewardTime, Double simulationTime){
        if(simulationTime == null){
            return 0. ;
        }
        double reward = 0. ;
        for(Map.Entry<Double,Double> entry : rewardTime.entrySet()){
            double t = entry.getKey()-simulationTime ;
            reward += Math.pow(this.gamma,t)*entry.getValue() ;
        }
        System.out.println(reward);
        return reward ;
    }
}
