package fr.univlyon1.reward;

import fr.univlyon1.configurations.Configuration;

import java.util.HashMap;
import java.util.Map;

/**
 * Compute the reward as sum(lambda^dt * r)
 */
public class NstepTime implements RewardShaping {
    protected double gamma ;
    public NstepTime(Configuration conf){
        this.gamma = conf.getGamma();
    }

    public Double constructReward(HashMap<Double,Double> rewardTime, Double previousTime,Double time){
        if(previousTime == null){
            return 0. ;
        }
        double reward = 0. ;
        for(Map.Entry<Double,Double> entry : rewardTime.entrySet()){
            double t = entry.getKey()-previousTime ;
            reward += Math.pow(this.gamma,t)*entry.getValue() ;
        }
        //System.out.println(reward);
        //reward += Math.pow(this.gamma,time-previousTime) ;
        return reward ;
    }
}
