package fr.univlyon1.reward;

import fr.univlyon1.configurations.Configuration;

import java.util.HashMap;
import java.util.Map;


public class RewardSMDP implements RewardShaping{
    protected double gamma ;

    public RewardSMDP(Configuration conf){
        this.gamma = conf.getGamma();
    }


    @Override
    public Double constructReward(HashMap<Double, Double> rewardTime, Double previousTime,Double time) {
        if(previousTime == null){
            return 0. ;
        }
        double reward = 0. ;
        double totalTime = time-previousTime ;
        for(Map.Entry<Double,Double> entry : rewardTime.entrySet()){
            reward += entry.getValue() ;
        }
        return reward ;
    }

    public RewardSMDP(double gamma){
        this.gamma = gamma ;
    }

}
