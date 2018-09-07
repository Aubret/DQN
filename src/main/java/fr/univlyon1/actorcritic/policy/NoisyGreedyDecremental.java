package fr.univlyon1.actorcritic.policy;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;

public class NoisyGreedyDecremental extends NoisyGreedy{
    private double minStd;
    private Random random ;
    private int numberStep ;
    private int schedule ;

    private int stepStd;

    public NoisyGreedyDecremental(double std, double mean, int stdInit , int schedule,long seed, Policy greedy) {
        super(std,mean,seed,greedy);
        this.minStd = std;
        this.random = new Random(seed);
        this.numberStep = 1 ;
        this.stepStd = stdInit ;
        this.schedule = schedule ;
        this.std = Math.min(1.f, Math.max(this.minStd, 0.5-0.05*this.stepStd));
    }
    @Override
    public INDArray getAction(INDArray results) {
        this.modifyStd();
        return super.getAction(results);
    }

    private void modifyStd(){
        this.numberStep ++ ;
        if(this.numberStep % this.schedule == 0){
            this.numberStep = 0 ;
            this.stepStd++ ; // Augmente de 1 tous les schedule iterations
            this.std = Math.min(1.f, Math.max( this.minStd, 0.5-0.05*this.stepStd)) ;
            System.out.println("new std :"+this.std);
        }
    }
}
