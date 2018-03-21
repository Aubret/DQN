package fr.univlyon1.actorcritic.policy;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

public class EgreedyDecrement implements Policy {
    private double minEpsilon;
    private Random random ;
    private int numberStep ;
    private int schedule ;
    private double epsilon ;

    private int stepEpsilons;

    public EgreedyDecrement(double minEpsilon, int schedule,long seed) {
        this.minEpsilon = epsilon;
        this.random = new Random(seed);
        this.numberStep = 0 ;
        this.stepEpsilons = 1 ;
        this.schedule = schedule ;
        this.epsilon =1 ;
    }

    @Override
    public Integer getAction(INDArray results) {
        this.modifyEpsilon();
        int indice ;
        if(this.random.nextDouble() < this.epsilon) {
            indice = random.nextInt(results.size(1));
        }else {
            indice = Nd4j.argMax(results).getInt(0);
        }
        return indice ;
    }

    private void modifyEpsilon(){
        this.numberStep ++ ;
        if(this.numberStep % this.schedule == 0){
            this.numberStep = 0 ;
            this.stepEpsilons++ ; // Augmente de 1 tous les schedule iterations
            this.epsilon = Math.min(1f, Math.max(this.minEpsilon, 1. / this.stepEpsilons )) ;
        }
    }
}
