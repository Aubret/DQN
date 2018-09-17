package fr.univlyon1.actorcritic.policy;

import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.learning.Informations;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;

public class EgreedyDecrement<A> extends Egreedy<A> {
    private double minEpsilon;
    private int numberStep ;
    private int schedule ;

    private int stepEpsilons;

    public EgreedyDecrement(double minEpsilon, int schedule,long seed,ActionSpace<A> actionSpace, Policy greedypolicy, int stepEpsilons) {
        super(1,seed,actionSpace,greedypolicy);
        this.minEpsilon = minEpsilon;
        this.numberStep = 1 ;
        this.stepEpsilons = stepEpsilons;
        this.epsilon = Math.min(1f, Math.max(this.minEpsilon, 1. /(double)this.stepEpsilons)) ;
        this.schedule = schedule ;
    }

    @Override
    public Object getAction(INDArray results,Informations information) {
        this.modifyEpsilon();
        return super.getAction(results,information);
    }

    private void modifyEpsilon(){
        this.numberStep ++ ;
        if(this.numberStep % this.schedule == 0){
            this.numberStep = 0 ;
            this.stepEpsilons++ ; // Augmente de 1 tous les schedule iterations
            this.epsilon = Math.min(1f, Math.max(this.minEpsilon, 1. / (double)this.stepEpsilons )) ;
            System.out.println("new Epsilon "+this.epsilon);
        }
    }
}
