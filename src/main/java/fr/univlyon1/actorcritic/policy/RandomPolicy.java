package fr.univlyon1.actorcritic.policy;

import fr.univlyon1.environment.space.ActionSpace;
import org.nd4j.linalg.api.ndarray.INDArray;

public class RandomPolicy<A> implements Policy<A>{

    protected ActionSpace<A> actionSpace ;

    public RandomPolicy(ActionSpace<A> actionSpace){
        this.actionSpace = actionSpace ;
    }
    @Override
    public Object getAction(INDArray results) {
        return this.actionSpace.randomAction();
    }
}
