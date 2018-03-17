package main.java.fr.univlyon1.actorcritic;

import main.java.fr.univlyon1.Configuration;
import main.java.fr.univlyon1.actorcritic.policy.Policy;
import main.java.fr.univlyon1.environment.ActionSpace;
import main.java.fr.univlyon1.learning.TDBatch;
import main.java.fr.univlyon1.networks.Approximator;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ContinuousActorCritic<A> implements Learning<A> {
    private Approximator mlp ;
    private ActionSpace<A> actionSpace ;
    private TDBatch<A> td ;
    private Policy policy ;
    private Double reward ;
    private int epoch ;
    private int countStep ;


    public ContinuousActorCritic(){
    }

    @Override
    public Configuration getConf() {
        return null;
    }

    @Override
    public A getAction(INDArray input) {
        return null;
    }

    @Override
    public void putReward(Double reward) {

    }

    @Override
    public Approximator getApproximator() {
        return null;
    }

    @Override
    public ActionSpace<A> getActionSpace() {
        return null;
    }

    @Override
    public void stop() {

    }
}
