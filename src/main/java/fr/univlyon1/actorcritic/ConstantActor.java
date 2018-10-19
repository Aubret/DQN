package fr.univlyon1.actorcritic;

import fr.univlyon1.actorcritic.policy.Policy;
import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.Observation;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.learning.Algorithm;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.memory.RandomExperienceReplay;
import fr.univlyon1.networks.Approximator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

public class ConstantActor<A> implements Learning<A> {
    protected ActionSpace<A> actionSpace ;
    protected ObservationSpace observationSpace;
    protected ExperienceReplay<A> ep ;
    protected long seed ;
    protected Configuration conf ;
    protected Random random ;



    public ConstantActor(ObservationSpace observationSpace, ActionSpace<A> actionSpace, Configuration conf, long seed){
        this.observationSpace = observationSpace;
        this.actionSpace = actionSpace ;
        this.conf = conf ;
        this.seed = seed ;
        this.random = new Random(seed);
    }


    @Override
    public void init() {
        this.ep = new RandomExperienceReplay<A>(conf.getSizeExperienceReplay(),seed,conf.getReadfile());


    }

    @Override
    public void putReward(Double reward) {

    }

    @Override
    public A getAction(Observation observation, Double time) {
        Double interdist ;
        if(observation.getData().getDouble(4) == 1.) {
            interdist = -1. ;
        }else{
            interdist=1. ;
        }

        INDArray behaviore= Nd4j.create(new double[]{interdist,1.,-1.});
        return this.actionSpace.mapNumberToAction(behaviore);
    }

    @Override
    public Configuration getConf() {
        return null;
    }

    @Override
    public ObservationSpace getObservationSpace() {
        return null;
    }

    @Override
    public ExperienceReplay<A> getExperienceReplay() {
        return this.ep ;
    }

    @Override
    public Approximator getApproximator() {
        return null;
    }

    @Override
    public Approximator getModelApproximator() {
        return null;
    }

    @Override
    public Policy getPolicy() {
        return null;
    }

    @Override
    public ActionSpace getActionSpace() {
        return null;
    }

    @Override
    public void stop() {

    }
}
