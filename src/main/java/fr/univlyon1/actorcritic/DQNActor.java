package fr.univlyon1.actorcritic;

import fr.univlyon1.actorcritic.policy.EgreedyDecrement;
import fr.univlyon1.actorcritic.policy.Policy;
import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.environment.ObservationSpace;
import fr.univlyon1.learning.TDBatch;
import fr.univlyon1.memory.RandomExperienceReplay;
import fr.univlyon1.agents.AgentDRL;
import fr.univlyon1.environment.ActionSpace;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.networks.Mlp;
import org.nd4j.linalg.api.ndarray.INDArray;

public class DQNActor<A> implements Learning<A> {
    private Configuration conf ;
    private Approximator mlp ;
    private ActionSpace<A> actionSpace ;
    private TDBatch<A> td ;
    private Policy policy ;
    private Double reward ;
    private int epoch ;
    private int countStep ;

    public DQNActor(ObservationSpace observationSpace, ActionSpace<A> actionSpace, Configuration conf, long seed){
        this.conf = conf ;
        this.mlp =new Mlp(observationSpace.getShape()[0],actionSpace.getSize(),seed,false,conf.getLearning_rate(),conf.getNumLayers(),conf.getNumHiddenNodes(),true,false) ;
        this.actionSpace = actionSpace ;
        int batchSize = conf.getBatchSize();
        int iterations = conf.getIterations() ;
        this.epoch = conf.getEpochs() ; // nombre d'itérations avant pour chaque clônage de réseau.
        this.countStep=0;
        this.td = new TDBatch<A>(conf.getGamma(),this, new RandomExperienceReplay<A>(conf.getSizeExperienceReplay()),batchSize,iterations) ;// experience can be null
        //this.policy = new Egreedy(0.2,seed);
        this.policy = new EgreedyDecrement(conf.getMinEpsilon(),conf.getStepEpsilon(),seed);
    }

    @Override
    public void putReward(Double reward){
        this.reward = reward;
    }

    @Override
    public Approximator getApproximator() {
        return this.mlp ;
    }

    @Override
    public ActionSpace<A> getActionSpace() {
        return this.actionSpace ;
    }

    @Override
    public Configuration getConf() {
        return conf ;
    }

    @Override
    public A getAction(INDArray input) {
        if(AgentDRL.getCount() > 50) { // Ne pas overfitter sur les premières données arrivées
            this.td.evaluate(input, this.reward); //Evaluation
            this.countStep++;
            if (this.countStep == this.epoch) {
                countStep = 0;
                //System.out.println("CLONE");
                //this.mlp = this.td.getApproximator().clone();
                this.td.epoch();
            }
        }
        INDArray results  = this.mlp.getOneResult(input); // get action behavioure
        int indiceBehaviore = (Integer)this.policy.getAction(results);
        A actionBehaviore = this.actionSpace.mapNumberToAction(indiceBehaviore);

        this.td.step(input,actionBehaviore,results); // step learning algorithm
        return actionBehaviore;
    }

    @Override
    public void stop() {
        this.td.getApproximator().stop();
    }


}
