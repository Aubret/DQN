package fr.univlyon1.actorcritic;

import fr.univlyon1.actorcritic.policy.NoisyGreedy;
import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.environment.ObservationSpace;
import fr.univlyon1.memory.RandomExperienceReplay;
import fr.univlyon1.networks.GradMlp;
import fr.univlyon1.actorcritic.policy.Policy;
import fr.univlyon1.environment.ActionSpace;
import fr.univlyon1.learning.TDActorCritic;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.networks.Mlp;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ContinuousActorCritic<A> implements Learning<A> {
    private Configuration conf ;
    private Approximator policyApproximator ;
    private ActionSpace<A> actionSpace ;
    private TDActorCritic<A> td ;
    private Policy policy ;
    private Double reward ;
    private int epoch ;
    private int countStep ;


    public ContinuousActorCritic(ObservationSpace observationSpace, ActionSpace<A> actionSpace, Configuration conf, long seed){
        this.conf = conf ;
        this.actionSpace = actionSpace ;
        this.policyApproximator =new Mlp(observationSpace.getShape()[0],actionSpace.getSize(),seed,false,conf.getLearning_rate(),conf.getNumLayers(),conf.getNumHiddenNodes(),false,true) ;
        this.policy = new NoisyGreedy(conf.getNoisyGreedyStd(),conf.getNoisyGreedyMean(),seed);
        Approximator critic = new GradMlp(observationSpace.getShape()[0]+this.actionSpace.getSize(), 1, seed, true, conf.getLearning_rateCritic(), conf.getNumCriticLayers(), conf.getNumCriticHiddenNodes(), true, false);
        this.td = new TDActorCritic<A>(conf.getGamma(),
                this,
                new RandomExperienceReplay<A>(conf.getSizeExperienceReplay())
                ,conf.getBatchSize()
                ,conf.getIterations()
                ,critic
                );
    }

    @Override
    public Configuration getConf() {
        return conf;
    }

    @Override
    public A getAction(INDArray input) {
        INDArray result = this.policyApproximator.getOneResult(input);
        INDArray resultBehaviore = (INDArray)this.policy.getAction(result);
        //if(AgentDRL.getCount() > 50) { // Ne pas overfitter sur les premières données arrivées
            this.td.evaluate(input, this.reward); //Evaluation
            this.countStep++;
            if (this.countStep == this.epoch) {
                countStep = 0;
                this.td.epoch();
            }
        //}
        A actionBehaviore = this.actionSpace.mapNumberToAction(resultBehaviore);
        this.td.step(input,actionBehaviore,result); // step learning algorithm
        return actionBehaviore;
    }

    @Override
    public void putReward(Double reward) {
        this.reward = reward;
    }

    @Override
    public Approximator getApproximator() {
        return this.policyApproximator;
    }

    @Override
    public ActionSpace<A> getActionSpace() {
        return this.actionSpace;
    }

    @Override
    public void stop() {

    }

    public Approximator getPolicyApproximator() {
        return policyApproximator;
    }

    public void setPolicyApproximator(Approximator policyApproximator) {
        this.policyApproximator = policyApproximator;
    }

}
