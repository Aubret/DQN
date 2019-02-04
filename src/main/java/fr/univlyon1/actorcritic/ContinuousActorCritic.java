package fr.univlyon1.actorcritic;

import fr.univlyon1.actorcritic.policy.*;
import fr.univlyon1.agents.AgentDRL;
import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.environment.space.Observation;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.learning.Algorithm;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.memory.RandomExperienceReplay;
import fr.univlyon1.memory.prioritizedExperienceReplay.PrioritizedExperienceReplay;
import fr.univlyon1.memory.prioritizedExperienceReplay.StochasticPrioritizedExperienceReplay;
import fr.univlyon1.networks.*;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.learning.TDActorCritic;
import fr.univlyon1.networks.lossFunctions.LossError;
import fr.univlyon1.networks.lossFunctions.LossIdentity;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.conf.Updater;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.RmsProp;

@Getter
@Setter
/**
 * Class used with two neural nets, one for the policy and one for the critic
 */
public class ContinuousActorCritic<A> implements Learning<A> {
    protected Configuration conf ;
    protected Mlp policyApproximator ;
    protected Mlp criticApproximator ;
    protected Mlp cloneMaximizeCriticApproximator ;
    protected ActionSpace<A> actionSpace ;
    protected Algorithm<A> td ;
    protected Policy policy ;
    protected Double reward ;
    protected int epoch ;
    protected int countStep ;
    protected ObservationSpace observationSpace;
    protected ExperienceReplay<A> ep ;
    protected long seed ;

    public ContinuousActorCritic(ObservationSpace observationSpace, ActionSpace<A> actionSpace, Configuration conf, long seed){
        this.conf = conf ;
        this.actionSpace = actionSpace ;
        this.observationSpace = observationSpace ;
        this.seed = seed ;
        //this.policy = new NoisyGreedy(conf.getNoisyGreedyStd(),conf.getNoisyGreedyMean(),seed);
        this.epoch = conf.getEpochs() ;
    }

    public void init(){
        this.initActor(seed);
        this.initCritic(seed);
        this.ep = new RandomExperienceReplay<A>(conf.getSizeExperienceReplay(),seed,conf.getReadfile());
        //ExperienceReplay<A> ep = new PrioritizedExperienceReplay<A>(conf.getSizeExperienceReplay());
        //this.ep = new StochasticPrioritizedExperienceReplay<A>(conf.getSizeExperienceReplay(),seed,conf.getReadfile());
        this.ep.load(actionSpace);
        this.td = new TDActorCritic<A>(conf.getGamma(),
                this,
                this.ep,
                conf.getBatchSize(),
                conf.getIterations(),
                this.criticApproximator,
                this.cloneMaximizeCriticApproximator
        );
        //Policy mixtePolicy = new NoisyGreedy(conf.getNoisyGreedyStd(),conf.getNoisyGreedyMean(),seed,this.getPolicyApproximator());
        /*this.policy = new EgreedyDecrement<A>(conf.getMinEpsilon(),
                conf.getStepEpsilon(),
                seed,
                actionSpace,
                mixtePolicy,
                conf.getInitStdEpsilon());*/

        /*this.policy = new Egreedy<A>(conf.getMinEpsilon(),
                seed,
                actionSpace,
                this.getPolicyApproximator());*/
        /*this.policy = new EgreedyDecrement<A>(conf.getMinEpsilon(),
                conf.getStepEpsilon(),
                seed,
                actionSpace,
                this.policyApproximator, // If no noise
                conf.getInitStdEpsilon()
        );*/
        /*this.policy = new NoisyGreedyDecremental(
                conf.getNoisyGreedyStd(),
                conf.getNoisyGreedyMean(),
                conf.getInitStdEpsilon(),
                conf.getStepEpsilon(),
                seed,
                this.getPolicyApproximator()
        );*/
        //this.policy = new NoisyGreedy(conf.getNoisyGreedyStd(),conf.getNoisyGreedyMean(),seed,this.getPolicyApproximator());
        Policy mixtePolicy = new ParameterNoise<A>(conf.getNoisyGreedyStd(),this.seed,this.actionSpace, this.getPolicyApproximator(),conf.getStepEpsilon());
        this.policy = new Egreedy<A>(conf.getMinEpsilon(),this.seed,this.actionSpace,mixtePolicy);

    }

    @Override
    public Configuration getConf() {
        return conf;
    }

    @Override
    public A getAction(Observation observation, Double time) {
        INDArray input = observation.getData() ;
        //INDArray result = this.policyApproximator.getOneResult(input);
        //INDArray resultBehaviore = Nd4j.zeros(this.getActionSpace().getSize()).add(0.1);
        A actionBehaviore;
        this.td.evaluate(input, this.reward,time); //Evaluation
        if(AgentDRL.getCount() > 1000) { // Ne pas overfitter sur les premières données arrivées
            INDArray resultBehaviore = this.td.behave(input);//(INDArray)this.policy.getAction(input);
            this.td.learn();
            this.countStep++;
            if (this.countStep == this.epoch) {
                countStep = 0;
                this.td.epoch();
                //System.out.println("An epoch : "+ AgentDRL.getCount());
            }
            actionBehaviore = this.actionSpace.mapNumberToAction(resultBehaviore);
        }else
            actionBehaviore= this.actionSpace.mapNumberToAction(this.actionSpace.randomAction());
        this.td.step(observation,actionBehaviore,time); // step learning algorithm
        return actionBehaviore;
    }

    @Override
    public ExperienceReplay<A> getExperienceReplay() {
        return this.ep ;
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
    public Approximator getModelApproximator() {
        return null;
    }

    @Override
    public ActionSpace<A> getActionSpace() {
        return this.actionSpace;
    }

    @Override
    public void stop() {
        System.out.println("Policy approximator");
        this.policyApproximator.stop();
        System.out.println("critic approximator");
        this.criticApproximator.stop();
    }

    private void initActor(long seed){
        //this.policyApproximator =new Mlp(conf.getNumLstmOutputNodes()+observationSpace.getShape()[0],this.actionSpace.getSize(),this.seed);
        this.policyApproximator =new NormalizedMlp(observationSpace.getShape()[0],this.actionSpace.getSize(),this.seed);
        this.policyApproximator.setLearning_rate(conf.getLearning_rate());
        this.policyApproximator.setNumNodes(conf.getNumHiddenNodes());
        this.policyApproximator.setNumLayers(conf.getNumLayers());
        //this.policyApproximator.setEpsilon(true); // On retropropage le gradient avec epsilon et non une foncitno de perte
        this.policyApproximator.setLossFunction(new LossError());
        this.policyApproximator.setEpsilon(false);
        this.policyApproximator.setMinimize(false); // On souhaite minimiser le gradient
        this.policyApproximator.setListener(true);
        this.policyApproximator.setUpdater(new Adam(conf.getLearning_rate()));
        this.policyApproximator.setLastActivation(Activation.TANH);
        this.policyApproximator.setHiddenActivation(Activation.ELU);
        this.policyApproximator.setNumNodesPerLayer(conf.getLayersHiddenNodes());
        this.policyApproximator.setName("Policy");
        //this.policyApproximator.setL2(0.0002);
        //this.policyApproximator.setDropout(true);
        ((NormalizedMlp)this.policyApproximator).setLayerNormalization(true);
        ((NormalizedMlp)this.policyApproximator).setBatchNormalization(false);
        //this.policyApproximator.setLossFunction(new LossError());

        this.policyApproximator.init() ; // A la fin
    }

    private void initCritic(long seed){
        this.criticApproximator = new Mlp(observationSpace.getShape()[0]+this.actionSpace.getSize(), 1, this.seed);
        this.criticApproximator.setLearning_rate(conf.getLearning_rateCritic());
        this.criticApproximator.setListener(true);
        this.criticApproximator.setNumNodes(conf.getNumCriticHiddenNodes());
        this.criticApproximator.setNumLayers(conf.getNumCriticLayers());
        this.criticApproximator.setEpsilon(false);
        this.criticApproximator.setHiddenActivation(Activation.ELU);
        //this.criticApproximator.setDropout(true);
        this.criticApproximator.setUpdater(new Adam(conf.getLearning_rateCritic()));
        this.criticApproximator.setNumNodesPerLayer(conf.getLayersCriticHiddenNodes());
        this.criticApproximator.setName("Critic");
        //this.criticApproximator.setL2(0.00001);
        this.criticApproximator.init() ;

        this.cloneMaximizeCriticApproximator = new Mlp(observationSpace.getShape()[0]+this.actionSpace.getSize(), 1, this.seed);
        this.cloneMaximizeCriticApproximator.setLearning_rate(conf.getLearning_rateCritic());
        this.cloneMaximizeCriticApproximator.setListener(false);
        this.cloneMaximizeCriticApproximator.setNumNodes(conf.getNumCriticHiddenNodes());
        this.cloneMaximizeCriticApproximator.setNumLayers(conf.getNumCriticLayers());
        this.cloneMaximizeCriticApproximator.setMinimize(false);
        this.cloneMaximizeCriticApproximator.setEpsilon(false);
        this.cloneMaximizeCriticApproximator.setHiddenActivation(Activation.ELU);
        this.cloneMaximizeCriticApproximator.setLossFunction(new LossIdentity());
        this.cloneMaximizeCriticApproximator.setUpdater(new Adam(conf.getLearning_rateCritic()));
        this.cloneMaximizeCriticApproximator.setNumNodesPerLayer(conf.getLayersCriticHiddenNodes());
        this.cloneMaximizeCriticApproximator.setName("dQdA");
        //this.cloneMaximizeCriticApproximator.setL2(0.00001);
        //this.cloneMaximizeCriticApproximator.setListener(true);
        this.cloneMaximizeCriticApproximator.init();
        this.cloneMaximizeCriticApproximator.setParams(this.criticApproximator.getParams());
    }

    public Approximator getPolicyApproximator() {
        return this.policyApproximator;
    }

    public void setPolicyApproximator(Approximator policyApproximator) {
        this.policyApproximator = (Mlp)policyApproximator;
    }

    public Double getScore(){
        return this.td.getInformation().getScore() ;
    }



}
