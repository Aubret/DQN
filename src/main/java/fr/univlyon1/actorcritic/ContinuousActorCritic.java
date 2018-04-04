package fr.univlyon1.actorcritic;

import fr.univlyon1.actorcritic.policy.*;
import fr.univlyon1.agents.AgentDRL;
import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.memory.RandomExperienceReplay;
import fr.univlyon1.memory.prioritizedExperienceReplay.PrioritizedExperienceReplay;
import fr.univlyon1.networks.*;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.learning.TDActorCritic;
import fr.univlyon1.networks.lossFunctions.LossIdentity;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.conf.Updater;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;

@Getter
@Setter
public class ContinuousActorCritic<A> implements Learning<A> {
    private Configuration conf ;
    private Mlp policyApproximator ;
    private Mlp criticApproximator ;
    private Mlp cloneMaximizeCriticApproximator ;
    private ActionSpace<A> actionSpace ;
    private TDActorCritic<A> td ;
    private Policy policy ;
    private Double reward ;
    private int epoch ;
    private int countStep ;
    private ObservationSpace observationSpace;

    public ContinuousActorCritic(ObservationSpace observationSpace, ActionSpace<A> actionSpace, Configuration conf, long seed){
        this.conf = conf ;
        this.actionSpace = actionSpace ;
        this.observationSpace = observationSpace ;
        //this.policy = new NoisyGreedy(conf.getNoisyGreedyStd(),conf.getNoisyGreedyMean(),seed);
        this.epoch = conf.getEpochs() ;
        this.initActor(seed);
        this.initCritic(seed);
        ExperienceReplay<A> ep = new RandomExperienceReplay<A>(conf.getSizeExperienceReplay(),seed);
        //ExperienceReplay<A> ep = new PrioritizedExperienceReplay<A>(conf.getSizeExperienceReplay());
        this.td = new TDActorCritic<A>(conf.getGamma(),
                this,
                ep,
                conf.getBatchSize(),
                conf.getIterations(),
                this.criticApproximator,
                this.cloneMaximizeCriticApproximator
        );
        /*Policy mixtePolicy = new NoisyGreedy(conf.getNoisyGreedyStd(),conf.getNoisyGreedyMean(),seed,this.getPolicyApproximator());
        this.policy = new Egreedy<A>(conf.getMinEpsilon(),
                seed,
                actionSpace,
                mixtePolicy);*/

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
        this.policy = new NoisyGreedy(conf.getNoisyGreedyStd(),conf.getNoisyGreedyMean(),seed,this.getPolicyApproximator());
    }

    @Override
    public Configuration getConf() {
        return conf;
    }

    @Override
    public A getAction(INDArray input) {
        //INDArray result = this.policyApproximator.getOneResult(input);
        //INDArray resultBehaviore = Nd4j.zeros(this.getActionSpace().getSize()).add(0.1);
        A actionBehaviore;
        if(AgentDRL.getCount() > 2000) { // Ne pas overfitter sur les premières données arrivées
            INDArray resultBehaviore = (INDArray)this.policy.getAction(input);
            this.td.evaluate(input, this.reward); //Evaluation
            this.countStep++;
            if (this.countStep == this.epoch) {
                countStep = 0;
                this.td.epoch();
                //System.out.println("An epoch : "+ AgentDRL.getCount());
            }
            actionBehaviore = this.actionSpace.mapNumberToAction(resultBehaviore);
        }else{
            actionBehaviore= this.actionSpace.mapNumberToAction(this.actionSpace.randomAction());
        }
        this.td.step(input,actionBehaviore); // step learning algorithm
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
        System.out.println("Policy approximator");
        this.policyApproximator.stop();
        System.out.println("critic approximator");
        this.criticApproximator.stop();
    }

    private void initActor(long seed){
        this.policyApproximator =new Mlp(this.observationSpace.getShape()[0],this.actionSpace.getSize(),seed);
        this.policyApproximator.setLearning_rate(conf.getLearning_rate());
        this.policyApproximator.setNumNodes(conf.getNumHiddenNodes());
        this.policyApproximator.setNumLayers(conf.getNumLayers());
        this.policyApproximator.setEpsilon(true); // On retropropage le gradient avec epsilon et non une foncitno de perte
        //this.policyApproximator.setLossFunction(new LossError());
        //this.policyApproximator.setEpsilon(false);
        this.policyApproximator.setMinimize(false); // On souhaite minimiser le gradient
        this.policyApproximator.setListener(true);
        this.policyApproximator.setUpdater(Updater.RMSPROP);
        this.policyApproximator.setLastActivation(Activation.TANH);
        this.policyApproximator.setHiddenActivation(Activation.RELU);
        /*this.policyApproximator.setBatchNormalization(true);
        this.policyApproximator.setFinalBatchNormalization(true);*/
        //this.policyApproximator.setLossFunction(new LossError());
        //this.policyApproximator.setListener(true);

        this.policyApproximator.init() ; // A la fin
    }

    private void initCritic(long seed){
        //this.criticApproximator = new GradMlp(observationSpace.getShape()[0]+this.actionSpace.getSize(), 1, seed);
        this.criticApproximator = new Mlp(observationSpace.getShape()[0]+this.actionSpace.getSize(), 1, seed);
        this.criticApproximator.setLearning_rate(conf.getLearning_rateCritic());
        this.criticApproximator.setListener(true);
        this.criticApproximator.setNumNodes(conf.getNumCriticHiddenNodes());
        this.criticApproximator.setNumLayers(conf.getNumCriticLayers());
        this.criticApproximator.setEpsilon(false);
        this.criticApproximator.setUpdater(Updater.RMSPROP);
        /*this.criticApproximator.setBatchNormalization(true);
        this.criticApproximator.setFinalBatchNormalization(true);*/
        this.criticApproximator.init() ;

        this.cloneMaximizeCriticApproximator = new Mlp(observationSpace.getShape()[0]+this.actionSpace.getSize(), 1, seed);
        this.cloneMaximizeCriticApproximator.setLearning_rate(conf.getLearning_rate());
        this.cloneMaximizeCriticApproximator.setListener(false);
        this.cloneMaximizeCriticApproximator.setNumNodes(conf.getNumCriticHiddenNodes());
        this.cloneMaximizeCriticApproximator.setNumLayers(conf.getNumCriticLayers());
        this.cloneMaximizeCriticApproximator.setMinimize(false);
        this.cloneMaximizeCriticApproximator.setEpsilon(false);
        this.cloneMaximizeCriticApproximator.setLossFunction(new LossIdentity());
        /*this.cloneMaximizeCriticApproximator.setBatchNormalization(true);
        this.cloneMaximizeCriticApproximator.setFinalBatchNormalization(true);*/
        this.cloneMaximizeCriticApproximator.init();
        this.cloneMaximizeCriticApproximator.setParams(this.criticApproximator.getParams());
    }

    public Approximator getPolicyApproximator() {
        return policyApproximator;
    }

    public void setPolicyApproximator(Approximator policyApproximator) {
        this.policyApproximator = (Mlp)policyApproximator;
    }

}
