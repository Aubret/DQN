package fr.univlyon1.actorcritic;

import fr.univlyon1.actorcritic.policy.EgreedyDecrement;
import fr.univlyon1.actorcritic.policy.NoisyGreedy;
import fr.univlyon1.actorcritic.policy.Policy;
import fr.univlyon1.agents.AgentDRL;
import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.learning.TDActorCritic;
import fr.univlyon1.learning.TDLstm;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.networks.LSTM;
import fr.univlyon1.networks.Mlp;
import fr.univlyon1.networks.lossFunctions.LossError;
import fr.univlyon1.networks.lossFunctions.LossIdentity;
import org.deeplearning4j.nn.conf.Updater;
import org.nd4j.linalg.activations.Activation;
<<<<<<< HEAD
import org.nd4j.linalg.api.ndarray.INDArray;
=======
import org.nd4j.linalg.learning.config.Adam;
>>>>>>> master

public class LstmActorCritic<A> extends ContinuousActorCritic<A> {

    public LSTM observationApproximator ;

    public LstmActorCritic(ObservationSpace observationSpace, ActionSpace<A> actionSpace, Configuration conf, long seed) {
        super(observationSpace, actionSpace, conf, seed);
        this.initLstm();
    }

    public void init(){
        this.initActor(seed);
        this.initCritic(seed);
        this.td = new TDLstm<A>(conf.getGamma(),
                this,
                new SequentialExperienceReplay<A>(conf.getSizeExperienceReplay(),conf.getFile(),conf.getForwardTime(),conf.getBackpropTime()),
                conf.getIterations(),
                this.criticApproximator,
                this.cloneMaximizeCriticApproximator,
                this.observationApproximator
        );
        this.policy = new NoisyGreedy(conf.getNoisyGreedyStd(),conf.getNoisyGreedyMean(),seed,this.getPolicyApproximator());
        /*this.policy = new EgreedyDecrement<A>(conf.getMinEpsilon(),
                conf.getStepEpsilon(),
                seed,
                actionSpace,
                mixtePolicy,
                conf.getInitStdEpsilon());*/

    }

    @Override
    public A getAction(INDArray input,Double time) {
        //INDArray result = this.policyApproximator.getOneResult(input);
        //INDArray resultBehaviore = Nd4j.zeros(this.getActionSpace().getSize()).add(0.1);
        A actionBehaviore;
        this.td.evaluate(input, this.reward); //Evaluation
        if(AgentDRL.getCount() > 0) { // Ne pas overfitter sur les premières données arrivées
            INDArray resultBehaviore = this.td.behave(input);
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
        this.td.step(input,actionBehaviore,time); // step learning algorithm
        return actionBehaviore;
    }

    public void initLstm(){
        this.observationApproximator = new LSTM(observationSpace.getShape()[0], conf.getNumLstmOutputNodes(), seed);
        this.observationApproximator.setLearning_rate(conf.getLearning_rateLstm());
        this.observationApproximator.setListener(true);
        this.observationApproximator.setNumNodesPerLayer(conf.getLayersLstmHiddenNodes());
        this.observationApproximator.setNumLayers(conf.getNumLstmlayers());
<<<<<<< HEAD
        this.observationApproximator.setMinimize(true);
        this.observationApproximator.setUpdater(Updater.ADAM);
        this.observationApproximator.setLossFunction(new LossError());
        this.observationApproximator.setEpsilon(false);
        this.observationApproximator.setLastActivation(Activation.TANH);
=======
        this.observationApproximator.setUpdater(new Adam(conf.getLearning_rateCritic()));
        this.observationApproximator.setEpsilon(true);

>>>>>>> master
        this.observationApproximator.init() ;
    }


    private void initActor(long seed){
        this.policyApproximator =new Mlp(conf.getNumLstmOutputNodes(),this.actionSpace.getSize(),seed);
        this.policyApproximator.setLearning_rate(conf.getLearning_rate());
        this.policyApproximator.setNumNodes(conf.getNumHiddenNodes());
        this.policyApproximator.setNumLayers(conf.getNumLayers());
        //this.policyApproximator.setEpsilon(true); // On retropropage le gradient avec epsilon et non une foncitno de perte
        this.policyApproximator.setLossFunction(new LossError());
        this.policyApproximator.setEpsilon(false);
        this.policyApproximator.setMinimize(false); // On souhaite minimiser le gradient
        this.policyApproximator.setListener(true);
        this.policyApproximator.setUpdater(Updater.ADAM);
        this.policyApproximator.setLastActivation(Activation.TANH);
        this.policyApproximator.setHiddenActivation(Activation.RELU);
        this.policyApproximator.setNumNodesPerLayer(conf.getLayersHiddenNodes());
        //this.policyApproximator.setL2(0.0002);
        //this.policyApproximator.setDropout(true);
        //this.policyApproximator.setBatchNormalization(true);
        //this.policyApproximator.setFinalBatchNormalization(true);
        //this.policyApproximator.setLossFunction(new LossError());

        this.policyApproximator.init() ; // A la fin
    }

    private void initCritic(long seed){
        this.criticApproximator = new Mlp(conf.getNumLstmOutputNodes()+this.actionSpace.getSize(), 1, seed);
        this.criticApproximator.setLearning_rate(conf.getLearning_rateCritic());
        this.criticApproximator.setListener(true);
        this.criticApproximator.setNumNodes(conf.getNumCriticHiddenNodes());
        this.criticApproximator.setNumLayers(conf.getNumCriticLayers());
        this.criticApproximator.setEpsilon(false);
        this.criticApproximator.setHiddenActivation(Activation.RELU);
        //this.criticApproximator.setDropout(true);
        this.criticApproximator.setUpdater(Updater.ADAM);
        this.criticApproximator.setNumNodesPerLayer(conf.getLayersCriticHiddenNodes());
        //this.criticApproximator.setL2(0.001);
        //this.criticApproximator.setBatchNormalization(true);
        //this.criticApproximator.setFinalBatchNormalization(true);
        this.criticApproximator.init() ;

        this.cloneMaximizeCriticApproximator = new Mlp(conf.getNumLstmOutputNodes()+this.actionSpace.getSize(), 1, seed);
        this.cloneMaximizeCriticApproximator.setLearning_rate(conf.getLearning_rateCritic());
        this.cloneMaximizeCriticApproximator.setListener(false);
        this.cloneMaximizeCriticApproximator.setNumNodes(conf.getNumCriticHiddenNodes());
        this.cloneMaximizeCriticApproximator.setNumLayers(conf.getNumCriticLayers());
        this.cloneMaximizeCriticApproximator.setMinimize(false);
        this.cloneMaximizeCriticApproximator.setEpsilon(false);
        this.cloneMaximizeCriticApproximator.setHiddenActivation(Activation.RELU);
        this.cloneMaximizeCriticApproximator.setLossFunction(new LossIdentity());
        this.cloneMaximizeCriticApproximator.setUpdater(Updater.ADAM);
        this.cloneMaximizeCriticApproximator.setNumNodesPerLayer(conf.getLayersCriticHiddenNodes());
        //this.cloneMaximizeCriticApproximator.setL2(0.001);
        //this.cloneMaximizeCriticApproximator.setListener(true);
        //this.cloneMaximizeCriticApproximator.setBatchNormalization(true);
        //this.cloneMaximizeCriticApproximator.setFinalBatchNormalization(true);
        this.cloneMaximizeCriticApproximator.init();
        this.cloneMaximizeCriticApproximator.setParams(this.criticApproximator.getParams());
    }

}