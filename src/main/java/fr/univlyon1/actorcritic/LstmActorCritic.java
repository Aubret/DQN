package fr.univlyon1.actorcritic;

import fr.univlyon1.actorcritic.policy.Egreedy;
import fr.univlyon1.actorcritic.policy.NoisyGreedy;
import fr.univlyon1.actorcritic.policy.ParameterNoise;
import fr.univlyon1.actorcritic.policy.Policy;
import fr.univlyon1.agents.AgentDRL;
import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.configurations.ListPojo;
import fr.univlyon1.configurations.PojoInteraction;
import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.interactions.Replayable;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.Observation;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.learning.TDLstm2D;
import fr.univlyon1.learning.TDLstmFilter;
import fr.univlyon1.memory.OneVehicleSequentialExperienceReplay;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.memory.prioritizedExperienceReplay.SequentialPrioritizedExperienceReplay;
import fr.univlyon1.networks.*;
import fr.univlyon1.networks.lossFunctions.LossError;
import fr.univlyon1.networks.lossFunctions.LossIdentity;
import fr.univlyon1.networks.lossFunctions.LossMseSaveScore;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import java.io.File;
import java.util.Collection;
import java.util.List;

/**
 * Class using 3 neural nets, one for memory, one for the policy and one for critic.
 * @param <A>
 */
public class LstmActorCritic<A> extends ContinuousActorCritic<A> {

    protected LSTM observationApproximator ;
    protected int learn_step ;
    protected int restartMemory ;

    public LstmActorCritic(ObservationSpace observationSpace, ActionSpace<A> actionSpace, Configuration conf, long seed) {
        super(observationSpace, actionSpace, conf, seed);
        this.learn_step = conf.getLearn();
        this.restartMemory = 30 ;
    }

    public void init(){
        this.initActor();
        this.initCritic();
        this.initLstm();
        this.ep = new SequentialExperienceReplay<A>(conf.getSizeExperienceReplay(),conf.getReadfile(),conf.getForwardTime(),conf.getBackpropTime(),this.seed,conf.getForward());
        //this.ep = new OneVehicleSequentialExperienceReplay<A>(conf.getSizeExperienceReplay(),conf.getReadfile(),conf.getForwardTime(),conf.getBackpropTime(),this.seed,conf.getForward());
        this.td = new TDLstm2D<A>(conf.getGamma(),
        //this.td = new TDLstmFilter<A>(conf.getGamma(),
                this,
                (SequentialExperienceReplay<A>)this.ep,
                //new SequentialPrioritizedExperienceReplay<A>(conf.getSizeExperienceReplay(),conf.getReadfile(),conf.getForwardTime(),conf.getBackpropTime(),this.seed,conf.getLearn(),conf.getForward()),
                //new SequentialFixedNumber<A>(conf.getSizeExperienceReplay(),conf.getFile(),conf.getForwardTime(),conf.getBackpropTime(),this.seed,conf.getLearn()),
                conf.getIterations(),
                conf.getBatchSize(),
                this.criticApproximator,
                this.cloneMaximizeCriticApproximator,
                this.observationApproximator
        );
        //this.policy = new NoisyGreedyDecremental(conf.getNoisyGreedyStd(),conf.getNoisyGreedyMean(),conf.getInitStdEpsilon(),conf.getStepEpsilon(),seed,this.getPolicyApproximator());
        //Policy mixtePolicy = new NoisyGreedyDecremental(conf.getNoisyGreedyStd(),conf.getNoisyGreedyMean(),conf.getInitStdEpsilon(),conf.getStepEpsilon(),seed,this.getPolicyApproximator());

        /*Policy mixtePolicy = new NoisyGreedy(conf.getNoisyGreedyStd(),conf.getNoisyGreedyMean(),seed,this.getPolicyApproximator());
        Policy mixtePolicy2 = new EgreedyDecrement<A>(conf.getMinEpsilon(),
                conf.getStepEpsilon(),
                seed,
                actionSpace,
                mixtePolicy,
                conf.getInitStdEpsilon());
        //this.policy = mixtePolicy2;
        this.policy = new DoublePolicy<A>(mixtePolicy2,new Egreedy<A>(0.2,seed,actionSpace,this.getPolicyApproximator()));*/
        //this.policy = new ParameterNoise<A>(conf.getNoisyGreedyStd(),this.seed,this.actionSpace, this.getPolicyApproximator(),conf.getStepEpsilon());
        Policy mixtePolicy = new ParameterNoise<A>(conf.getNoisyGreedyStd(),this.seed,this.actionSpace, this.getPolicyApproximator(),conf.getStepEpsilon());
        //this.policy = new NoisyGreedy(conf.getNoisyGreedyStd(),conf.getNoisyGreedyMean(),seed,mixtePolicy);
        this.policy = new Egreedy<A>(conf.getMinEpsilon(),this.seed,this.actionSpace,mixtePolicy);
        //this.policy = new Egreedy<A>(conf.getMinEpsilon(),this.seed,this.actionSpace,this.getPolicyApproximator());

    }

    @Override
    public A getAction(Observation observation, Double time) {
        INDArray input = observation.getData() ;
        //INDArray result = this.policyApproximator.getOneResult(input);
        //INDArray resultBehaviore = Nd4j.zeros(this.getActionSpace().getSize()).add(0.1);
        A actionBehaviore;
        INDArray resultBehaviore;
        //if(AgentDRL.getCount() > 1000)
        this.td.evaluate(observation, this.reward,time); //Evaluation
        if(AgentDRL.getCount() > 300) { // Ne pas overfitter sur les premières données arrivées
            resultBehaviore = this.td.behave(input);
            actionBehaviore = this.actionSpace.mapNumberToAction(resultBehaviore);
            if(AgentDRL.getCount()%this.learn_step== 0) {
                this.td.learn();
                this.countStep++;
                if (this.countStep == this.epoch) {
                    countStep = 0;
                    this.td.epoch();
                    //System.out.println("An epoch : "+ AgentDRL.getCount());
                }
            }
        }else {
            resultBehaviore = (INDArray)this.actionSpace.randomAction();
            actionBehaviore = this.actionSpace.mapNumberToAction(resultBehaviore);
        }
        //if(AgentDRL.getCount() > 1000)
        this.td.step(observation,actionBehaviore,time); // step learning algorithm
        return actionBehaviore;
    }

    public void initLstm(){
        this.observationApproximator = new LSTM2D(observationSpace.getShape()[0]+this.actionSpace.getSize(), conf.getNumLstmOutputNodes(), seed);
        this.observationApproximator.setLearning_rate(conf.getLearning_rateLstm());
        this.observationApproximator.setListener(true);
        this.observationApproximator.setNumNodesPerLayer(conf.getLayersLstmHiddenNodes());
        this.observationApproximator.setNumLayers(conf.getNumLstmlayers());
        this.observationApproximator.setNumNodes(conf.getNumLstmHiddenNodes());
        this.observationApproximator.setUpdater(new Adam(conf.getLearning_rateLstm()));
        this.observationApproximator.setEpsilon(false);
        this.observationApproximator.setMinimize(true);
        this.observationApproximator.setLossFunction(new LossError()); //new LossMseSaveScore());
        this.observationApproximator.setHiddenActivation(Activation.TANH);
        this.observationApproximator.setLastActivation(Activation.TANH);
        //this.observationApproximator.setImportModel("resources/models/lstm");
        this.observationApproximator.setName("Lstm");
        //this.observationApproximator.setL2(0.001);
        this.observationApproximator.init() ;
    }

    private void initActor(){
        this.policyApproximator =new Mlp(conf.getNumLstmOutputNodes()+observationSpace.getShape()[0],this.actionSpace.getSize(),this.seed);
        //this.policyApproximator =new NormalizedMlp(conf.getNumLstmOutputNodes()+observationSpace.getShape()[0],this.actionSpace.getSize(),this.seed);
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
        //this.policyApproximator.setL2(0.0001);
        //this.policyApproximator.setDropout(true);
        //((NormalizedMlp)this.policyApproximator).setLayerNormalization(true);
        //((NormalizedMlp)this.policyApproximator).setBatchNormalization(false);
        //this.policyApproximator.setLossFunction(new LossError());

        this.policyApproximator.init() ; // A la fin
    }

    private void initCritic(){
        this.criticApproximator = new Mlp(conf.getNumLstmOutputNodes()+observationSpace.getShape()[0]+this.actionSpace.getSize(), 1, this.seed);
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
        //this.criticApproximator.setL2(0.001);
        this.criticApproximator.init() ;

        this.cloneMaximizeCriticApproximator = new Mlp(conf.getNumLstmOutputNodes()+observationSpace.getShape()[0]+this.actionSpace.getSize(), 1, this.seed);
        this.cloneMaximizeCriticApproximator.setLearning_rate(conf.getLearning_rateCritic());
        this.cloneMaximizeCriticApproximator.setListener(true);
        this.cloneMaximizeCriticApproximator.setNumNodes(conf.getNumCriticHiddenNodes());
        this.cloneMaximizeCriticApproximator.setNumLayers(conf.getNumCriticLayers());
        this.cloneMaximizeCriticApproximator.setMinimize(false);
        this.cloneMaximizeCriticApproximator.setEpsilon(false);
        this.cloneMaximizeCriticApproximator.setHiddenActivation(Activation.ELU);
        this.cloneMaximizeCriticApproximator.setLossFunction(new LossIdentity());
        this.cloneMaximizeCriticApproximator.setUpdater(new Adam(conf.getLearning_rateCritic()));
        this.cloneMaximizeCriticApproximator.setNumNodesPerLayer(conf.getLayersCriticHiddenNodes());
        this.cloneMaximizeCriticApproximator.setName("dQdA");
        //this.cloneMaximizeCriticApproximator.setL2(0.001);
        //this.cloneMaximizeCriticApproximator.setListener(true);
        this.cloneMaximizeCriticApproximator.init();
        this.cloneMaximizeCriticApproximator.setParams(this.criticApproximator.getParams());
    }

    @Override
    public void stop() {
        System.out.println("Policy approximator");
        this.policyApproximator.stop();
        System.out.println("critic approximator");
        this.criticApproximator.stop();
        System.out.println("observation approximator");
        this.observationApproximator.stop();
        if(AgentDRL.isWriteFile() && !this.conf.getWritefile().equals("")){
            ListPojo<A> point = new ListPojo<A>();
            Collection<? extends Replayable<A>> memory = this.ep.getMemory();
            for(Replayable<A> replayable : memory){
                if(replayable instanceof Interaction) {
                    Interaction<A> interaction = (Interaction<A>)replayable ;
                    point.add(new PojoInteraction<A>(interaction, this.actionSpace));
                }
            }

            try {
                JAXBContext context = JAXBContext.newInstance(ListPojo.class);
                Marshaller m = context.createMarshaller();
                m.marshal(point,new File(getConf().getWritefile()));
            } catch (JAXBException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public Approximator getModelApproximator() {
        return this.observationApproximator;
    }



}
