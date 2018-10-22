package fr.univlyon1.actorcritic;

import fr.univlyon1.actorcritic.policy.EgreedyDecrement;
import fr.univlyon1.actorcritic.policy.GreedyDiscrete;
import fr.univlyon1.actorcritic.policy.Policy;
import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.environment.space.Observation;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.learning.TDBatch;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.memory.RandomExperienceReplay;
import fr.univlyon1.agents.AgentDRL;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.networks.Mlp;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;

@Getter
@Setter
public class DQNActor<A> implements Learning<A> {
    private Configuration conf ;
    private Mlp mlp ;
    private ActionSpace<A> actionSpace ;
    private TDBatch<A> td ;
    private Policy<A> policy ;
    private Double reward ;
    private int epoch ;
    private int countStep ;
    private ObservationSpace observationSpace ;
    private long seed ;

    public DQNActor(ObservationSpace observationSpace, ActionSpace<A> actionSpace, Configuration conf, long seed){
        this.conf = conf ;
        this.observationSpace = observationSpace ;
        this.initMlp(seed);
        this.actionSpace = actionSpace ;
        this.epoch = conf.getEpochs() ; // nombre d'itérations avant pour chaque clônage de réseau.
        this.countStep=0;
        this.seed = seed ;
    }

    public void init(){
        int batchSize = conf.getBatchSize();
        int iterations = conf.getIterations() ;
        this.td = new TDBatch<A>(conf.getGamma(),this, new RandomExperienceReplay<A>(conf.getSizeExperienceReplay(),seed,conf.getReadfile()),batchSize,iterations) ;// experience can be null
        //this.policy = new Egreedy(0.2,seed);
        this.policy = new EgreedyDecrement<A>(conf.getMinEpsilon(),conf.getStepEpsilon(),seed,this.actionSpace,new GreedyDiscrete(),conf.getInitStdEpsilon());
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
    public Approximator getModelApproximator() {
        return null;
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
    public A getAction(Observation observation, Double time) {
        INDArray input = observation.getData() ;
        if(AgentDRL.getCount() > 50) { // Ne pas overfitter sur les premières données arrivées
            this.td.evaluate(observation, this.reward,time); //Evaluation
            this.countStep++;
            if (this.countStep == this.epoch) {
                countStep = 0;
                //System.out.println("CLONE");
                //this.mlp = this.td.getApproximator().clone();
                this.td.epoch();
            }
        }
        INDArray results  = this.td.behave(input); // get action behavioure
        int indiceBehaviore = (Integer)this.policy.getAction(results,null);
        A actionBehaviore = this.actionSpace.mapNumberToAction(indiceBehaviore);

        this.td.step(observation,actionBehaviore,time); // step learning algorithm
        return actionBehaviore;
    }

    @Override
    public ObservationSpace getObservationSpace() {
        return this.observationSpace;
    }

    @Override
    public ExperienceReplay<A> getExperienceReplay() {
        return null;
    }

    @Override
    public void stop() {
        this.td.getApproximator().stop();
    }

    private void initMlp(long seed){
        this.mlp =new Mlp(observationSpace.getShape()[0],actionSpace.getSize(),seed);
        this.mlp.setEpsilon(false);
        this.mlp.setLearning_rate(conf.getLearning_rate());
        this.mlp.setNumLayers(conf.getNumLayers());
        this.mlp.setNumNodes(conf.getNumHiddenNodes());
        this.mlp.setListener(true);
        this.mlp.init() ;// en dernier
        //,false,conf.getLearning_rate(),conf.getNumLayers(),conf.getNumHiddenNodes(),true,false) ;

    }
}
