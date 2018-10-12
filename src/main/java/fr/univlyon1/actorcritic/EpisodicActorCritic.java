package fr.univlyon1.actorcritic;

import fr.univlyon1.actorcritic.policy.NoisyGreedy;
import fr.univlyon1.agents.AgentDRL;
import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.Observation;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.learning.TDBatch;
import fr.univlyon1.networks.Mlp;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class EpisodicActorCritic<A> extends ContinuousActorCritic<A> {

    public EpisodicActorCritic(ObservationSpace observationSpace, ActionSpace<A> actionSpace, Configuration conf, long seed) {
        super(observationSpace, actionSpace, conf, seed);
        if(conf.getReadfile() == null || conf.getReadfile().isEmpty()){
            System.out.println("Il manque un fichier de chargement");
        }
        this.countStep = 0 ;
        //this.policy = new NoisyGreedy(conf.getNoisyGreedyStd(),conf.getNoisyGreedyMean(),seed,this.getPolicyApproximator());
    }

    public void init(){
        super.init();
        this.policy = this.policyApproximator ;
        this.learn();
    }

    public void learn(){
        ((TDBatch)this.td).setBatchSize(conf.getBatchSize());

        for(int i = 0;i < this.epoch; i++){
            this.td.learn();
        }
        ((TDBatch)this.td).setBatchSize(0);
    }

    @Override
    public A getAction(Observation observation, Double time) {
        INDArray input =observation.getData() ;
        this.td.evaluate(input, this.reward,time); //Evaluation

        //actionBehaviore = this.actionSpace.mapNumberToAction(this.actionSpace.randomAction());
        //INDArray resultBehaviore = (INDArray)this.policy.getAction(input);modif après
        INDArray resultBehaviore = this.td.behave(input);
        A actionBehaviore ;
        actionBehaviore = this.actionSpace.mapNumberToAction(resultBehaviore);
        if (this.countStep == 120) {
            this.countStep = 0;
            this.learn();
        }
        this.td.learn();
        this.td.step(observation,actionBehaviore,time); // step learning algorithm
        this.countStep++ ;
        return actionBehaviore;
    }

}

