package fr.univlyon1.actorcritic;

import fr.univlyon1.actorcritic.policy.NoisyGreedy;
import fr.univlyon1.agents.AgentDRL;
import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.ObservationSpace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class EpisodicActorCritic<A> extends ContinuousActorCritic<A> {
    public EpisodicActorCritic(ObservationSpace observationSpace, ActionSpace<A> actionSpace, Configuration conf, long seed) {
        super(observationSpace, actionSpace, conf, seed);
        if(conf.getFile() == null){
            System.out.println("Il manque un fichier de chargement");
        }
        //this.policy = new NoisyGreedy(conf.getNoisyGreedyStd(),conf.getNoisyGreedyMean(),seed,this.getPolicyApproximator());
    }

    public void init(){
        super.init();
        this.policy = this.policyApproximator ;
        this.learn();
    }

    public void learn(){
        for(int i = 0;i < this.epoch; i++){
            this.td.learn();
            this.td.epoch();
        }
    }

    @Override
    public A getAction(INDArray input,Double time) {
        A actionBehaviore;
        this.td.evaluate(input, this.reward); //Evaluation
        INDArray resultBehaviore = (INDArray)this.policy.getAction(input);
        actionBehaviore = this.actionSpace.mapNumberToAction(resultBehaviore);
        if (this.countStep == 100) {
            this.countStep = 0;
            this.learn();
            //System.out.println("An epoch : "+ AgentDRL.getCount());
        }
        this.td.learn();
        this.td.step(input,actionBehaviore,time); // step learning algorithm
        this.countStep++ ;
        return actionBehaviore;
    }

}

