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
        this.policy = new NoisyGreedy(conf.getNoisyGreedyStd(),conf.getNoisyGreedyMean(),seed,this.getPolicyApproximator());
        this.learn();
    }

    public void learn(){
        for(int i = 0;i < this.epoch; i++){
            this.td.learn();
        }
        //this.td.setBatchSize(0);
    }

    @Override
    public A getAction(INDArray input) {
        A actionBehaviore;
        this.td.evaluate(input, this.reward); //Evaluation
        INDArray resultBehaviore = (INDArray)this.policy.getAction(input);
        actionBehaviore = this.actionSpace.mapNumberToAction(resultBehaviore);
        this.td.step(input,actionBehaviore); // step learning algorithm
        if (this.countStep == 200) {
            this.countStep = 0;
            this.learn();
            this.td.epoch();
            //System.out.println("An epoch : "+ AgentDRL.getCount());
        }
        this.countStep++ ;
        return actionBehaviore;
    }

}

