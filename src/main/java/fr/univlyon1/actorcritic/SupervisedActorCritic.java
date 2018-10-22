package fr.univlyon1.actorcritic;

import fr.univlyon1.agents.AgentDRL;
import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.Observation;
import fr.univlyon1.environment.space.ObservationSpace;
import org.nd4j.linalg.api.ndarray.INDArray;

public class SupervisedActorCritic<A> extends ContinuousActorCritic<A> {
    public SupervisedActorCritic(ObservationSpace observationSpace, ActionSpace<A> actionSpace, Configuration conf, long seed) {
        super(observationSpace, actionSpace, conf, seed);
        if(conf.getReadfile() == null || conf.getReadfile().isEmpty()){
            System.out.println("Il manque un fichier de chargement");
        }
        this.policy = this.policyApproximator ;
        this.learn();
    }

    public void learn(){
        for(int i = 0;i < 50000 ; i++){
            this.td.learn();
            this.countStep++;
            if (this.countStep == this.epoch) {
                countStep = 0;
                this.td.epoch();
                //System.out.println("An epoch : "+ AgentDRL.getCount());
            }
        }
    }

    @Override
    public A getAction(Observation observation, Double time) {
        INDArray input = observation.getData() ;
        A actionBehaviore;
        this.td.evaluate(observation,this.reward,time);
        this.td.learn();
        INDArray resultBehaviore = this.td.behave(input);//INDArray)this.policy.getAction(input);
        actionBehaviore = this.actionSpace.mapNumberToAction(resultBehaviore);
        this.td.step(observation,actionBehaviore,time);
        return actionBehaviore;
    }
}
