package main.java.fr.univlyon1.agents;

import main.java.fr.univlyon1.actorcritic.Learning;
import main.java.fr.univlyon1.environment.ActionSpace;
import main.java.fr.univlyon1.environment.Observation;
import main.java.fr.univlyon1.environment.ObservationSpace;
import main.java.fr.univlyon1.actorcritic.DQNActor;

import java.io.FileWriter;
import java.io.PrintWriter;

/**
 * Classe principale de l'agent
 * @param <A>
 * @param <O>
 */
public class DqnAgent<A,O> implements AgentRL<A> {
    private static int count = 0 ;

    private ActionSpace<A> actionSpace ;
    private ObservationSpace observationSpace;
    private Learning<A> learning;
    private long seed ;
    private PrintWriter rewardResults ;
    private double totalReward = 0 ;
    private double waitTotalReward =0 ;

    public DqnAgent(ActionSpace<A> actionSpace, ObservationSpace observationSpace, long seed) {
        this.seed = seed ;
        this.actionSpace = actionSpace ;
        this.observationSpace = observationSpace ;
        this.learning = new DQNActor<A>(observationSpace,actionSpace,seed);
        try {
            FileWriter fw = new FileWriter("sim/arthur/rewards2.csv");
            this.rewardResults = new PrintWriter(fw);
        }catch(Exception e){
            e.printStackTrace();
        }
    }

    @Override
    public Object control(Double reward,Observation observation) {
        count++ ;
        if(reward != null) {
            if(count > 200)
                this.waitTotalReward+=reward ;
            this.rewardResults.println(count+";"+reward);
            this.totalReward+=reward ;
            this.learning.putReward(reward);
        }
        A action = this.learning.getAction(observation.getData());
        //System.out.println(this.actionSpace.mapNumberToAction(0));
        //return this.actionSpace.mapNumberToAction(0);
        //System.out.println(action);
        return action ;
    }

    @Override
    public void stop() {
        this.learning.stop();
        System.out.println("Nombre de décisions : "+count);
        System.out.println("Total reward : "+this.totalReward);
        System.out.println("Attend Total reward : "+this.waitTotalReward);
        this.rewardResults.close();
    }

    public static int getCount() {
        return count;
    }

    public static void setCount(int count) {
        DqnAgent.count = count;
    }
}
