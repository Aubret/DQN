package fr.univlyon1.agents;

import fr.univlyon1.actorcritic.ContinuousActorCritic;
import fr.univlyon1.actorcritic.Learning;
import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.environment.ActionSpace;
import fr.univlyon1.environment.ObservationSpace;
import fr.univlyon1.environment.Observation;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.Unmarshaller;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

/**
 * Classe principale de l'agent
 * @param <A>
 */
public class AgentDRL<A> implements AgentRL<A> {
    private static int count = 0 ;

    private ActionSpace<A> actionSpace ;
    private ObservationSpace observationSpace;
    private Learning<A> learning;
    private long seed ;
    private PrintWriter rewardResults ;
    private double totalReward = 0 ;
    private double waitTotalReward =0 ;
    private boolean print = false ;
    private Configuration configuration ;

    public AgentDRL(ActionSpace<A> actionSpace, ObservationSpace observationSpace, long seed){
        this.seed = seed ;
        this.actionSpace = actionSpace ;
        this.observationSpace = observationSpace ;
        try {
            JAXBContext context = JAXBContext.newInstance(Configuration.class);

            Unmarshaller unmarshaller = context.createUnmarshaller();
            //unmarshaller.setProperty("jaxb.encoding", "UTF-8");
            //unmarshaller.setProperty("jaxb.formatted.output", true);
            this.configuration = (Configuration)unmarshaller.unmarshal( new File("resources/learning/ddpg.xml"));
        }catch(Exception e){
            e.printStackTrace();
        }

        //this.learning = new DQNActor<A>(observationSpace,actionSpace,seed);
        this.learning = new ContinuousActorCritic<A>(observationSpace,actionSpace,this.configuration,seed);
        if(this.print) {
            try {
                FileWriter fw = new FileWriter("sim/arthur/continuous_rewards.csv");
                this.rewardResults = new PrintWriter(fw);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }


    }

    @Override
    public Object control(Double reward,Observation observation) {
        count++ ;
        if(reward != null) {
            if(count > 200)
                this.waitTotalReward+=reward ;
            if(this.print)
                this.rewardResults.println(count+";"+reward);
            this.totalReward+=reward ;
            this.learning.putReward(reward);
        }

        A action = this.learning.getAction(observation.getData());
        System.out.println(action);
        /*double[] values = new double[]{-0.2, -0.7, 0., 0.,1.} ;
        INDArray vals = Nd4j.create(values);*/
        //A action = this.actionSpace.mapNumberToAction(vals);
        //System.out.println(action);
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
        if(this.print)
            this.rewardResults.close();
    }

    public static int getCount() {
        return count;
    }

    public static void setCount(int count) {
        AgentDRL.count = count;
    }
}
