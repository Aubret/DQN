package fr.univlyon1.agents;

import fr.univlyon1.actorcritic.ContinuousActorCritic;
import fr.univlyon1.actorcritic.Learning;
import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.ContinuousAction;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.environment.space.Observation;
import fr.univlyon1.learning.TD;
import fr.univlyon1.learning.TDActorCritic;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.Unmarshaller;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.concurrent.TimeUnit;

/**
 * Classe principale de l'agent
 * @param <A>
 */
public class AgentDRL<A> implements AgentRL<A> {
    private static int count = 0 ;
    private A action ;

    private ActionSpace<A> actionSpace ;
    private ObservationSpace observationSpace;
    private Learning<A> learning;
    private long seed ;
    private PrintWriter rewardResults ;
    private double totalReward = 0 ;
    private boolean print = true ;
    private Configuration configuration ;

    private long time ;

    public AgentDRL(ActionSpace<A> actionSpace, ObservationSpace observationSpace, long seed){
        this.time = System.currentTimeMillis();
        this.seed = seed ;
        this.actionSpace = actionSpace ;
        Nd4j.getRandom().setSeed(seed);
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
                FileWriter fw = new FileWriter("sim/arthur/continuous_rewards2.csv");
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
            this.learning.putReward(reward);
        }

        A action = this.learning.getAction(observation.getData());
        //A action = this.actionSpace.mapNumberToAction(0);
        this.action = action ;
        if(reward != null ){
            if(this.print) {
                this.totalReward += reward;
                if(action instanceof ContinuousAction) {
                    INDArray res = ((INDArray) this.actionSpace.mapActionToNumber(action));
                    //TD td = ((TD)(((ContinuousActorCritic)this.learning).getTd()));
                    //Double qvalue = td.getQvalue();
                    String str ="";
                    for(int i = 0 ; i < res.size(1) ; i++){
                        str+=";"+res.getDouble(i);
                    }
                    this.rewardResults.println(count + ";" + reward + ";" + res.getDouble(1) + ";" + res.getDouble(0));
                }else
                    this.rewardResults.println(count + ";" + reward) ;
            }
        }


        if(action instanceof ContinuousAction)
            ((ContinuousAction) action).unNormalize();
        if(count % 50 == 0)
            System.out.println(action);
        return action ;
    }

    @Override
    public void stop() {
        this.learning.stop();
        System.out.println("Nombre de décisions : "+count);
        System.out.println("Total reward : "+this.totalReward);
        System.out.println("Last action : "+this.action);
        TimeUnit t = TimeUnit.SECONDS ;
        System.out.println("Time : "+t.convert(System.currentTimeMillis() - time,TimeUnit.SECONDS));
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
