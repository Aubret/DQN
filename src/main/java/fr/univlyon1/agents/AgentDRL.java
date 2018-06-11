package fr.univlyon1.agents;

import fr.univlyon1.actorcritic.*;
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
    private static String filename = "a6_rewards46";

    private A action ;

    private ActionSpace<A> actionSpace ;
    private ObservationSpace observationSpace;
    private Learning<A> learning;
    private PrintWriter rewardResults ;
    private double totalReward = 0 ;
    private boolean print =true ;
    private Configuration configuration ;

    private long time ;

    public AgentDRL(ActionSpace<A> actionSpace, ObservationSpace observationSpace, long seed){
        this.time = System.currentTimeMillis();
        //Nd4j.getMemoryManager().setAutoGcWindow(5000);
        //Nd4j.getMemoryManager().togglePeriodicGc(false);
        this.actionSpace = actionSpace ;
        Nd4j.getRandom().setSeed(seed);
        this.observationSpace = observationSpace ;
        try {
            JAXBContext context = JAXBContext.newInstance(Configuration.class);
            Unmarshaller unmarshaller = context.createUnmarshaller();
            //String f = "resources/learning/ddpg.xml";
            //String f = "resources/learning/justhour_ddpg.xml";
            String f = "resources/learning/lstm.xml";
            this.configuration = (Configuration)unmarshaller.unmarshal( new File(f));
        }catch(Exception e){
            e.printStackTrace();;
        }

        //1-5 mémoire intiailisée
        //6-10 sans mémoire
        //11-13 sans heure
        //14-15 une seule boucle électro magnétique
        //16-19 Deux boucles électro-magnétiques
        //20 - 22 SAns mémoire
        //23 40% de connectés
        //24 - 27 60Secondes seulement
        //27-28 30 secondes
        //29 - 30 60 secondes + 40% véhicules connectés
        //31 60 secondes, 40% véhicules, 3 voies;
        //32 100% véhicules 3v oies
        //this.learning = new DQNActor<A>(observationSpace,actionSpace,seed);
        //this.learning = new ContinuousActorCritic<A>(observationSpace,actionSpace,this.configuration,seed);
        //this.learning = new RandomActor<A>(observationSpace,actionSpace,this.configuration,seed);
        //this.learning = new SupervisedActorCritic<A>(observationSpace,actionSpace,this.configuration,seed);
        //this.learning = new EpisodicActorCritic<A>(observationSpace,actionSpace,this.configuration,seed);
        //a6 2-6 lstm test2.xprj
        //a6 7-12 correction vraie récompense moyenne
        //a6 13 - 17 Vitesse minmale avec changement output lstm, marche tjrs pas très bien sur 17
        // 18 on inaugure le nouveau experience replay priorisé sur test
        //21 nouveau paramétrage fonctionne
        //22 encore nouveau sur test3
        // 23-29 tests sur graves
        //30 - ?
        //mon 2e modèle fonctionnel ? test3.xprj / 32 - 37
        // 40 premier exemple d'adaptatio aux lanes avecc cheat seed59
        // 42
        this.learning = new LstmActorCritic<A>(observationSpace,actionSpace,this.configuration,seed);
        //this.learning = new ContinuousActorCritic<A>(observationSpace,actionSpace,this.configuration,seed);
        this.learning.init();
        if(this.print) {
            try {
                //FileWriter fw = new FileWriter("sim/arthur/continuous_rewards_baseline.csv");
                FileWriter fw = new FileWriter("sim/arthur/results/"+filename+".csv");
                //FileWriter fw = new FileWriter("sim/arthur/results/a6_baseline.csv");

                this.rewardResults = new PrintWriter(fw);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }


    }

    @Override
    public A control(Double reward,Observation observation, Double time) {
        //System.out.println(observation+" ---> "+reward);
        count++ ;
        if(reward != null) {
            this.learning.putReward(reward);
        }
        INDArray data = observation.getData();
        A action = this.learning.getAction(data,time);
        //A action = this.learning.getActionSpace().mapNumberToAction(0);
        this.action = action ;
        if(reward != null ){
            if(this.print) {
                this.totalReward += reward;
                if(action instanceof ContinuousAction && this.learning instanceof ContinuousActorCritic) {
                    //------------REward and action------------
                    INDArray res = ((INDArray) this.actionSpace.mapActionToNumber(action));
                    //TD td = ((TD)(((ContinuousActorCritic)this.learning).getTd()));
                    //Double qvalue = td.getQvalue();
                    String str ="";
                    for(int i = 0 ; i < res.size(1) ; i++){
                        str+=";"+res.getDouble(i);
                    }
                    Double score =((ContinuousActorCritic)this.learning).getScore() ;
                    str+=";"+ (score == null ? 0 : score) ;
                    String inputs ="";
                    for(int i = 0 ; i < data.size(1) ; i++){
                        inputs=inputs.concat(";"+data.getDouble(i));
                    }

                    this.rewardResults.println(count +";"+reward+str+inputs);
                    //------------Inputs-----------------------


                }else {
                    this.rewardResults.println(count + ";" + reward);
                }
            }
        }

        //action = actionSpace.mapNumberToAction(Nd4j.create(new double[]{-1,-1}));

        if(action instanceof ContinuousAction)
            ((ContinuousAction) action).unNormalize();
        if(count % 500== 0)
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

    public static String getFilename() {
        return filename;
    }

    public static void setFilename(String filename) {
        AgentDRL.filename = filename;
    }
}
