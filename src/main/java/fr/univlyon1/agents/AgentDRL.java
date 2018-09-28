package fr.univlyon1.agents;

import fr.univlyon1.actorcritic.*;
import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.environment.space.*;
import fr.univlyon1.learning.TD;
import fr.univlyon1.learning.TDActorCritic;
import fr.univlyon1.reward.NstepTime;
import fr.univlyon1.reward.RewardSMDP;
import fr.univlyon1.reward.RewardShaping;
import fr.univlyon1.selfsupervised.PomdpLearner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.Unmarshaller;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;

/**
 * Classe principale de l'agent
 * @param <A>
 */
public class AgentDRL<A> implements AgentRL<A> {
    protected static int count = 0 ;
    protected static String filename = "a6_rewards63";
    protected static boolean writeFile = true ;


    protected Double previousTime ;
    protected A action ;

    protected ActionSpace<A> actionSpace ;
    protected Learning<A> learning;
    protected PrintWriter rewardResults ;
    protected double totalReward = 0 ;
    protected boolean print =true ;
    protected Configuration configuration ;
    protected RewardShaping rewardShaping;

    protected ArrayList<PomdpLearner<A>> pomdpLearners ;

    protected long time ;

    public AgentDRL(ActionSpace<A> actionSpace, ObservationSpace observationSpace, long seed){
        this.time = System.currentTimeMillis();
        //Nd4j.getMemoryManager().setAutoGcWindow(5000);
        //Nd4j.getMemoryManager().togglePeriodicGc(false);
        this.actionSpace = actionSpace ;
        this.pomdpLearners = new ArrayList<>();
        Nd4j.getRandom().setSeed(seed);
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
        // 43-44avec cheat
        // 50 sans cheat 0.5 learning rate
        // 55-56-57 sans cheat fonctionne bien du monis normalement
        this.learning = new LstmActorCritic<A>(observationSpace,actionSpace,this.configuration,seed);
        //this.learning = new ContinuousActorCritic<A>(observationSpace,actionSpace,this.configuration,seed);
        //this.rewardShaping = new NstepTime(this.configuration);
        this.rewardShaping = new NstepTime(this.configuration);
        this.learning.init();
        if(this.print) {
            try {
                //FileWriter fw = new FileWriter("sim/arthur/continuous_rewards_baseline.csv");
                File fw = new File("sim/arthur/results/"+filename+".csv");
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
    public A control(HashMap<Double,Double> rewardTime, ArrayList<Double> evaluation, Observation observation, Double time) {
        //System.out.println(observation+" ---> "+reward);
        count++ ;
        Double reward=null;
        if(rewardTime != null) {
            reward = this.rewardShaping.constructReward(rewardTime,this.previousTime,time);
            this.learning.putReward(reward);
        }
        INDArray data = observation.getData();
        A action = this.learning.getAction(data,time);
        for(PomdpLearner pomdpLearner : this.pomdpLearners){
            pomdpLearner.step();
        }
        //A action = this.actionSpace.mapNumberToAction(Nd4j.create(new double[]{-1.,1.}));
        //A action = this.learning.getActionSpace().mapNumberToAction(0);
        this.action = action ;
        if(rewardTime != null ){
            if(this.print) {
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

                    for (int i = 0; i < evaluation.size(); i++) {
                        inputs = inputs.concat(";" + evaluation.get(i));
                    }

                    this.rewardResults.println(count +";"+evaluation.get(0)+str+inputs);
                    //------------Inputs-----------------------


                }else {
                    this.rewardResults.println(count + ";" + evaluation);
                }
            }
        }

        //action = actionSpace.mapNumberToAction(Nd4j.create(new double[]{-1,-1}));

        if(action instanceof ContinuousAction)
            ((ContinuousAction) action).unNormalize();
        if(count % 500== 0)
            System.out.println(action);
        this.previousTime=time ;
        return action ;
    }

    public void notify(Observation observation){
        for(int i = 0 ; i < this.pomdpLearners.size(); i++){
            this.pomdpLearners.get(i).notify(observation);
        }
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

    public static boolean isWriteFile() {
        return writeFile;
    }
}
