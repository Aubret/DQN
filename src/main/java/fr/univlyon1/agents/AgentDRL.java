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
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.Unmarshaller;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Main class of an agent
 * @param <A>
 */
@Slf4j
public class AgentDRL<A> implements AgentRL<A> {
    protected static int count = 0 ;
    //protected static String filename = "a6_rewards63";
    //protected static String filename = "simple_onrampNoAction";
    protected static String filename = "simple_onramp_1500-700";
    protected static boolean writeFile = false;
    protected PrintWriter rewardResults ;
    protected PrintWriter rewardLearningResults ;


    protected Double previousTime ;
    protected A action ;

    protected ActionSpace<A> actionSpace ;
    protected Learning<A> learning;
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

        this.learning = new LstmActorCritic<A>(observationSpace,actionSpace,this.configuration,seed);
        //this.learning = new ConstantActor<A>(observationSpace,actionSpace,this.configuration,seed);
        //this.learning = new RandomActor<A>(observationSpace,actionSpace,this.configuration,seed);
        //this.learning = new ContinuousActorCritic<A>(observationSpace,actionSpace,this.configuration,seed);
        this.rewardShaping = new NstepTime(this.configuration);
        this.learning.init();
        if(this.print) {
            try {
                //FileWriter fw = new FileWriter("sim/arthur/continuous_rewards_baseline.csv");
                File fw = new File("sim/arthur/results/"+filename+".csv");
                //FileWriter fw = new FileWriter("sim/arthur/results/a6_baseline.csv");
                File fw2 = new File("sim/arthur/results/"+filename+"_reward.csv");
                this.rewardResults = new PrintWriter(fw);
                this.rewardLearningResults = new PrintWriter(fw2);
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
        A action = this.learning.getAction(observation,time);
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
            log.info(action.toString());
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
        A action = this.learning.getAction(observation,time);
        for(PomdpLearner pomdpLearner : this.pomdpLearners){
            pomdpLearner.step();
        }
        //A action = this.actionSpace.mapNumberToAction(Nd4j.create(new double[]{-1.,1.}));
        //A action = this.learning.getActionSpace().mapNumberToAction(0);
        this.action = action ;
        if(rewardTime != null ){
            if(this.print) {
                if(action instanceof ContinuousAction /*&& this.learning instanceof ContinuousActorCritic*/) {
                    //------------REward and action------------
                    INDArray res = ((INDArray) this.actionSpace.mapActionToNumber(action));
                    //TD td = ((TD)(((ContinuousActorCritic)this.learning).getTd()));
                    //Double qvalue = td.getQvalue();
                    String str ="";
                    for(int i = 0 ; i < res.size(1) ; i++){
                        str+=";"+res.getDouble(i);
                    }
                    /*Double score =((ContinuousActorCritic)this.learning).getScore() ;
                    str+=";"+ (score == null ? 0 : score) ;*/
                    String inputs ="";
                    for(int i = 0 ; i < data.size(1) ; i++){
                        inputs=inputs.concat(";"+data.getDouble(i));
                    }

                    for (int i = 0; i < evaluation.size(); i++) {
                        inputs = inputs.concat(";" + evaluation.get(i));
                    }

                    this.rewardResults.println(count +";"+evaluation.get(0)+str+inputs);

                    for(Map.Entry<Double,Double> ite : rewardTime.entrySet()){
                        this.rewardLearningResults.println(ite.getKey()+";"+ite.getValue());
                    }

                }else {
                    this.rewardResults.println(count + ";" + evaluation);
                }
            }
        }

        //action = actionSpace.mapNumberToAction(Nd4j.create(new double[]{-1,-1}));

        if(action instanceof ContinuousAction)
            ((ContinuousAction) action).unNormalize();
        if(count % 500== 0)
            log.info(action.toString());
        this.previousTime=time ;
        return action ;
    }

    public void notify(Observation observation){
        for(int i = 0 ; i < this.pomdpLearners.size(); i++){
            this.pomdpLearners.get(i).notify(observation);
            //System.out.println(((SpecificObservation)observation).getLabels());
        }
    }

    @Override
    public void stop() {
        this.learning.stop();
        for(int i = 0 ; i < this.pomdpLearners.size(); i++){
            this.pomdpLearners.get(i).stop();
        }
        log.info("Nombre de décisions : "+count);
        log.info("Total reward : "+this.totalReward);
        log.info("Last action : "+this.action);
        TimeUnit t = TimeUnit.SECONDS ;
        log.info("Time : "+t.convert(System.currentTimeMillis() - time,TimeUnit.SECONDS));
        if(this.print) {
            this.rewardResults.close();
            this.rewardLearningResults.close();
        }
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
