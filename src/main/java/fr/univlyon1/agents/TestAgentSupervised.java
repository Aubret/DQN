package fr.univlyon1.agents;

import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.configurations.SupervisedConfiguration;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.Observation;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.networks.LSTM;
import fr.univlyon1.networks.LSTM2D;
import fr.univlyon1.networks.LSTMMeanPooling;
import fr.univlyon1.networks.lossFunctions.LossError;
import fr.univlyon1.selfsupervised.ModelLearner;
import fr.univlyon1.selfsupervised.PomdpLearner;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import scala.App;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.Unmarshaller;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;

/*
 Agent qui load un fichier et qui exerce le lstm dessus
 */
public class TestAgentSupervised<A> implements AgentRL<A> {
    private PomdpLearner<A> pomdpLearners;
    private Configuration configuration ;
    private SupervisedConfiguration supervisedConfiguration ;
    private ObservationSpace observationSpace ;
    private ActionSpace<A> actionSpace ;
    private long seed ;

    public TestAgentSupervised(ActionSpace<A> actionSpace, ObservationSpace observationSpace, long seed){
        AgentDRL.writeFile = false ;
        this.actionSpace = actionSpace ;
        this.observationSpace = observationSpace;
        this.seed = seed ;
        try {
            JAXBContext context = JAXBContext.newInstance(Configuration.class);
            Unmarshaller unmarshaller = context.createUnmarshaller();
            //String f = "resources/learning/ddpg.xml";
            //String f = "resources/learning/justhour_ddpg.xml";
            String f = "resources/learning/lstm.xml";
            this.configuration = (Configuration)unmarshaller.unmarshal( new File(f));
        }catch(Exception e){
            e.printStackTrace();
        }
        try {
            JAXBContext context = JAXBContext.newInstance(SupervisedConfiguration.class);
            Unmarshaller unmarshaller = context.createUnmarshaller();
            String f = "resources/learning/supervised_conf.xml";
            this.supervisedConfiguration = (SupervisedConfiguration)unmarshaller.unmarshal( new File(f));
        }catch(Exception e){
            e.printStackTrace();;
        }

        Approximator approx = initLstm();
        ExperienceReplay<A> ep = new SequentialExperienceReplay<A>(configuration.getSizeExperienceReplay(), configuration.getReadfile(), configuration.getForwardTime(), configuration.getBackpropTime(),seed, configuration.getForward());
        this.pomdpLearners = new ModelLearner<A>(approx,configuration,supervisedConfiguration,ep,actionSpace,observationSpace,seed);
    }

    public Approximator initLstm(){
        //LSTM2D lstm = new LSTM2D(observationSpace.getShape()[0]+this.actionSpace.getSize(),this.configuration.getNumLstmOutputNodes(),seed);
        LSTMMeanPooling lstm = new LSTMMeanPooling(observationSpace.getShape()[0]+this.actionSpace.getSize(),this.configuration.getNumLstmOutputNodes(),seed);
        lstm.setLearning_rate(configuration.getLearning_rateLstm());
        lstm.setListener(true);
        lstm.setNumNodesPerLayer(configuration.getLayersLstmHiddenNodes());
        lstm.setNumLayers(configuration.getNumLstmlayers());
        lstm.setNumNodes(configuration.getNumLstmHiddenNodes());
        lstm.setUpdater(new Adam(configuration.getLearning_rateLstm()));
        lstm.setEpsilon(false);
        lstm.setMinimize(true);
        lstm.setLossFunction(new LossError());
        lstm.setHiddenActivation(Activation.TANH);
        lstm.setLastActivation(Activation.TANH);
        lstm.setExportModel("resources/models/lstm");
        //this.observationApproximator.setL2(0.001);
        lstm.init() ;
        return lstm ;
    }

    @Override
    public A control(Double reward, Observation observation, Double dt) {
        return null;
    }

    @Override
    public A control(HashMap<Double, Double> reward, ArrayList<Double> evaluation, Observation observation, Double dt) {
        return null;
    }

    @Override
    public void notify(Observation observation) {
        this.pomdpLearners.step();
    }

    @Override
    public void stop() {
        this.pomdpLearners.stop();
    }
}
