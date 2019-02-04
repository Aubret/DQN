package fr.univlyon1.agents;

import fr.univlyon1.configurations.SupervisedConfiguration;
import fr.univlyon1.environment.space.*;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.selfsupervised.ModelLearner;
import fr.univlyon1.selfsupervised.SaveLearner;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.Unmarshaller;
import java.io.File;
import java.io.PrintWriter;

/**
 * Class used for self-supervised learning, no more used
 * @param <A>
 */
public class SelfSupervisedAgentDRL<A> extends AgentDRL<A> {

    SupervisedConfiguration supervisedConfiguration ;

    public SelfSupervisedAgentDRL(ActionSpace<A> actionSpace, ObservationSpace observationSpace, long seed) {
        super(actionSpace, observationSpace, seed);
        try {
            JAXBContext context = JAXBContext.newInstance(SupervisedConfiguration.class);
            Unmarshaller unmarshaller = context.createUnmarshaller();
            String f = "resources/learning/supervised_conf.xml";
            this.supervisedConfiguration = (SupervisedConfiguration)unmarshaller.unmarshal( new File(f));
        }catch(Exception e){
            e.printStackTrace();;
        }
        Approximator modelLearner = this.learning.getModelApproximator();
        assert(modelLearner != null);
        //this.pomdpLearners.add(new ModelLearner<A>(modelLearner,this.configuration, this.supervisedConfiguration, this.learning.getExperienceReplay(),this.actionSpace,observationSpace,seed));
        this.pomdpLearners.add(new SaveLearner<A>(this.configuration,this.supervisedConfiguration,this.actionSpace));

    }
}
