package fr.univlyon1.selfsupervised;

import fr.univlyon1.agents.AgentDRL;
import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.configurations.ListPojo;
import fr.univlyon1.configurations.PojoInteraction;
import fr.univlyon1.configurations.SupervisedConfiguration;
import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.interactions.Replayable;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.Observation;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.environment.space.SpecificObservation;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.memory.ObservationsReplay.SpecificObservationReplay;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.selfsupervised.modelLearning.Learner;
import fr.univlyon1.selfsupervised.modelLearning.LstmMlpLearner;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import java.io.File;
import java.util.Collection;

public class ModelLearner<A> implements PomdpLearner<A> {

    protected Configuration configuration ;
    protected ExperienceReplay<A> experienceReplay ;
    protected ExperienceReplay<A> notifications;
    protected Learner learner ;
    protected ActionSpace<A> actionSpace ;
    protected ObservationSpace observationSpace ;

    protected int iterations;
    protected int learn ;
    protected SupervisedConfiguration supervisedConfiguration ;
    protected long seed ;


    public ModelLearner(Approximator commonApproximator, Configuration configuration, SupervisedConfiguration supervisedConfiguration, ExperienceReplay<A> ep, ActionSpace<A> actionSpace, ObservationSpace observationSpace, long seed){
        this.experienceReplay = ep ;
        this.notifications = new SpecificObservationReplay<A>(configuration.getSizeExperienceReplay(),supervisedConfiguration.getReadfile());
        this.experienceReplay.load(actionSpace);
        this.notifications.load(actionSpace);

        this.configuration =  configuration ;
        this.iterations = configuration.getIterations() ;
        this.learn = configuration.getLearn() ;
        this.actionSpace = actionSpace ;
        this.observationSpace = observationSpace ;
        this.supervisedConfiguration = supervisedConfiguration ;
        this.seed = seed;
        this.learner =new LstmMlpLearner<A>(commonApproximator,supervisedConfiguration,ep,notifications,configuration,actionSpace,observationSpace,seed);
    }


    @Override
    public void step() {
        if(AgentDRL.getCount()%learn==0){
            for(int i = 0; i < this.iterations ; i++){
                this.learner.learn();
            }
        }
    }

    @Override
    public void notify(Observation observation) {
        assert(observation instanceof SpecificObservation);
        this.notifications.addInteraction((SpecificObservation<A>)observation);
    }

    @Override
    public void stop() {
        this.learner.stop();
    }

}
