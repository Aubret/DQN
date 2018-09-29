package fr.univlyon1.selfsupervised;

import fr.univlyon1.agents.AgentDRL;
import fr.univlyon1.configurations.*;
import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.interactions.MiniObs;
import fr.univlyon1.environment.interactions.Replayable;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.Observation;
import fr.univlyon1.environment.space.SpecificObservation;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.memory.ObservationsReplay.SpecificObservationReplay;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import java.io.File;
import java.util.Collection;

public class SaveLearner<A> implements PomdpLearner<A> {
    protected ExperienceReplay<A> notifications;
    protected Configuration configuration ;
    protected SupervisedConfiguration supervisedConfiguration;
    protected ActionSpace<A> actionSpace ;

    public SaveLearner (Configuration configuration, SupervisedConfiguration supervisedConfiguration, ActionSpace<A> actionSpace){
        this.notifications = new SpecificObservationReplay<A>(configuration.getSizeExperienceReplay(),supervisedConfiguration.getFile());
        this.configuration = configuration ;
        this.supervisedConfiguration = supervisedConfiguration ;
        this.actionSpace = actionSpace ;
    }

    @Override
    public void step() {}

    @Override
    public void notify(Observation observation) {
        assert(observation instanceof SpecificObservation);
        this.notifications.addInteraction((SpecificObservation<A>)observation);
    }

    @Override
    public void stop() {
        if(AgentDRL.isWriteFile()){
            System.out.println("Saving notifications");
            ListPojoObs<A> point = new ListPojoObs<A>();
            Collection<? extends Replayable<A>> memory = this.notifications.getMemory();
            for(Replayable<A> replayable : memory){
                if(replayable instanceof SpecificObservation) {
                    SpecificObservation<A> spo = (SpecificObservation<A>)replayable ;
                    MiniObs miniObs =new MiniObs(spo);
                    point.add(new PojoSpecificObservation<A>(miniObs));
                }
            }

            try {
                JAXBContext context = JAXBContext.newInstance(ListPojoObs.class);
                Marshaller m = context.createMarshaller();
                m.marshal(point,new File(this.supervisedConfiguration.getFile().get(0)));
            } catch (JAXBException e) {
                e.printStackTrace();
            }
            System.out.println("End saving notifications");
        }
    }
}
