package fr.univlyon1.actorcritic;

import fr.univlyon1.actorcritic.policy.Policy;
import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.Observation;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.networks.Approximator;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.xml.bind.JAXBException;

public interface Learning<A> {
    void init();
    void putReward(Double reward);
    A getAction(Observation observation, Double time);

    Configuration getConf();
    ObservationSpace getObservationSpace();
    ExperienceReplay<A> getExperienceReplay();
    Approximator getApproximator();
    Approximator getModelApproximator();
    Policy getPolicy();
    ActionSpace<A> getActionSpace();
    void stop();
}
