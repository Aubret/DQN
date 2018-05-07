package fr.univlyon1.actorcritic;

import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.networks.Approximator;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.xml.bind.JAXBException;

public interface Learning<A> {
    void init();
    Configuration getConf();
    A getAction(INDArray input,Double time);
    ObservationSpace getObservationSpace();
    void putReward(Double reward);
    Approximator getApproximator();
    ActionSpace<A> getActionSpace();
    void stop();
}
