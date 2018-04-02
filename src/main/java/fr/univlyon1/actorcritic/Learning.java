package fr.univlyon1.actorcritic;

import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.networks.Approximator;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface Learning<A> {
    Configuration getConf();
    A getAction(INDArray input);
    void putReward(Double reward);
    Approximator getApproximator();
    ActionSpace<A> getActionSpace();
    void stop();
}
